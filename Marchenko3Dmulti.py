#!/usr/bin/env python
# coding: utf-8

########################################################
# Marchenko redatuming with 3D dataset (multiple points)
########################################################
import warnings
warnings.filterwarnings('ignore')

import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import dask.array as da
import zarr
import pylops
import pylops_distributed

from scipy.sparse import csr_matrix, vstack
from scipy.linalg import lstsq, solve
from scipy.sparse.linalg import cg, lsqr
from scipy.signal import convolve, filtfilt
from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d

from pylops.basicoperators import *
from pylops.waveeqprocessing.mdd       import MDC
from pylops.utils.wavelets             import *
from pylops.utils.seismicevents        import *
from pylops.utils.tapers               import *
from pylops.waveeqprocessing.marchenko import directwave
from pylops.utils import dottest

from pylops_distributed.utils import dottest as ddottest
from pylops_distributed.basicoperators import Diagonal as dDiagonal
from pylops_distributed.basicoperators import Identity as dIdentity
from pylops_distributed.basicoperators import Roll as dRoll
from pylops_distributed.waveeqprocessing.mdd import MDC as dMDC
from pylops_distributed.waveeqprocessing.marchenko import Marchenko as dMarchenko
from pylops_distributed.optimization.cg import cg as dcg
from pylops_distributed.optimization.cg import cgls as dcgls



def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return v, vol


def run(subsampling, vsz, nvsx, dvsx, ovsx, nvsy, dvsy, ovsy, ivsrestart, ivsend, nvssim):
    client = pylops_distributed.utils.backend.dask(hardware='multi', client='be-linrgsn214:8786')
    client.restart()
    
    nworkers = len(np.array(list(client.ncores().values())))
    ncores = np.sum(np.array(list(client.ncores().values())))

    print('Client', client)
    print('Nworkers', nworkers)
    print('Ncores', ncores)
    print('Subsampling', subsampling)

    # Input parameters 
    vel = 2400.0        # velocity
    toff = 0.045        # direct arrival time shift
    nsmooth = 10        # time window smoothing 
    nfmax = 300         # max frequency for MDC (#samples)
    nstaper = 11        # source/receiver taper lenght
    n_iter = 10         # iterations

    inputfile_aux = os.environ["STORE_PATH"] + '3DMarchenko_auxiliary_2.npz' 
    zarrfile = os.environ["STORE_PATH"] + 'input3D_sub%d_ffirst.zarr' % subsampling


    # Load input
    inputdata_aux = np.load(inputfile_aux)

    # Receivers
    r = inputdata_aux['recs'][::subsampling].T
    nr = r.shape[1]
    dr = r[0,1]-r[0,0]

    # Sources
    s = inputdata_aux['srcs'][::subsampling].T
    ns = s.shape[1]
    ds = s[0,1]-s[0,0]

    # Virtual points
    vsy = np.arange(nvsy) * dvsy + ovsy 
    vsx = np.arange(nvsx) * dvsx + ovsx 
    VSX, VSY = np.meshgrid(vsx, vsy, indexing='ij')
    VSX, VSY = VSX.ravel(), VSY.ravel()

    # Density model
    rho = inputdata_aux['rho']
    z, x, y = inputdata_aux['z'], inputdata_aux['x'], inputdata_aux['y']
    
    # Time axis
    ot, dt, nt = 0, 2.5e-3, 601
    t = np.arange(nt)*dt
    
    # Display geometry
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 6))      
    ax.scatter(r[0], r[1], marker='.', s=200, c='r', edgecolors='k', label='Srcs-Recs')
    ax.scatter(VSX.ravel(), VSY.ravel(), marker='.', s=300, c='g', edgecolors='k', label='VS')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Geometry')
    plt.legend()
    plt.close()

    # Identify areal extent of each source-receiver to be used in MDC integral
    vertex, vols = voronoi_volumes(r[:2].T)
    darea = np.min(np.unique(vols))
    print('Integration area %f' % darea)

    # Read data
    dRtwosided_fft = 2 * np.sqrt(2 * nt - 1) * dt * darea * da.from_zarr(zarrfile)  # 2 * as per theory you need 2*R
    nchunks = [max(nfmax // (nworkers + 1), 1), ns, nr]
    dRtwosided_fft = dRtwosided_fft.rechunk(nchunks)
    dRtwosided_fft = client.persist(dRtwosided_fft)
    client.rebalance(dRtwosided_fft)
    print('Number of chunks', np.prod(dRtwosided_fft.numblocks))
    print('Chunks', dRtwosided_fft.chunks)

    # Read wavelet
    wav = ricker(t[:51], 20)[0]
    wav_c = np.argmax(wav)
    
    # Initialize output files where Green's functions are saved
    nvs_batch = 4
    nr_batch = 4
    gplus_filename = 'Gplus_multi_sub%d.zarr' % subsampling
    gminus_filename = 'Gminus_multi_sub%d.zarr' % subsampling
    gdir_filename = 'Gdir_multi_sub%d.zarr' % subsampling
    grtm_filename = 'Grtm_multi_sub%d.zarr' % subsampling
    gplus_filepath = os.environ["STORE_PATH"]+gplus_filename
    gminus_filepath = os.environ["STORE_PATH"]+gminus_filename
    gdir_filepath = os.environ["STORE_PATH"]+gdir_filename
    grtm_filepath = os.environ["STORE_PATH"]+grtm_filename
    
    if np.prod(np.array([nt, nr // nr_batch, (nvsy * nvsx) // nvs_batch])) * 4 > 2147483647:
            raise ValueError('Zarr file chunks too big for BLOSC Codec, increase number of nvs_batch and/or nr_batch')
    Gplus = zarr.open_array(gplus_filepath, mode='a', 
                            shape=(nt, nr, nvsy * nvsx), 
                            chunks=(nt, nr // nr_batch, (nvsy * nvsx) // nvs_batch),
                            #compressor=None,
                            synchronizer=zarr.ThreadSynchronizer(),
                            dtype=np.float32)
    Gminus = zarr.open_array(gminus_filepath, mode='a', 
                             shape=(nt, nr, nvsy * nvsx), 
                             chunks=(nt, nr // nr_batch, (nvsy * nvsx) // nvs_batch), 
                             #compressor=None,
                             synchronizer=zarr.ThreadSynchronizer(),
                             dtype=np.float32)
    Gdir = zarr.open_array(gdir_filepath, mode='a', 
                           shape=(nt, nr, nvsy * nvsx), 
                           chunks=(nt, nr // nr_batch, (nvsy * nvsx) // nvs_batch), 
                           #compressor=None,
                           synchronizer=zarr.ThreadSynchronizer(),
                           dtype=np.float32)
    Grtm = zarr.open_array(grtm_filepath, mode='a', 
                           shape=(nt, nr, nvsy * nvsx), 
                           chunks=(nt, nr // nr_batch, (nvsy * nvsx) // nvs_batch), 
                           #compressor=None,
                           synchronizer=zarr.ThreadSynchronizer(),
                           dtype=np.float32)
    print(Gplus.info, Gminus.info)

    # Common operator
    MarchenkoWM = dMarchenko(dRtwosided_fft, nt=nt, dt=dt, dr=darea, wav=wav,
                             toff=toff, nsmooth=nsmooth, saveRt=False, 
                             prescaled=True, dtype='float32')

    for ivs in range(ivsrestart, ivsend, nvssim):
        t0 = time.time()
        
        # Virtual point (x, y, z)
        vs = np.vstack((VSX[ivs:ivs + nvssim], VSY[ivs:ivs + nvssim], vsz*np.ones(nvssim)))

        # Create window
        distVS = np.sqrt((vs[0]-r[0][:, np.newaxis])**2 + (vs[1]-r[1][:, np.newaxis])**2 + (vs[2]-r[2][:, np.newaxis])**2)

        directVS = distVS/vel
        directVS_off = directVS - toff

        idirectVS_off = np.round(directVS_off/dt).astype(np.int)
        w = np.zeros((nr, nvssim, nt))
        for ir in range(nr):
            for ivsw in range(nvssim):
                w[ir, ivsw, :idirectVS_off[ir, ivsw]]=1            
        w = np.concatenate((np.flip(w, axis=-1), w[:,:, 1:]), axis=-1)
        if nsmooth>0:
            smooth=np.ones(nsmooth)/nsmooth
            w  = filtfilt(smooth, 1, w)    

        # Create analytical direct wave
        G0sub = np.zeros((nr, nvssim, nt))
        for ivsg0 in range(nvssim):
            G0sub[:, ivsg0] = directwave(wav, directVS[:,ivsg0], nt, dt, nfft=2**11, dist=distVS[:,ivsg0], kind='3d', derivative=False).T
        # Differentiate to get same as FD modelling
        G0sub = np.diff(G0sub, axis=-1)
        G0sub = np.concatenate([G0sub, np.zeros((nr, nvssim, 1))], axis=-1)     

        # Ensure w and G0sub_ana is float32
        G0sub = G0sub.astype(np.float32)
        w = w.astype(np.float32)

        # Inversion
        df1_inv_minus, df1_inv_plus, dp0_minus, dg_inv_minus, dg_inv_plus = \
            MarchenkoWM.apply_multiplepoints(directVS, G0=G0sub, nfft=2**11, rtm=True, greens=True,
                                             dottest=False, **dict(niter=n_iter, tol=0, client=client))

        # Remove acausal artefacts in upgoing Green's functions
        dg_inv_minus = dg_inv_minus * (1-w)            

        # Save Green's functions
        Gplus[:, :, ivs:ivs+nvssim] = (np.transpose(dg_inv_plus[:, :, nt-1:], (2, 0, 1))).astype(np.float32)
        Gminus[:, :, ivs:ivs+nvssim] = (np.transpose(dg_inv_minus[:, :, nt-1:], (2, 0, 1))).astype(np.float32)
        Gdir[:, :, ivs:ivs+nvssim] = (np.transpose(G0sub, (2, 0, 1))).astype(np.float32)
        Grtm[:, :, ivs:ivs+nvssim] = (np.transpose(dp0_minus[:, :, nt-1:], (2, 0, 1))).astype(np.float32)
        print('Working with points', ivs, '-', ivs+nvssim, '.... Excecution time: ', time.time() - t0, ' s')
        
    client.close()


if __name__ == '__main__':
    subsampling = int(sys.argv[1])
    vsz = float(sys.argv[2]) # depth of line
    nvsx = int(sys.argv[3]) # number of samples along x-line
    dvsx = float(sys.argv[4]) # sampling of x-line
    ovsx = float(sys.argv[5]) # origin of x-line
    nvsy = int(sys.argv[6]) # number of samples along y-line
    dvsy = float(sys.argv[7]) # sampling of y-line
    ovsy = float(sys.argv[8]) # origin of y-line
    ivsrestart = int(sys.argv[9]) # restart from virtual source with index ivsrestart
    ivsend = int(sys.argv[10]) # end at virtual source with index ivsend
    nvssim = int(sys.argv[11]) # number of virtual points to invert simultaneously
    run(subsampling, vsz, nvsx, dvsx, ovsx, nvsy, dvsy, ovsy, ivsrestart, ivsend, nvssim)






