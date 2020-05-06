#!/usr/bin/env python
# coding: utf-8

##########################################################
# Multi-dimensional deconvolution with 3D Marchenko fields
##########################################################
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


def run(subsampling, ivsrestart, ivsend, nvssim, kind):
    client = pylops_distributed.utils.backend.dask(hardware='multi', client='be-linrgsn214:8786')
    client.restart()
    
    nworkers = len(np.array(list(client.ncores().values())))
    ncores = np.sum(np.array(list(client.ncores().values())))
    print('Nworkers', nworkers)
    print('Ncores', ncores)
    
    t0 = time.time()

    # Input parameters 
    nfmax = 300 # max frequency for MDC (#samples)
    n_iter = 10 # iterations
    
    inputfile_aux = os.environ["STORE_PATH"] + '3DMarchenko_auxiliary_2.npz' 

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
    vsz = 650
    nvsx = 71
    dvsx = 20
    ovsx = 200
    nvsy = 41
    dvsy = 20
    ovsy = 200
    nvs = nvsx * nvsy
    vsy = np.arange(nvsy) * dvsy + ovsy 
    vsx = np.arange(nvsx) * dvsx + ovsx 
    VSX, VSY = np.meshgrid(vsx, vsy, indexing='ij')
    VSX, VSY = VSX.ravel(), VSY.ravel()
    
    # Time axis
    ot, dt, nt = 0, 2.5e-3, 601
    t = np.arange(nt)*dt

    # Density model
    rho = inputdata_aux['rho']
    z, x, y = inputdata_aux['z'], inputdata_aux['x'], inputdata_aux['y']

    # Display geometry
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(13, 11))      
    ax.scatter(r[0], r[1], marker='.', s=200, c='r', edgecolors='k', label='Srcs-Recs')
    ax.scatter(VSX, VSY, marker='.', s=500, c='g', edgecolors='k', label='VS')
    ax.scatter(VSX.ravel()[ivsrestart:ivsend], 
               VSY.ravel()[ivsrestart:ivsend], marker='.', s=500, 
               c='y', edgecolors='k', label='VS selected')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Geometry')
    plt.legend()

    fig, ax1 = plt.subplots(1, 1, figsize=(13, 6))
    ax1.imshow(rho[np.argmin(np.abs(y-VSY[nvsy//2]))].T, cmap='gray', vmin=1000, vmax=5000,
               extent = (x[0], x[-1], z[-1], z[0]))
    ax1.axhline(r[2, 0], color='b', lw=4)
    ax1.axhline(s[2, 0], color='r', linestyle='--', lw=4)
    ax1.scatter(vsx, vsz * np.ones(nvsx), marker='.', s=400, c='g', edgecolors='k')
    ax1.axis('tight')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('z [m]')
    ax1.set_xlim(x[0], x[-1])
    plt.show()

    # Load input data   
    if kind=='Mck':
        Gplus = da.from_zarr(os.environ["STORE_PATH"] + 'Gplus_sub%d.zarr' % subsampling)
        Gminus = da.from_zarr(os.environ["STORE_PATH"] + 'Gminus_sub%d.zarr' % subsampling)
    else:
        Gplus = da.from_zarr(os.environ["STORE_PATH"] + 'Gdir_sub%d.zarr' % subsampling)
        Gminus = da.from_zarr(os.environ["STORE_PATH"] + 'Grtm_sub%d.zarr' % subsampling)
    print(Gplus, Gminus)

    # Read data
    nchunks_in = [nt, nr, nvs // nworkers]
    nchunks = [max(nfmax // (nworkers + 1), 1), nr, nvs]
    
    Gplus = Gplus.rechunk(nchunks_in)
    Gplus = da.concatenate((da.zeros((nt-1, nr, nvs), dtype=np.float32), Gplus), axis=0)
    Gplus = Gplus.rechunk([2*nt-1, nchunks_in[1], nchunks_in[2]])
    Gplus = da.fft.rfft(Gplus, 2*nt-1, axis=0)
    Gplus = Gplus[:nfmax]
    Gplus = Gplus.rechunk(nchunks)
    Gplus = client.persist(Gplus)  
    client.rebalance(Gplus)
    print('Done loading data...')
    print('Number of chunks', np.prod(Gplus.numblocks))
    print('Chunks', Gplus.chunks)
   
    # Initialize output files where Green's functions are saved
    nvs_batch = 4
    Radj_filename = 'R_%s_adj_sub%d.zarr' % (kind, subsampling)
    Rinv_filename = 'R_%s_inv_sub%d.zarr' % (kind, subsampling)
    Radj_filepath = os.environ["STORE_PATH"]+Radj_filename
    Rinv_filepath = os.environ["STORE_PATH"]+Rinv_filename

    Radj = zarr.open_array(Radj_filepath, mode='a', 
                           shape=(nt, nvsy * nvsx, nvsy * nvsx), 
                           chunks=(nt, (nvsy * nvsx) // nvs_batch, (nvsy * nvsx) // nvs_batch),
                           #compressor=None,
                           synchronizer=zarr.ThreadSynchronizer(),
                           dtype=np.float32)
    Rinv = zarr.open_array(Rinv_filepath, mode='a', 
                           shape=(nt, nvsy * nvsx, nvsy * nvsx), 
                           chunks=(nt, (nvsy * nvsx) // nvs_batch, (nvsy * nvsx) // nvs_batch),
                           #compressor=None,
                           synchronizer=zarr.ThreadSynchronizer(),
                           dtype=np.float32)
    print(Radj.info, Rinv.info)
    
    # Create operator
    Gplusop = dMDC(Gplus, nt=2*nt-1, nv=nvssim, dt=dt,  dr=dvsx*dvsy, twosided=True)
    print('Done with preparation.... Execution time: ', time.time() - t0, ' s')

    for ivs in range(ivsrestart, ivsend, nvssim):
        t0 = time.time()        
     
        # Create data
        Gminus_vs = Gminus[:, :, ivs:ivs+nvssim]
        Gminus_vs = da.concatenate((da.zeros((nt-1, nr, nvssim), dtype=np.float32), Gminus_vs), axis=0)
        Gminus_vs = client.persist(Gminus_vs)
           
        # Adjoint
        Radj_vs = Gplusop.H * Gminus_vs.ravel()
        Radj_vs = Radj_vs.reshape(2*nt-1, nvs, nvssim)
        Radj_vs = Radj_vs.compute()

        # Inversion
        Rinv_vs = dcgls(Gplusop, Gminus_vs.ravel(), niter=n_iter, tol=0, client=client)[0]
        Rinv_vs = Rinv_vs.reshape(2*nt-1, nvs, nvssim)
        Rinv_vs = Rinv_vs.compute()
        print('Working with points', ivs, '-', ivs+nvssim, '... Excecution time: ', time.time() - t0, ' s')
                        
        # Save Reflectivities
        t0 = time.time()
        Radj[:, :, ivs:ivs+nvssim] = (Radj_vs[nt-1:]).astype(np.float32)
        Rinv[:, :, ivs:ivs+nvssim] = (Rinv_vs[nt-1:]).astype(np.float32)
        print('........................................ Saving time: ', time.time() - t0, ' s')
        
    client.close()


if __name__ == '__main__':
    subsampling = int(sys.argv[1])
    ivsrestart = int(sys.argv[2]) # restart from virtual source with index ivsrestart
    ivsend = int(sys.argv[3]) # end at virtual source with index ivsend
    nvssim = int(sys.argv[4]) # number of virtual points to invert simultaneously
    kind = str(sys.argv[5]) # kind of input data (Ss: single-scattering or Mck: Marchenko)
    run(subsampling, ivsrestart, ivsend, nvssim, kind)






