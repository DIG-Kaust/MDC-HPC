#!/usr/bin/env python
# coding: utf-8

######################################
# Marchenko redatuming with 3D dataset
######################################
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

from dask_jobqueue import PBSCluster
from dask.distributed import Client

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


def run(subsampling, vsz, nvsx, dvsx, ovsx, nvsy, dvsy, ovsy, ixrestart, ixend, iyrestart, iyend):
    nworkers = 14
    cluster = PBSCluster(cores=16,
                         memory='128GB',
                         shebang='#!/bin/bash',
                         resource_spec='nodes=1:baloo',
                         queue='normal',
                         #name='Marchenko-dask',
                         walltime='24:00:00',
                         project='account')
    cluster.scale(jobs=nworkers)
    client = Client(cluster)
    nworkers_connected = 0
    while nworkers > nworkers_connected:
        time.sleep(30)
        client.restart()
        print('Client', client)
        nworkers_connected = len(np.array(list(client.ncores().values())))
        print('Nworkers', nworkers_connected)        

    nworkers_connected = len(np.array(list(client.ncores().values())))
    ncores = np.sum(np.array(list(client.ncores().values())))
    
    print('Client', client)
    print('Dashboard',cluster.dashboard_link)
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
    if ixend == -1:
        ixend = nvsx
    if iyend == -1:
        iyend = nvsy
    vsy = np.arange(nvsy) * dvsy + ovsy 
    vsx = np.arange(nvsx) * dvsx + ovsx 
    VSX, VSY = np.meshgrid(vsx, vsy, indexing='ij')

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
    plt.show()

    # Identify areal extent of each source-receiver to be used in MDC integral
    vertex, vols = voronoi_volumes(r[:2].T)
    darea = np.min(np.unique(vols))
    print('Integration area %f' % darea)

    # Read data
    dRtwosided_fft = 2 * da.from_zarr(zarrfile)  # 2 * as per theory you need 2*R
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
    gplus_filename = 'Gplus_sub%d.zarr' % subsampling
    gminus_filename = 'Gminus_sub%d.zarr' % subsampling
    gdir_filename = 'Gdir_sub%d.zarr' % subsampling
    grtm_filename = 'Grtm_sub%d.zarr' % subsampling
    gplus_filepath = os.environ["STORE_PATH"]+gplus_filename
    gminus_filepath = os.environ["STORE_PATH"]+gminus_filename
    gdir_filepath = os.environ["STORE_PATH"]+gdir_filename
    grtm_filepath = os.environ["STORE_PATH"]+grtm_filename

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

    for ivsx, vsxp in enumerate(vsx[ixrestart:ixend], ixrestart):
        for ivsy, vsyp in enumerate(vsy[iyrestart:iyend], iyrestart):
            t0 = time.time()
            
            # Virtual point (x, y, z)
            vs = np.array([vsxp, vsyp, vsz])
            
            # Create window
            distVS = np.sqrt((vs[0]-r[0])**2 +(vs[1]-r[1])**2 +(vs[2]-r[2])**2)
            directVS = distVS/vel
            directVS_off = directVS - toff

            idirectVS_off = np.round(directVS_off/dt).astype(np.int)
            w = np.zeros((nr, nt))
            for ir in range(nr):
                w[ir, :idirectVS_off[ir]]=1            
            w = np.hstack((np.fliplr(w), w[:, 1:]))

            if nsmooth>0:
                smooth=np.ones(nsmooth)/nsmooth
                w  = filtfilt(smooth, 1, w)    

            # Create analytical direct wave
            G0sub = directwave(wav, directVS, nt, dt, nfft=2**11, dist=distVS, kind='3d') 

            # differentiate to get same as FD modelling
            G0sub = np.diff(G0sub, axis=0)
            G0sub = np.vstack([G0sub, np.zeros(nr)])

            # Inversion
            dRop = dMDC(dRtwosided_fft, nt=2*nt-1, nv=1, dt=dt, dr=darea, twosided=True,
                        saveGt=False)
            dR1op = dMDC(dRtwosided_fft, nt=2*nt-1, nv=1, dt=dt, dr=darea, twosided=True, 
                         saveGt=False, conj=True)
            dRollop = dRoll((2*nt-1) * nr,
                           dims=(2*nt-1, nr),
                           dir=0, shift=-1)

            # Input focusing function
            dfd_plus =  np.concatenate((np.fliplr(G0sub.T).T, 
                                        np.zeros((nt-1, nr))))
            dfd_plus = da.from_array(dfd_plus)

            dWop = dDiagonal(w.T.flatten())
            dIop = dIdentity(nr*(2*nt-1))

            dMop = pylops_distributed.VStack([pylops_distributed.HStack([dIop, -1*dWop*dRop]),
                                             pylops_distributed.HStack([-1*dWop*dRollop*dR1op, dIop])])*pylops_distributed.BlockDiag([dWop, dWop])
            dGop = pylops_distributed.VStack([pylops_distributed.HStack([dIop, -1*dRop]),
                                             pylops_distributed.HStack([-1*dRollop*dR1op, dIop])])


            # Run standard redatuming as benchmark
            dp0_minus = dRop * dfd_plus.flatten()
            dp0_minus = dp0_minus.reshape((2*nt-1), nr).T

            # Create data
            dd = dWop*dRop*dfd_plus.flatten()
            dd = da.concatenate((dd.reshape(2*nt-1, nr), da.zeros((2*nt-1, nr))))

            # Inverse focusing functions
            df1_inv = dcgls(dMop, dd.ravel(), niter=n_iter, tol=0, client=client)[0]
            df1_inv = df1_inv.reshape(2*(2*nt-1), nr)

            # Add initial guess to estimated focusing functions
            df1_inv_tot = df1_inv + da.concatenate((da.zeros((2*nt-1, nr)), dfd_plus))

            # Estimate Green's functions
            dg_inv = dGop * df1_inv_tot.flatten()
            dg_inv = dg_inv.reshape(2*(2*nt-1), nr)

            dd, dp0_minus, df1_inv_tot, dg_inv = da.compute(dd, dp0_minus, df1_inv_tot, dg_inv)
            dg_inv = np.real(dg_inv)

            # Extract up and down focusing and Green's functions from model vectors
            df1_inv_minus, df1_inv_plus = df1_inv_tot[:(2*nt-1)].T, df1_inv_tot[(2*nt-1):].T
            dg_inv_minus, dg_inv_plus =  -dg_inv[:(2*nt-1)].T, np.fliplr(dg_inv[(2*nt-1):].T)
            
            # Remove acausal artefacts in upgoing Green's functions
            dg_inv_minus = dg_inv_minus * (1-w)            

            # Save Green's functions
            #print(ivsx * nvsy + ivsy)
            Gplus[:, :, ivsx * nvsy + ivsy] = (dg_inv_plus[:, nt-1:].T).astype(np.float32)
            Gminus[:, :, ivsx * nvsy + ivsy] = (dg_inv_minus[:, nt-1:].T).astype(np.float32)
            Gdir[:, :, ivsx * nvsy + ivsy] = (G0sub).astype(np.float32)
            Grtm[:, :, ivsx * nvsy + ivsy] = (dp0_minus[:, nt-1:].T).astype(np.float32)

            print('Working with point', vs, '.... Excecution time: ', time.time() - t0, ' s')
            
            """
            # Visualization
            clip=1e-5
            
            
            fig, ax = plt.subplots(1, 1,  sharey=True, figsize=(20, 5))
            im = ax.imshow(w.T, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
            ax.plot(np.arange(0, nr), directVS_off, '--r')
            ax.plot(np.arange(0, nr), -directVS_off, '--r')
            ax.set_title('Window') 
            ax.set_xlabel(r'$x_R$')
            ax.set_ylabel(r'$t$')
            ax.axis('tight')
            ax.set_xlim(800, 1000);
            fig.colorbar(im, ax=ax);
            

            fig, axs = plt.subplots(3, 1, sharey=True, figsize=(16, 20))
            axs[0].imshow(dp0_minus.T, cmap='gray', vmin=-clip, vmax=clip, extent=(0, nr, t[-1], -t[-1]))
            axs[0].set_title(r'$p_0^-$'), axs[0].set_xlabel(r'$x_R$'), axs[0].set_ylabel(r'$t$')
            axs[0].axis('tight')
            axs[0].set_xlim(nr//2-100,nr//2+100)
            axs[0].set_ylim(1, -1)
            axs[1].imshow(df1_inv_minus.T, cmap='gray', vmin=-clip, vmax=clip, extent=(0, nr, t[-1], -t[-1]))
            axs[1].set_title(r'$f^-$'), axs[0].set_xlabel(r'$x_R$')
            axs[1].axis('tight')
            axs[1].set_xlim(nr//2-100,nr//2+100)
            axs[1].set_ylim(1, -1)
            axs[2].imshow(df1_inv_plus.T, cmap='gray', vmin=-clip, vmax=clip, extent=(0, nr, t[-1], -t[-1]))
            axs[2].set_title(r'$f^+$'), axs[0].set_xlabel(r'$x_R$')
            axs[2].axis('tight')
            axs[2].set_xlim(nr//2-100,nr//2+100)
            axs[2].set_ylim(1, -1);

            
            fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 3))
            ax.imshow(dp0_minus.T, cmap='gray', vmin=-clip, vmax=clip,
                      extent=(0, nr, t[-1], -t[-1]))
            ax.set_title(r'$p^0$'), ax.set_xlabel(r'$x_R$')
            ax.axis('tight')
            ax.set_ylim(1., 0)
            ax.set_xlim(nr//2-100,nr//2+100)

            fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 3))
            ax.imshow(dg_inv_minus.T * (1-w.T), cmap='gray', vmin=-clip, vmax=clip,
                      extent=(0, nr, t[-1], -t[-1]))
            ax.set_title(r'$g^-$'), ax.set_xlabel(r'$x_R$')
            ax.axis('tight')
            ax.set_ylim(1., 0)
            ax.set_xlim(nr//2-100,nr//2+100)

            fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 3))
            ax.imshow(dp0_minus.T - dg_inv_minus.T, cmap='gray', vmin=-clip, vmax=clip,
                      extent=(0, nr, t[-1], -t[-1]))
            ax.set_title(r'$p^0 - g^-$'), ax.set_xlabel(r'$x_R$')
            ax.axis('tight')
            ax.set_ylim(1., 0)
            ax.set_xlim(nr//2-100,nr//2+100);
            """
    #plt.show()
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
    ixrestart = int(sys.argv[9]) # restart from x-line with index ixrestart
    ixend = int(sys.argv[10]) # end at x-line with index ixend
    iyrestart = int(sys.argv[11]) # restart from y-line with index iyrestart
    iyend = int(sys.argv[12]) # end at y-line with index iyend
    run(subsampling, vsz, nvsx, dvsx, ovsx, nvsy, dvsy, ovsy, ixrestart, ixend, iyrestart, iyend)






