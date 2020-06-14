######################################
# Marchenko redatuming with 3D dataset
######################################
import warnings
warnings.filterwarnings('ignore')

import os
import time
import sys
import numpy as np
import scipy as sp
import dask.array as da
import zarr
import pylops
import pylops_distributed

from dask.distributed import Client
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


def run(subsampling, nworkers, nworkersmin, vsz, nvsx, dvsx, ovsx, nvsy, dvsy, ovsy, ivsin, ivsend):
    # Connect to cluster
    client = Client()
    print('Client', client)
    nworkers_connected = len(np.array(list(client.ncores().values())))
    itry = 0
    while nworkers > nworkers_connected:
        time.sleep(10)
        client.restart()
        print('Client', client)
        nworkers_connected = len(np.array(list(client.ncores().values())))
        print('Nworkers', nworkers_connected)
        itry +=1
        if itry == 10:
            nworkers == nworkersmin
        if itry == 30:
            raise ConnectionError('Cannot connect to minimum number of workers...')
    ncores = np.sum(np.array(list(client.ncores().values())))

    print('Client', client)
    print('Nworkers', nworkers_connected)
    print('Ncores', ncores)
    print('Subsampling', subsampling)

    # Check log file to define starting virtual point
    logfile = os.environ["STORE_PATH"] + 'log_ivstart%d_ivend%d.txt' % (ivsin, ivsend)
    if os.path.exists(logfile):
        with open(logfile, 'r+') as f:
            lines = f.readlines()
            if len(lines) > 0:
                ivsin = max(int(lines[-1].split(' ')[0]) + 1, ivsin)
    print('Starting from virtual source %d...' % ivsin)

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
    if ivsend == -1:
        ivsend = nvsy * nvsx
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

    firstivs = True
    for ivs in range(ivsin, ivsend):
        t0 = time.time()

        # Virtual point (x, y, z)
        vs = np.array([VSX[ivs], VSY[ivs], vsz])

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
        #df1_inv_minus, df1_inv_plus = df1_inv_tot[:(2*nt-1)].T, df1_inv_tot[(2*nt-1):].T
        dg_inv_minus, dg_inv_plus = -dg_inv[:(2*nt-1)].T, np.fliplr(dg_inv[(2*nt-1):].T)

        # Remove acausal artefacts in upgoing Green's functions
        dg_inv_minus = dg_inv_minus * (1-w)

        # Save Green's functions
        Gplus[:, :, ivs] = (dg_inv_plus[:, nt-1:].T).astype(np.float32)
        Gminus[:, :, ivs] = (dg_inv_minus[:, nt-1:].T).astype(np.float32)
        Gdir[:, :, ivs] = (G0sub).astype(np.float32)
        Grtm[:, :, ivs] = (dp0_minus[:, nt-1:].T).astype(np.float32)

        if firstivs:
            message = str(ivs) + ' - vs=' + str(vs) + \
                      ' - FIRST - .... Excecution time: ' + str(time.time() - t0) + ' s'
            firstivs = False
        else:
            message = str(ivs) + ' - vs=' + str(vs) + \
                      '.... Excecution time: ' + str(time.time() - t0) + ' s'
        print(message)
        with open(logfile, 'a') as f:
            f.write(message + '\n')
    client.close()


if __name__ == '__main__':
    subsampling = int(sys.argv[1]) # data subsampling
    nworkers = float(sys.argv[2])  # number of expected workers
    nworkersmin = float(sys.argv[3])  # minimum number of allowed workers
    vsz = float(sys.argv[4]) # depth of line
    nvsx = int(sys.argv[5]) # number of samples along x-line
    dvsx = float(sys.argv[6]) # sampling of x-line
    ovsx = float(sys.argv[7]) # origin of x-line
    nvsy = int(sys.argv[8]) # number of samples along y-line
    dvsy = float(sys.argv[9]) # sampling of y-line
    ovsy = float(sys.argv[10]) # origin of y-line
    ivsin = int(sys.argv[11]) # index of first virtual point to compute
    ivsend = int(sys.argv[12]) # index of last virtual point to compute
    run(subsampling, nworkers, nworkersmin, vsz, nvsx, dvsx, ovsx,
        nvsy, dvsy, ovsy, ivsin, ivsend)
