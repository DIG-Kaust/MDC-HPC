#!/usr/bin/env python
# coding: utf-8

#########################################################################
# Multi-dimensional convolution timing benchmarks - single virtual source
#########################################################################
import os
import sys
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import dask.array as da
import zarr
import pylops
import pylops_distributed
import scooby

from datetime import date
from timeit import repeat
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


def total_ram():
        tmem = psutil.virtual_memory().total
        return '{:.1f} GB'.format(tmem / (1024.0 ** 3))


def run(subsampling):
    
    client = pylops_distributed.utils.backend.dask(hardware='multi', client='be-linrgsn130:8786')
    client.restart()
    print(client)

    nworkers = len(np.array(list(client.ncores().values())))
    ncores = np.sum(np.array(list(client.ncores().values())))
    print('Nworkers', nworkers)
    print('Ncores', ncores)
    print('Subsampling', subsampling)

    # Inputs
    nfmax = 300         # max frequency for MDC (#samples)
    darea = 1           # areal extent for spatial integration (not needed here...)
    ffirst = True       # frequency in first axis of zarr file
    rechunk = True      # rechunk R
    rebalance = True    # rebalance R across nodes

    # Timing parameters
    nrepeat = min(4, subsampling+1)
    ntime = min(10, subsampling+1)

    # Input file names
    inputfile_aux = os.environ["STORE_PATH"]+'3DMarchenko_auxiliary_2.npz' 
    zarrfile = os.environ["STORE_PATH"]+'input3D_sub%d%s.zarr' % (subsampling, '_ffirst' if ffirst else '')

    # Load auxiliary input (contains sources, recs, virtual source etc.)
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
    vs = inputdata_aux['vs']

    # Time axis
    ot, dt, nt = 0, 2.5e-3, 601
    t = np.arange(nt)*dt

    # Density model
    rho = inputdata_aux['rho']
    z, x, y = inputdata_aux['z'], inputdata_aux['x'], inputdata_aux['y']

    # Read subsurface fields and wavelet to apply to subsurface fields
    G0sub = inputdata_aux['G0'][:, ::subsampling]
    wav = ricker(t[:51], 20)[0]
    wav_c = np.argmax(wav)

    # Convolve with wavelet
    G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full') 
    G0sub = G0sub[wav_c:][:nt]

    # Ensure G0sub_ana is float32
    G0sub = G0sub.astype(np.float32)

    # Read Reflection response from Zarr file
    dRtwosided_fft = 2 * da.from_zarr(zarrfile)  # 2 * as per theory you need 2*R

    #nchunks = [max(nfmax // ncores, 1), ns, nr]
    #corechunks=True
    nchunks = [max(nfmax // (nworkers + 1), 1), ns, nr]
    #nfchunks = [nfmax // nworkers]*nworkers
    #nfchunks[-1] += nfmax - np.sum(np.array(nfchunks))
    #nchunks = (tuple(nfchunks), (ns,), (nr,))
    corechunks=False
    if not ffirst:
        dRtwosided_fft = dRtwosided_fft.transpose(2, 1, 0)
    if rechunk:
        dRtwosided_fft = dRtwosided_fft.rechunk(nchunks)
    else:
        nchunks = dRtwosided_fft.chunksize
    dRtwosided_fft = client.persist(dRtwosided_fft)
    client.rebalance(dRtwosided_fft)
    print('Number of chunks', np.prod(dRtwosided_fft.numblocks))
    print('Chunks', dRtwosided_fft.chunks)

    # Create distributed MDC operator
    dRop = dMDC(dRtwosided_fft, nt=2*nt-1, nv=1, dt=dt, dr=darea, 
                twosided=True, saveGt=False)

    # Input focusing function
    dfd_plus = np.concatenate((np.fliplr(G0sub.T).T, np.zeros((nt-1, nr)))).astype(np.float32)
    dfd_plus = da.from_array(dfd_plus)

    # Run standard redatuming as benchmark
    dp0_minus = dRop.matvec(dfd_plus.flatten())
    p0_minus = dp0_minus.compute()
    p0_minus = p0_minus.reshape((2*nt-1), nr)
       
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 8))
    ax.imshow(p0_minus, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
    ax.set_title(r'$p_0^-$')
    ax.set_xlabel(r'$x_R$')
    ax.set_ylabel(r'$t$')
    ax.axis('tight');

    # Time forward
    dp0_forw = dRop.matvec(dfd_plus.flatten())

    t0 = time.time()
    exctime = np.array(repeat(lambda: dp0_forw.compute(), number=ntime, repeat=nrepeat))
    meantime, stdtime = np.mean(exctime/ntime), np.std(exctime/ntime)
    print( time.time() - t0, meantime*ntime*nrepeat)
    print('Forward:', meantime, stdtime)
    df = pd.DataFrame(dict(nworkers=nworkers, ncores=ncores, ram=total_ram(), 
                           subsampling=subsampling, ffirst=ffirst,
                           meantime=meantime, stdtime=stdtime, 
                           nchunks=str(nchunks), rebalance=rebalance,
                           corechunks=corechunks,
                           nrepeat=nrepeat, ntime=ntime,
                           time=date.today()), index=[0])

    # add to csv file
    header=True
    if os.path.isfile('Benchmarks/benchmark_forw.csv'):
        df_other = pd.read_csv('Benchmarks/benchmark_forw.csv')
        df = pd.concat([df_other, df])
    df.to_csv('Benchmarks/benchmark_forw.csv', index=False)

    # Time adjoint
    dp0_adj = dRop.rmatvec(dfd_plus.flatten())

    t0 = time.time()
    exctime = np.array(repeat(lambda: dp0_adj.compute(), number=ntime, repeat=nrepeat))
    meantime, stdtime = np.mean(exctime/ntime), np.std(exctime/ntime)
    print( time.time() - t0, meantime*ntime*nrepeat)
    print('Adjoint:', meantime, stdtime)
    df = pd.DataFrame(dict(nworkers=nworkers, ncores=ncores, ram=total_ram(), 
                           subsampling=subsampling, ffirst=ffirst,
                           meantime=meantime, stdtime=stdtime, 
                           nchunks=str(nchunks), rebalance=rebalance,
                           corechunks=corechunks,
                           nrepeat=nrepeat, ntime=ntime,
                           time=date.today()), index=[0])

    # add to csv file
    header=True
    if os.path.isfile('Benchmarks/benchmark_adj.csv'):
        df_other = pd.read_csv('Benchmarks/benchmark_adj.csv')
        df = pd.concat([df_other, df])
    df.to_csv('Benchmarks/benchmark_adj.csv', index=False)
    
    client.close()
    #plt.show()
    plt.close()
    

if __name__ == '__main__':
    run(int(sys.argv[1]))


