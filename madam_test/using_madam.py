import numpy as np
import healpy as hp
import pylab as plt

from mpi4py import MPI

import libmadam_wrapper as madam
from gbpipe.gbsim import sim_noise1f

from gbpipe.utils import set_logger
from scipy.interpolate import interp1d

def set_parameters(nside, fsample, nsample):
    pars = {}

    pars['info'] = 0 
    #pars['nthreads'] = 
    pars['nsubchunk'] = 0
    #pars['isubchunk'] = 
    #pars['time_unit'] = 
    pars['base_first'] = 1.0
    #pars['nshort'] = 
    pars['nside_map'] = nside
    pars['nside_cross'] = nside // 2
    pars['nside_submap'] = nside // 4
    #pars['good_baseline_fraction'] = 
    #pars['concatenate_messages'] = 
    pars['allreduce'] = True
    #pars['reassign_submaps'] = 
    #pars['pixmode_map'] = 
    #pars['pixmode_cross'] = 
    #pars['pixlim_map'] = 
    #pars['pixlim_cross'] = 
    #pars['incomplete_matrices'] = 
    #pars['allow_decoupling'] = 
    #pars['kfirst'] = 
    #pars['basis_func'] = 
    #pars['basis_order'] = 
    #pars['iter_min'] = 
    pars['iter_max'] = 10
    #pars['cglimit'] = 
    pars['fsample'] = fsample
    #pars['mode_detweight'] = 
    #pars['flag_by_horn'] = 
    #pars['write_cut'] = 
    #pars['checknan'] = 
    #pars['sync_output'] = 
    #pars['skip_existing'] = 
    pars['temperature_only'] = True
    #pars['force_pol'] = 
    pars['noise_weights_from_psd'] = True
    #pars['radiometers'] = 
    #pars['psdlen'] = 
    #pars['psd_down'] = 
    pars['kfilter'] = True
    pars['diagfilter'] = 0
    #pars['precond_width_min'] = 
    #pars['precond_width_max'] = 
    #pars['use_fprecond'] = 
    #pars['use_cgprecond'] = 
    #pars['rm_monopole'] = 
    pars['path_output'] = '/home/klee_ext/kmlee/hpc_data/madam_test/'
    pars['file_root'] = 'madam_test'

    pars['write_map'] = True
    pars['write_binmap'] = True
    pars['write_hits'] = True
    pars['write_matrix'] = True
    pars['write_wcov'] = True
    pars['write_base'] = True
    pars['write_mask'] = True
    pars['write_leakmatrix'] = True 

    #pars['unit_tod'] = 
    #pars['file_gap_out'] = 
    #pars['file_mc'] = 
    pars['write_tod'] = True
    #pars['file_inmask'] = 
    #pars['file_spectrum'] = 
    #pars['file_gap'] = 
    #pars['binary_output'] = 
    #pars['nwrite_binary'] = 
    #pars['file_covmat'] = 
    #pars['detset'] = 
    #pars['detset_nopol'] = 
    pars['survey'] = ['hm1:{} - {}'.format(0, nsample / 2),]
    pars['bin_subsets'] = True
    #pars['mcmode'] = 

    return pars


def using_madam():
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger()

    if itask == 0:
        log.warning('Running with ', ntask, ' MPI tasks')

    log.info('Calling Madam')

    nside = 64
    npix = hp.nside2npix(nside)
    fsample = 100
    dt = 600
    nnz = 1
    nsample = fsample * dt

    pars = set_parameters(nside, fsample, nsample)

    cl_length = 3 * nside
    cl = np.zeros(shape=cl_length)
    cl[5] = 1

    np.random.seed(0) 

    dets =['det0']    
    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = np.arange(len(pixels)) % npix 

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    pixweights[:] = 1

    signal = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    #signal[:] = np.sin(2*np.pi*pixels/npix*2)*3
    #signal[:] += np.random.randn(nsample * ndet)

    noisesim, (psdf, psdv) = sim_noise1f(nsample, 1, 1, fsample=fsample, alpha=1, rseed=0, return_psd=True)
    noise  = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    noise[:]  = noisesim
    psdf = psdf[:len(psdf)//2]
    psdv = psdv[:len(psdv)//2]
    plt.plot(signal)
    plt.plot(noise)

    signal_in = signal + noise

    nperiod = 4
    periods = np.zeros(nperiod, dtype=int)
    periods[0] = int(nsample*0)
    periods[1] = int(nsample*0)
    periods[2] = int(nsample*0)
    periods[3] = int(nsample*0)

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf)
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)
    psdvals[:] = psdv[:npsdbin]

    hmap = np.zeros(npix, dtype=int)
    bmap = np.zeros(npix, dtype=float)

    for p, s in zip(pixels, signal):
        hmap[p] += 1
        bmap[p] += s

    hmap_tot = np.zeros(npix, dtype=int)
    bmap_tot = np.zeros(npix, dtype=float)

    comm.Reduce(hmap, hmap_tot, op=MPI.SUM, root=0)
    comm.Reduce(hmap, hmap_tot, op=MPI.SUM, root=0)

    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    plt.show()


def map_madam():
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger()

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))

    log.info('Calling Madam')

    nside = 64
    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = 600
    nnz = 1
    nsample = fsample * dt

    pars = set_parameters(nside, fsample, nsample)

    np.random.seed(0) 

    dets =['det0']    
    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = np.arange(len(pixels)) % npix 

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    pixweights[:] = 1

    signal = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    #signal[:] = np.sin(2*np.pi*pixels/npix*2)*3
    #signal[:] += np.random.randn(nsample * ndet)

    noisesim, (psdf, psdv) = sim_noise1f(nsample, 0.1, 1, fsample=fsample, alpha=1.2, rseed=0, return_psd=True)
    noise  = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    noise[:]  = noisesim
    psdf = psdf[:len(psdf)//2]
    psdv = psdv[:len(psdv)//2]
    plt.figure()
    plt.loglog(psdf, psdv)
    plt.figure()
    plt.plot(signal)
    plt.plot(noise)

    signal_in = signal + noise

    nperiod = 10
    periods = np.zeros(nperiod, dtype=int)
    for i in range(nperiod):
        periods[i] = 10000*i#int(nsample*0.025)

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf)
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)
    psdvals[:] = psdv[:npsdbin]

    hmap = np.zeros(npix, dtype=int)
    bmap = np.zeros(npix, dtype=float)

    for p, s in zip(pixels, signal):
        hmap[p] += 1
        bmap[p] += s

    hmap_tot = np.zeros(npix, dtype=int)
    bmap_tot = np.zeros(npix, dtype=float)

    comm.Reduce(hmap, hmap_tot, op=MPI.SUM, root=0)
    comm.Reduce(hmap, hmap_tot, op=MPI.SUM, root=0)

    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    plt.show()
   
if __name__=='__main__':
    map_madam()
