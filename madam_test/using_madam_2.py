import os
import numpy as np
import healpy as hp
#import pylab as plt

import smtplib
import traceback
from email.mime.text import MIMEText

from mpi4py import MPI

import libmadam_wrapper as madam
from gbpipe import gbparam, gbdir
from gbpipe.gbsim import sim_noise1f, sim_noise1f_old

from gbpipe.utils import set_logger, hostname
from scipy.interpolate import interp1d


def set_parameters(nside, fsample, nsample, outpath):
    pars = {}

    pars['info'] = 2
    #pars['nthreads'] = 1
    pars['nsubchunk'] = 0
    #pars['isubchunk'] = 1
    #pars['time_unit'] = 
    pars['base_first'] = 10
    #pars['nshort'] = 10
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
    #pars['kfirst'] = False
    pars['basis_func'] = 'polynomial'
    pars['basis_order'] = 0
    #pars['iter_min'] = 
    pars['iter_max'] = 100
    #pars['cglimit'] = 
    pars['fsample'] = fsample
    #pars['mode_detweight'] = 
    #pars['flag_by_horn'] = 
    #pars['write_cut'] = 
    #pars['checknan'] = 
    #pars['sync_output'] = 
    #pars['skip_existing'] = 
    pars['temperature_only'] = False
    #pars['force_pol'] = False
    pars['noise_weights_from_psd'] = True
    #pars['radiometers'] = 
    #pars['psdlen'] = 
    #pars['psd_down'] = 
    #pars['kfilter'] = False
    #pars['diagfilter'] = 0.0
    #pars['precond_width_min'] = 
    #pars['precond_width_max'] = 
    #pars['use_fprecond'] = 
    #pars['use_cgprecond'] = 
    #pars['rm_monopole'] = True
    #pars['path_output'] = '/home/klee_ext/kmlee/hpc_data/madam_test/'
    pars['path_output'] = outpath 
    pars['file_root'] = 'madam_test'

    pars['write_map'] = True
    pars['write_binmap'] = True
    pars['write_hits'] = True
    pars['write_matrix'] = False#True
    pars['write_wcov'] = False#True
    pars['write_base'] = True
    pars['write_mask'] = False#True
    pars['write_leakmatrix'] = False

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
    #pars['survey'] = ['hm1:{} - {}'.format(0, nsample / 2),]
    #pars['bin_subsets'] = True
    #pars['mcmode'] = 

    return pars


def pixels_for_detector(module, pix_mod, ra, dec, psi, nside=1024):
    npix = nside * nside * 12

    param = gbparam.GBparam()
    gbpix = param.pixinfo
    module_idx = module

    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]

    v_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    p_det = gbdir.psi2vec_xp(v_det, psi_det)

    v_det = v_det[pix_mod]
    p_det = p_det[pix_mod]

    rmat = gbdir.Rot_matrix_equatorial(ra, dec, psi, deg=True)
    v_obs = gbdir.Rotate(v_det, rmat)
    p_obs = gbdir.Rotate(p_det, rmat)
    pixs = hp.vec2pix(nside, v_obs[:,0], v_obs[:,1], v_obs[:,2])
    psis = gbdir.angle_from_meridian(v_obs, p_obs)

    return pixs, psis


def tod_ascii2fits(outpath, remove_asc=False):
    tod_asc = np.genfromtxt(outpath + 'tod_madam_asc.dat')
    tod_asc = tod_asc.T
    np.savez_compressed(outpath + 'tod_madam', signal=tod_asc[0], base_function=tod_asc[1], aa=tod_asc[2])
    if remove_asc:
        os.remove(outpath + 'tod_madam_asc.dat')

    return


def map_madam():
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger()

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))


    nside = 128
    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = npix // fsample #600
    nnz = 1
    nsample = fsample * dt
    module = 1
    pix_mod = 0
    length = 86400000
    #length = 6000000

    log.info('Loading tod')
    dat = np.load('/home/klee_ext/kmlee/pixel_tod/tod_mod1pix0.npz')

    ra = dat['ra'][:length]
    dec = dat['dec'][:length]
    psi = dat['psi'][:length]
    ix = dat['ix'][:length]
    iy = dat['iy'][:length]
    noise = dat['noise'][:length]

    log.info('Calculating pointings')
    nsample = len(ra) 
    log.info('number of samples = {}'.format(nsample))
    signal = ix + iy
    pix_obs, psi_obs = pixels_for_detector(module, pix_mod, ra, dec, psi, nside)

    outpath_pre = '/home/klee_ext/kmlee/test_madam/madam'
    outpath = outpath_pre + ('_%03d/' % (0))
    i = 0
    while os.path.isdir(outpath):
        i += 1
        outpath = outpath_pre + ('_%03d/' % (i))

    os.mkdir(outpath)
    log.info('Directory {} have been made.'.format(outpath))

    log.info('Setting parameter')
    pars = set_parameters(nside, fsample, nsample, outpath)

    np.random.seed(1) 

    dets =['det0']    
    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = pix_obs 

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    pixweights[:nsample] = 1 
    #pixweights[nsample:nsample*2] = np.cos(2*psi_obs)
    #pixweights[nsample*2:nsample*3] = np.sin(2*psi_obs) 


    log.info('loading noise')
    dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/noise1f_1day_alpha1_fknee1.npz')
    noisesim = dat['noisesim']
    psdf = dat['psdf']
    psdv = dat['psdv']
    #noisesim, (psdf, psdv) = sim_noise1f(nsample, wnl=300e1, fknee=1, fsample=fsample, alpha=1, rseed=0, return_psd=True)

    noise_gen = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    noise_gen[:] = noisesim
    psdf = psdf[:len(psdf)//2]
    psdv = psdv[:len(psdv)//2]

    np.savez_compressed(outpath+'tod_raw', signal=signal, noise=noise_gen)
    ## defining signal
    #signal = signal + noise #- baseline
    signal = signal + noise_gen #- baseline

    signal_in = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    signal_in[:] = signal 

    log.info('Setting periods')
    nperiod = 144
    periods = np.zeros(nperiod, dtype=int)
    for i in range(nperiod):
        periods[i] = int(nsample//nperiod*i)

    print (periods)

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf)
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)
    psdvals[:] = np.abs(psdv[:npsdbin])

    """
    log.info('hmap, bmap')
    hmap = np.zeros(npix, dtype=int)
    bmap = np.zeros(npix, dtype=float)

    for p, s in zip(pixels, signal):
        hmap[p] += 1
        bmap[p] += s

    hmap_tot = np.zeros(npix, dtype=int)
    bmap_tot = np.zeros(npix, dtype=float)

    comm.Reduce(hmap, hmap_tot, op=MPI.SUM, root=0)
    comm.Reduce(hmap, bmap_tot, op=MPI.SUM, root=0)
    """

    log.info('Calling Madam')
    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    log.info('ascii tod to npz')
    tod_ascii2fits(outpath, remove_asc=True)

    return


def pol_madam():
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger()

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))


    nside = 128
    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = npix // fsample #600
    nnz = 3
    nsample = fsample * dt
    module = 1
    pix_mod = 11 
    pix_mod1 = 7
    #length = 86400000
    length = 6000000

    log.info('Loading tod')
    #dat = np.load('/home/klee_ext/kmlee/hpc_data/GBsim_1day_0_5deg/2019-09-01/pixel_tod/tod_mod1pix0.npz')
    dat = np.load('/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/tod_mod1pix11.npz')
    dat1 = np.load('/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/tod_mod1pix7.npz')
    #dat2 = np.load('/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/tod_mod1pix12.npz')
    #dat3 = np.load('/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/tod_mod1pix15.npz')

    ra = dat['ra'][:length]
    dec = dat['dec'][:length]
    psi = dat['psi'][:length]
    ix = dat['ix'][:length]
    iy = dat['iy'][:length]
    noise = dat['noise'][:length]

    ra1 = dat1['ra'][:length]
    dec1 = dat1['dec'][:length]
    psi1 = dat1['psi'][:length]
    ix1 = dat1['ix'][:length]
    iy1 = dat1['iy'][:length]
    noise1 = dat1['noise'][:length]

    log.info('Calculating pointings')
    nsample = len(ra) 
    log.info('number of samples = {}'.format(nsample))
    signal = ix #- iy
    signal1 = ix1 #- iy1
    pix_obs, psi_obs = pixels_for_detector(module, pix_mod, ra, dec, psi, nside)
    pix_obs1, psi_obs1 = pixels_for_detector(module, pix_mod1, ra1, dec1, psi1, nside)

    outpath_pre = '/home/klee_ext/kmlee/test_madam/madam'
    outpath = outpath_pre + ('_%03d/' % (0))
    i = 0
    while os.path.isdir(outpath):
        i += 1
        outpath = outpath_pre + ('_%03d/' % (i))

    os.mkdir(outpath)
    log.info('Directory {} have been made.'.format(outpath))

    log.info('Setting parameter')
    pars = set_parameters(nside, fsample, nsample, outpath)

    np.random.seed(1) 

    dets =['det11', 'det7']    
    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    ## concatenate the arrays
    signal = np.append(signal, signal1)
    noise = np.append(noise, noise1)
    pix_obs = np.append(pix_obs, pix_obs1)
    psi_obs = np.append(psi_obs, psi_obs1)

    del(signal1)
    del(noise1)
    del(pix_obs1)
    del(psi_obs1)

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = pix_obs 

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    #pixweights[:nsample] = 1 
    #pixweights[nsample:nsample*2] = 0#np.cos(2*psi_obs)
    #pixweights[nsample*2:nsample*3] = 0#np.sin(2*psi_obs) 
    pixweights[::3] = 1
    pixweights[1::3] = np.cos(2*psi_obs)
    pixweights[2::3] = np.sin(2*psi_obs) 


    log.info('Generating noise psd.')
    noisesim, (psdf, psdv) = sim_noise1f(nsample, wnl=300e-6, fknee=1, 
                                fsample=fsample, alpha=1, rseed=0, 
                                return_psd=True)
    psdf = psdf[:len(psdf)//2]
    psdv = psdv[:len(psdv)//2]

    np.savez_compressed(outpath+'tod_raw', signal=signal, noise=noise)
    ## defining signal
    signal = signal + noise*1e11

    signal_in = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    signal_in[:] = signal 

    log.info('Setting periods')
    nperiod = 100
    periods = np.zeros(nperiod, dtype=int)
    for i in range(nperiod):
        periods[i] = int(nsample//nperiod*i)

    print (periods)

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf)
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)
    psdvals[:] = np.append(np.abs(psdv[:npsdbin]),np.abs(psdv[:npsdbin]))
    
    log.info('Calling Madam')
    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    log.info('ascii tod to npz')
    tod_ascii2fits(outpath, remove_asc=True)

    return


def pol_madam_v2(
        signaltype, 
        pix_mod=[7,11], 
        nside=1024, 
        obstime=86400,
        inpath='/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/',
        outdir=None, 
        pars=None,
        savetod=False):

    # some lines to play with multiple files efficienly.
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger(level='INFO')

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))

    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = npix // fsample #600
    nnz = 3
    nsample = fsample * dt
    length = obstime * fsample 
    #length = 600 * fsample

    module = 1
    #pix_mod = [7, 11]#np.arange(23)#[7, 2, 11]

    log.info('Using module {}, detector {}'.format(module, pix_mod))
    log.info('Loading tod')

    fnames = []
    dets = []

    for pix_idx in pix_mod:
        dets.append('mod{}det{}'.format(module, pix_idx))
        fnames.append(os.path.join(inpath, 'tod_mod{}pix{}.npz'.format(module, pix_idx)))

    ra = []
    dec = []
    psi_arr = []
    ix_arr = []
    iy_arr = []
    noise_arr = []
    ra_tmp = []
    dec_tmp = []

    for f in fnames:
        log.debug('pixeltod: ' + f)
        dat = np.load(f)

        ra = dat['ra'][:length]
        dec = dat['dec'][:length]

        #if ra_tmp.all() != ra.all() or dec_tmp.all() != dec.all(): 
        #    log.error('The sky directions are not consistent between the pixel tod.')
        #    raise

        psi_arr.append(dat['psi'][:length])
        ix_arr.append(dat['ix'][:length])
        iy_arr.append(dat['iy'][:length])
        noise_arr.append(dat['noise'][:length])

        #ra_tmp = ra.copy()
        #dec_tmp = dec.copy()

    #del(ra_tmp)
    #del(dec_tmp)

    log.info('Calculating pointings')

    nsample = len(ra) 

    log.info('number of samples = {}'.format(nsample))

    pix_obs_arr = []
    psi_obs_arr = []
    for det, psi in zip(pix_mod, psi_arr):
        res = pixels_for_detector(module, det, ra, dec, psi, nside)
        pix_obs_arr.append(res[0]) 
        psi_obs_arr.append(res[1])

    if outdir is None:
        outpath_prefix = '/home/klee_ext/kmlee/test_madam/madam'
        outpath = outpath_prefix + ('_%03d/' % (0))
        i = 0
        while os.path.isdir(outpath):
            i += 1
            outpath = outpath_prefix + ('_%03d/' % (i))
    else:
        outpath = '/home/klee_ext/kmlee/test_madam/' + outdir

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
        log.info('Directory {} have been made.'.format(outpath))

    if pars is None:
        log.info('Setting parameter')
        pars = set_parameters(nside, fsample, nsample, outpath)
    else:
        pars['path_output'] = outpath

    np.random.seed(1) 

    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    log.info('Generating time stamp')
    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    ## concatenate the arrays

    signal_arr = ix_arr #- iy

    signal = np.concatenate(signal_arr, axis=None)
    noise = np.concatenate(noise_arr, axis=None)
    pix_obs = np.concatenate(pix_obs_arr, axis=None)
    psi_obs = np.concatenate(psi_obs_arr, axis=None)

    #del(signal_arr)
    #del(noise_arr)
    #del(pix_obs_arr)
    #del(psi_obs_arr)

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = pix_obs 
    #del(pix_obs)

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    #pixweights[:nsample] = 1 
    #pixweights[nsample:nsample*2] = 0#np.cos(2*psi_obs)
    #pixweights[nsample*2:nsample*3] = 0#np.sin(2*psi_obs) 
    pixweights[::3] = 1
    pixweights[1::3] = np.cos(2*psi_obs)
    pixweights[2::3] = np.sin(2*psi_obs) 
    #del(psi_obs)


    #log.info('Generating noise psd')
    #noisesim, (psdf, psdv) = sim_noise1f(nsample, wnl=300e-6, fknee=1, 
    #                            fsample=fsample, alpha=1, rseed=0, 
    #                            return_psd=True)


    ## defining signal
    if signaltype == 'signal_only':
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/noise1f_1day_alpha1_fknee1.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = False

        signal = signal
    elif signaltype == 'signal+1fnoise':
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/gb_noise1f_1day.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = True

        signal = signal + np.concatenate([noisesim]*ndet)
    elif signaltype == 'signal+wnoise':
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/gb_wnoise_1day.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = False

        signal = signal + np.concatenate([noisesim]*ndet)
    elif signaltype == 'wnoise_only':
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/gb_wnoise_1day.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = False

        signal = np.concatenate([noisesim]*ndet)
    elif signaltype == '1fnoise_only':
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/gb_noise1f_1day.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = True

        signal = np.concatenate([noisesim]*ndet)
    else:
        log.info('loading noise')
        dat = np.load('/home/klee_ext/kmlee/hpc_data/noise_ref/gb_noise1f_1day.npz')
        noisesim = dat['noisesim'][:length]
        psdf = dat['psdf']
        psdv = dat['psdv']
        pars['noise_weights_from_psd'] = True

        signal = signal + np.concatenate([noisesim]*ndet)

    if savetod:
        np.savez_compressed(os.path.join(outpath, 'tod_raw'), signal=signal, noise=noisesim)

    signal_in = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    signal_in[:] = signal 
    del(signal)
    del(noisesim)


    log.info('Setting periods')
    nperiod = 100
    periods = np.zeros(nperiod, dtype=int)
    for i in range(nperiod):
        periods[i] = int(nsample//nperiod*i)

    log.debug ('periods={}'.format(periods))

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf) 
    log.debug('npsdbin={}'.format(npsdbin))
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)

    for i in range(npsdtot):
        psdvals[i*npsdbin:(i+1)*npsdbin] = np.abs(psdv[:npsdbin])
    
    log.info('Calling Madam')
    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    if savetod:
        log.info('ascii tod to npz')
        tod_ascii2fits(outpath, remove_asc=True)

    return


def pol_madam_v2_toy(
        signaltype, 
        pix_mod=[7,11], 
        nside=1024, 
        obstime=86400,
        inpath='/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/',
        outdir=None, 
        pars=None,
        savetod=False):

    # some lines to play with multiple files efficienly.
    comm = MPI.COMM_WORLD
    itask = comm.Get_rank()
    ntask = comm.Get_size()

    log = set_logger(level='INFO')

    if itask == 0:
        log.warning('Running with {} MPI tasks'.format(ntask))

    npix = hp.nside2npix(nside)
    fsample = 1000
    dt = npix // fsample #600
    nnz = 3
    nsample = fsample * dt
    length = obstime * fsample 
    #length = 600 * fsample

    module = 1
    #pix_mod = [7, 11]#np.arange(23)#[7, 2, 11]

    log.info('Using module {}, detector {}'.format(module, pix_mod))
    log.info('Loading tod')

    fnames = []
    dets = []

    for pix_idx in pix_mod:
        dets.append('mod{}det{}'.format(module, pix_idx))
        fnames.append(os.path.join(inpath, 'tod_mod{}pix{}.npz'.format(module, pix_idx)))

    ra = []
    dec = []
    psi_arr = []
    ix_arr = []
    iy_arr = []
    noise_arr = []
    ra_tmp = []
    dec_tmp = []

    for f in fnames:
        log.debug('pixeltod: ' + f)
        dat = np.load(f)

        ra = dat['ra'][:length]
        dec = dat['dec'][:length]

        #if ra_tmp.all() != ra.all() or dec_tmp.all() != dec.all(): 
        #    log.error('The sky directions are not consistent between the pixel tod.')
        #    raise

        psi_arr.append(dat['psi'][:length])
        ix_arr.append(dat['ix'][:length])
        iy_arr.append(dat['iy'][:length])
        noise_arr.append(dat['noise'][:length])

        #ra_tmp = ra.copy()
        #dec_tmp = dec.copy()

    #del(ra_tmp)
    #del(dec_tmp)

    log.info('Calculating pointings')

    nsample = len(ra) 

    log.info('number of samples = {}'.format(nsample))

    pix_obs_arr = []
    psi_obs_arr = []
    for det, psi in zip(pix_mod, psi_arr):
        res = pixels_for_detector(module, det, ra, dec, psi, nside)
        pix_obs_arr.append(res[0]) 
        psi_obs_arr.append(res[1])

    if outdir is None:
        outpath_prefix = '/home/klee_ext/kmlee/test_madam/madam'
        outpath = outpath_prefix + ('_%03d/' % (0))
        i = 0
        while os.path.isdir(outpath):
            i += 1
            outpath = outpath_prefix + ('_%03d/' % (i))
    else:
        outpath = '/home/klee_ext/kmlee/test_madam/' + outdir

    if not os.path.isdir(outpath):
        os.mkdir(outpath)
        log.info('Directory {} have been made.'.format(outpath))

    if pars is None:
        log.info('Setting parameter')
        pars = set_parameters(nside, fsample, nsample, outpath)
    else:
        pars['path_output'] = outpath

    np.random.seed(1) 

    ndet = len(dets)
    weights = np.ones(ndet, dtype=float)

    log.info('Generating time stamp')
    timestamps = np.zeros(nsample, dtype=madam.TIMESTAMP_TYPE)
    timestamps[:] = np.arange(nsample) + itask * nsample

    ## concatenate the arrays

    signal_arr = ix_arr #- iy

    signal = np.concatenate(signal_arr, axis=None)
    noise = np.concatenate(noise_arr, axis=None)
    pix_obs = np.concatenate(pix_obs_arr, axis=None)
    psi_obs = np.concatenate(psi_obs_arr, axis=None)

    #del(signal_arr)
    #del(noise_arr)
    #del(pix_obs_arr)
    #del(psi_obs_arr)

    pixels = np.zeros(ndet * nsample, dtype=madam.PIXEL_TYPE)
    pixels[:] = pix_obs 
    #del(pix_obs)

    pixweights = np.zeros(ndet * nsample * nnz, dtype=madam.WEIGHT_TYPE)
    #pixweights[:nsample] = 1 
    #pixweights[nsample:nsample*2] = 0#np.cos(2*psi_obs)
    #pixweights[nsample*2:nsample*3] = 0#np.sin(2*psi_obs) 
    pixweights[::3] = 1
    pixweights[1::3] = np.cos(2*psi_obs)
    pixweights[2::3] = np.sin(2*psi_obs) 
    #del(psi_obs)


    #log.info('Generating noise psd')
    #noisesim, (psdf, psdv) = sim_noise1f(nsample, wnl=300e-6, fknee=1, 
    #                            fsample=fsample, alpha=1, rseed=0, 
    #                            return_psd=True)


    ## defining signal
    if signaltype == 'signal_only':
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=1, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = False
        signal = signal

    elif signaltype == 'signal+1fnoise':
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=1, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = True
        signal = signal + np.concatenate([noisesim]*ndet)*1e11

    elif signaltype == 'signal+wnoise':
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=0, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = False
        signal = signal + np.concatenate([noisesim]*ndet)*1e11
        
    elif signaltype == 'wnoise_only':
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=0, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = False
        signal = np.concatenate([noisesim]*ndet)*1e11

    elif signaltype == '1fnoise_only':
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=1, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = True
        signal = np.concatenate([noisesim]*ndet)*1e11

    else:
        noisesim, (psdf, psdv) = sim_noise1f_old(nsample, wnl=300e-6, fknee=1, 
                                    fsample=fsample, alpha=1, rseed=0, 
                                    return_psd=True)
        psdf = psdf[:len(psdf)//2]
        psdv = psdf[:len(psdv)//2]
        pars['noise_weights_from_psd'] = True
        signal = signal + np.concatenate([noisesim]*ndet)

    if savetod:
        np.savez_compressed(os.path.join(outpath, 'tod_raw'), signal=signal, noise=noisesim)

    signal_in = np.zeros(ndet * nsample, dtype=madam.SIGNAL_TYPE)
    signal_in[:] = signal 

    del(signal)
    del(noisesim)


    log.info('Setting periods')
    nperiod = 100
    periods = np.zeros(nperiod, dtype=int)
    for i in range(nperiod):
        periods[i] = int(nsample//nperiod*i)

    log.debug ('periods={}'.format(periods))

    npsd = np.ones(ndet, dtype=np.int64)
    npsdtot = np.sum(npsd)

    psdstarts = np.zeros(npsdtot)
    npsdbin = len(psdf) 
    log.debug('npsdbin={}'.format(npsdbin))
    psdfreqs = np.arange(npsdbin) * fsample / npsdbin
    psdfreqs[:] = psdf[:npsdbin]
    npsdval = npsdbin * npsdtot
    psdvals = np.ones(npsdval)

    for i in range(npsdtot):
        psdvals[i*npsdbin:(i+1)*npsdbin] = np.abs(psdv[:npsdbin])
    
    log.info('Calling Madam')
    madam.destripe(comm, pars, dets, weights, timestamps, pixels, pixweights,
                   signal_in, periods, npsd, psdstarts, psdfreqs, psdvals)

    if savetod:
        log.info('ascii tod to npz')
        tod_ascii2fits(outpath, remove_asc=True)

    return


def main():
    try:
        # test 2019-06-28
        inpath='/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01/pixel_tod/'
        nside = 1024
        fsample = 1000
        obstime = 86400
        nsample = obstime * fsample
        pars = set_parameters(nside, fsample, nsample, "")

        pars['noise_weights_from_psd'] = True
        outpath = '2019-07-08_toy_signal+1f_psd'
        pol_madam_v2_toy('1fnoise_only', pix_mod=[11,], inpath=inpath, nside=nside, obstime=obstime, 
                     outdir=outpath, savetod=True, pars=pars)

        pars['noise_weights_from_psd'] = False
        outpath = '2019-07-08_toy_signal+1f_nopsd'
        pol_madam_v2_toy('1fnoise_only', pix_mod=[11,], inpath=inpath, nside=nside, obstime=obstime, 
                     outdir=outpath, savetod=True, pars=pars)

        #pol_madam_v2_toy('signal_only',    pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_toy_signal_only')
        #pol_madam_v2_toy('signal+1fnoise', pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_toy_signal+1fnoise')
        #pol_madam_v2_toy('signal+wnoise',  pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_toy_signal+wnoise')
        #pol_madam_v2_toy('wnoise_only',    pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_toy_wnoise_only')
        #pol_madam_v2_toy('1fnoise_only',   pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_toy_1fnoise_only')

        #inpath='/home/klee_ext/kmlee/hpc_data/GBsim_1day_0_5deg/2019-09-01/pixel_tod/'
        #pol_madam_v2('signal_only',    nside=128, pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-04_gb_signal_only_new')
        #pol_madam_v2('signal+1fnoise', nside=128, pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-02_gb_signal+1fnoise')
        #pol_madam_v2('signal+wnoise',  nside=128, pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-02_gb_signal+wnoise_nopsd')
        #pol_madam_v2('wnoise_only',    nside=128, pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-02_gb_wnoise_only_nopsd')
        #pol_madam_v2('1fnoise_only',   nsdie=128, pix_mod=np.arange(23), inpath=inpath, outdir='2019-07-02_gb_1fnoise_only_nopsd')
        #pol_madam_v2('signal+1fnoise', nside=32, observationtime=6000, pix_mod=[7,8,9,10,11,12,13], inpath=inpath, outdir='2019-07-04_gb_signal+1fnoise_test', savetod=True)
        #pol_madam_v2('wnoise_only',  nside=32, observationtime=6000,  pix_mod=[7,8,9,10,11,12,13], inpath=inpath, outdir='2019-07-04_gb_wnoise_only_test', savetod=True)
        #pol_madam_v2('1fnoise_only', nside=32, observationtime=6000,  pix_mod=[7,8,9,10,11,12,13], inpath=inpath, outdir='2019-07-04_gb_1fnoise_only_test', savetod=True)


    except Exception as err:
        print ('There is some error. Sending the information message...')
        print (str(err))
        print (traceback.format_exc())

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login('phynbook@gmail.com', 'womostlgdkfuekmz')

        text = f'An error occured during running "{__file__}" on "{hostname()}".\n'
        text += f'{err}\n{traceback.format_exc()}\n'
        msg = MIMEText(text)
        msg['Subject'] = f'ERROR: {__file__} on {hostname()}'
        s.sendmail("kmlee@hep.korea.ac.kr", "kmlee@hep.korea.ac.kr", msg.as_string())
        s.quit()

if __name__=='__main__':
    #map_madam()
    #pol_madam()
    main()


