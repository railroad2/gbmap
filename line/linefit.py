import os
import numpy as np
import healpy as hp
#import pylab as plt
from progress.bar import Bar

from astropy.io import fits

from gbpipe import gbdir, gbparam
from gbpipe.gbsim import sim_noise1f
from gbpipe.utils import set_logger

from mpi4py import MPI
import libmadam_wrapper as madam


def read_tod(fname, module, pix):
    module_idx = module
    pix_idx_mod = pix

    hdu = fits.open(fname)
    header = hdu[1].header

    el = int(header['el'])
    isot0 = header['isot0']
    isot1 = header['isot1']
    fsample = int(header['fsample'])
    nmodules = list(map(int, header['nmodules'].split(',')))
    nmodpixs = list(map(int, header['nmodpixs'].split(',')))

    data = hdu[1].data

    ut = data['ut']
    az = data['az']
    dec = data['dec']
    ra = data['ra']
    psi_zen = data['psi_equ']

    ix = data['tod_ix_mod_{}'.format(module_idx)][:,pix_idx_mod]
    iy = data['tod_iy_mod_{}'.format(module_idx)][:,pix_idx_mod]
    #psi = data['tod_psi_mod{}'.format(module_idx)] # not used for actual observation.
    pix = data['tod_pix_mod_{}'.format(module_idx)][:,pix_idx_mod]
    del (hdu[1].data)
    hdu.close()

    #return pix, ix, iy 
    return ra, dec, psi_zen, ix, iy, pix


def read_noise(fname):
    hdu = fits.open(fname)
    header = hdu[1].header

    isot0 = header['isot0']
    isot1 = header['isot1']
    fsample = int(header['fsample'])

    data = hdu[1].data

    ut = data['ut']
    noise = data['N1f']

    del (hdu[1].data)
    hdu.close()

    #return pix, ix, iy 
    return ut, noise 


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
    
    
def collect_data(dat, module, pix_mod, nside=1024):
    ra, dec, psi_zen, ix, iy, pix = dat

    npix = nside * nside * 12

    pix_obs, psi_obs = pixels_for_detector(module, pix_mod, ra, dec, psi_zen)

    arr_map = [[] for i in range(npix)]
    psi_map = [[] for i in range(npix)]
    nhit = np.zeros(npix)
    npol = np.zeros(npix)

    for i, p in enumerate(pix_obs):
        arr_map[p].append(ix[i]-iy[i])
        psi_map[p].append(psi_obs[i])
    
    return arr_map, psi_map


def collect_data_multi(ra, dec, psi_zen, signal, module, pix_mod, arr_map, psi_map, nside=1024):
    npix = nside * nside * 12

    pix_obs, psi_obs = pixels_for_detector(module, pix_mod, ra, dec, psi_zen, nside=nside)

    #arr_map = [[] for i in range(npix)]
    #psi_map = [[] for i in range(npix)]
    nhit = np.zeros(npix)
    npol = np.zeros(npix)

    for i, p in enumerate(pix_obs):
        arr_map[p].append(signal[i])
        psi_map[p].append(psi_obs[i])
    
    #return arr_map, psi_map
    return


def calculate_baseline(signal, base_length):
    #옥수수빵
    length = len(signal)
    baseline = np.zeros(length)
    idx = np.arange(length)
    
    for i in range(int(np.ceil(length/base_length))):
        sig_tmp = signal[i*base_length:(i+1)*base_length]
        idx_tmp = idx[i*base_length:(i+1)*base_length]
        offset = np.average(sig_tmp)
        baseline[idx_tmp] = offset

    return baseline


def get_QU(arr, psi):
    Qs = np.array(arr) 
    psis = np.array(psi)
    M = np.array([np.cos(2*psis), np.sin(2*psis)]).T
    Mi = np.linalg.pinv(M)
    Q, U = np.matmul(Mi, Qs)

    return Q, U


def compare_with_org(Q, U, nside=1024):
    orgname = '/home/klee_ext/kmlee/maps/cmb_rseed42_0_5deg.fits'
    orgmap = hp.read_map(orgname, field=None)
    orgmap = hp.ud_grade(orgmap, nside_out=nside)
    Q_org = orgmap[1]
    U_org = orgmap[2]

    Q_diff = np.full(len(Q_org), hp.UNSEEN)
    U_diff = np.full(len(U_org), hp.UNSEEN)

    for i in range(len(Q_org)):
        if Q[i] != hp.UNSEEN: 
            Q_diff[i] = Q_org[i] - Q[i]
            U_diff[i] = U_org[i] - U[i]
        if abs(Q_diff[i]) > 1e-5:
            Q_diff[i] = hp.UNSEEN
        if abs(U_diff[i]) > 1e-5:
            U_diff[i] = hp.UNSEEN

    hp.mollview(Q_diff)
    hp.mollview(U_diff)

    plt.show()


def test_singlefile():
    fname = '/home/klee_ext/kmlee/hpc_data/GBsim_1day_beam/2019-09-01/GBtod_cmb145/GBtod_cmb145_2019-09-01T00:00:00.000_2019-09-01T00:10:00.000.fits'
    module = 1
    pix_mod = 0

    dat = read_tod(fname, module, pix_mod)
    ra, dec, psi, ix, iy, pix = dat
    arr_map, psi_map = collect_data(dat, module, pix_mod)

    npix = len(arr_map)

    avg_map = np.zeros(npix)
    std_map = np.zeros(npix)
    Q_map = np.zeros(npix)
    U_map = np.zeros(npix)

    for i in range(npix):
        Q_map[i] = hp.UNSEEN
        U_map[i] = hp.UNSEEN
        if len(arr_map[i]) > 1: 
            Q, U = get_QU(arr_map[i], psi_map[i])
            Q_map[i] = Q
            U_map[i] = U
            avg_map[i] = np.average(arr_map[i]) 
            std_map[i] = np.std(arr_map[i])
    
    hp.mollview(Q_map, min=-1.77987e-6, max=2.0722e-6)
    hp.mollview(U_map, min=-2.1874e-6, max=1.96767e-6)
    compare_with_org(Q_map, U_map, nside)
    plt.show()


def test_1day():
    module = 1
    pix_mod = 0
    nside = 1024
    npix = hp.nside2npix(nside)
    fsample = 1000
    length = 86400000

    log = set_logger()

    arr_map = [[] for i in range(npix)]
    psi_map = [[] for i in range(npix)]

    log.info('loading tod')
    dat = np.load('/home/klee_ext/kmlee/pixel_tod/tod_mod1pix0.npz')

    ra = dat['ra'][:length]
    dec = dat['dec'][:length]
    psi = dat['psi'][:length]
    ix = dat['ix'][:length]
    iy = dat['iy'][:length]
    noise = dat['noise'][:length]
    
    log.info('generating noise')
    noisesim, (psdf, psdv) = sim_noise1f(length, wnl=300e-1, fknee=1, fsample=fsample, alpha=1, rseed=0, return_psd=True)
    signal = ix + iy + noisesim #ix - iy + noise

    
    log.info('calculating baseline')
    baseline = calculate_baseline(signal, 3000)

    outpath_pre = '/home/klee_ext/kmlee/test_linefit/linefit'
    outpath = outpath_pre + ('_%03d/' % (0))
    i = 0
    while os.path.isdir(outpath):
        i += 1
        outpath = outpath_pre + ('_%03d/' % (i))

    os.mkdir(outpath)

    np.savez_compressed(outpath + 'tod_linear', signal=signal, baseline=baseline)

    signal = signal - baseline
    log.info('collecting pixel data')
    collect_data_multi(ra, dec, psi, signal, module, pix_mod, arr_map, psi_map, nside=nside)

    #plt.plot(signal)
    #plt.plot(noise)
    #plt.plot(baseline)

    #npix = len(arr_map)
    pixs = np.arange(npix)#hp.query_disc(nside=nside, vec=hp.ang2vec(np.radians(90-28), 0), radius=np.radians(30))

    avg_map = np.zeros(npix)
    std_map = np.zeros(npix)

    I_map = np.full(npix, hp.UNSEEN)
    Q_map = np.full(npix, hp.UNSEEN)
    U_map = np.full(npix, hp.UNSEEN)

    log.info('Making map')
    bar = Bar("Making maps", max=100)
    for i in pixs: #range(npix):
        if len(arr_map[i]) > 1: 
            #print (i)
            Q, U = get_QU(arr_map[i], psi_map[i])
            Q_map[i] = Q
            U_map[i] = U

            avg_map[i] = np.average(arr_map[i]) 
            std_map[i] = np.std(arr_map[i])

        if not(i%(len(pixs)//100)):
            bar.next()

    bar.finish()
    log.info('Making map finished')
    

    hp.write_map(outpath + 'IQU_linear.fits', (I_map, Q_map, U_map))

    #hp.mollview(Q_map, min=-1.77987e-6, max=2.0722e-6)
    #hp.mollview(U_map, min=-2.1874e-6, max=1.96767e-6)
    #compare_with_org(Q_map, U_map, nside)
    #plt.show()
    log.info('The end')

    return


if __name__=='__main__':
    test_1day()

