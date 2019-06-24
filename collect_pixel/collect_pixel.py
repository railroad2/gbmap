import os
import numpy as np

from astropy.io import fits
from progress.bar import Bar

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


def collect_1day_data(path, fname, module, pix_mod, nfile=200):
    #path_signal = '/home/klee_ext/kmlee/hpc_data/GBsim_1day_1deg/2019-09-01/GBtod_cmb145'
    #path_noise = '/home/klee_ext/kmlee/hpc_data/GBsim_1day_1deg/2019-09-01/GBtod_wnoise'
    path_signal = os.path.join(path, 'GBtod_cmb145')
    path_noise = os.path.join(path, 'GBtod_noise')

    flist_signal = os.listdir(path_signal) 
    flist_noise = os.listdir(path_noise)
    flist_signal.sort()
    flist_noise.sort()
    flist_signal = flist_signal[:nfile]
    flist_noise = flist_noise[:nfile]
    
    fsample = 1000

    ra_tot = []
    dec_tot = []
    psi_tot = []
    ix_tot = []
    iy_tot = []
    noise_tot = []

    bar = Bar("Reading tod files", max=len(flist_signal))
    for fs, fn in zip(flist_signal, flist_noise): 
        fname_signal = os.path.join(path_signal, fs)
        fname_noise = os.path.join(path_noise, fn) 
        dat = read_tod(fname_signal, module, pix_mod)
        ut, noise = read_noise(fname_noise)
        ra, dec, psi, ix, iy, pix = dat
        ra_tot = np.concatenate((ra_tot, ra))
        dec_tot = np.concatenate((dec_tot, dec))
        psi_tot = np.concatenate((psi_tot, psi))
        ix_tot = np.concatenate((ix_tot, ix))
        iy_tot = np.concatenate((iy_tot, iy))
        noise_tot = np.concatenate((noise_tot, noise))
        bar.next()

    bar.finish()

    if not os.path.isdir(os.path.join(path, 'pixel_tod')):
        os.mkdir(os.path.join(path, 'pixel_tod'))

    outname = os.path.join(path, 'pixel_tod/'+fname)

    np.savez_compressed(outname, 
        ra=ra_tot, dec=dec_tot, psi=psi_tot, ix=ix_tot, iy=iy_tot, noise=noise_tot)
    
    return

if __name__=='__main__':
    path = '/home/klee_ext/kmlee/hpc_data/GBsim_toy/2019-09-01'
    module = 1
    pixs_mod = [7, 11]
    nfile = 200

    #for pix_mod in pixs_mod:
    #    print ('toy tod pixel#'+str(pix_mod))
    #    outname = 'tod_mod{}pix{}'.format(module, pix_mod)
    #    collect_1day_data(path, outname, module, pix_mod, nfile)
    
    path = '/home/klee_ext/kmlee/hpc_data/GBsim_1day_0_5deg/2019-09-01'

    for pix_mod in pixs_mod:
        print ('gb tod pixel#'+str(pix_mod))
        outname = 'tod_mod{}pix{}'.format(module, pix_mod)
        collect_1day_data(path, outname, module, pix_mod, nfile)

    print ('END')
