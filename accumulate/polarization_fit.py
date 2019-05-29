import os
import sys

import numpy as np
import healpy as hp
import pylab as plt
from astropy.io import fits

from gbpipe import gbparam, gbdir, utils

def doit():
    if len(sys.argv)==1:
        pwd = '/home/klee_ext/kmlee/hpc_data/2019-05-16_GBsim_1week/2019-05-01/GBtod_cmb145/'
        fname = pwd + 'GBtod_cmb145_2019-05-01T00:00:00.000_2019-05-01T00:10:00.000.fits'
    elif len(sys.argv)==2:
        fname = sys.argv[1]
    hdu = fits.open(fname)
    header = hdu[1].header

    el = int(header['el'])
    isot0 = header['isot0']
    isot1 = header['isot1']
    fsample = int(header['fsample'])
    nmodules = list(map(int, header['nmodules'].split(',')))
    nmodpixs = list(map(int, header['nmodpixs'].split(',')))

    data = hdu[1].data
    param = gbparam.GBparam()
    gbpix = param.pixinfo

    ut = data['ut']
    az = data['az']
    dec = data['dec']
    ra = data['ra']
    ix1 = data['tod_ix_mod1']
    iy1 = data['tod_iy_mod1']
    psi1 = data['tod_psi_mod1']

    lst = gbdir.unixtime2lst_1s(ut)  

    #need routines for 1 detector

    # module #n
    module_idx = 1
    rmat = gbdir.Rot_matrix(el=el, az=az, lst=lst) 
    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]

    # beam direction
    r_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    # polarization vector
    p_det = gbdir.psi2vec_xp(r_det, psi_det)

    r_obs = gbdir.Rotate(r_det, rmat)
    p_obs = gbdir.Rotate(p_det, rmat)
    psi_obs = gbdir.angle_from_meridian(r_obs, p_obs) # Debugging is needed. 

    nside = 1024
    pix_obs = hp.vec2pix(nside, r_obs[:,0,:], r_obs[:,1,:], r_obs[:,2,:])

    # find a pixel with many hits 
    uniq, cnt = np.unique(pix_obs, return_counts=True)
    pix_target = uniq[cnt==max(cnt)][0]

    ix_target = ix1[pix_obs==pix_target]
    iy_target = iy1[pix_obs==pix_target]
    psi_target = psi_obs[pix_obs==pix_target]

    ix_target -= 2.7255/2
    iy_target -= 2.7255/2

    print ('max(cnt)', max(cnt))
    print ('pix_target', pix_target)
    print ('ix_target', ix_target)
    print ('iy_target', iy_target)
    print ('psi_target', psi_target)
    print ('intensity', ix_target + iy_target)
    print ('Q', ix_target - iy_target)
    print ('U', np.sqrt(ix_target * iy_target))

    #plt.plot(psi_target, ix_target, '.'); 
    plt.plot(psi_target, ix_target-iy_target, '.'); 
    plt.show()
    

if __name__=="__main__":
    doit()

