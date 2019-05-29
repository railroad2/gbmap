import os
import sys
import numpy as np
import healpy as hp
import pylab as plt
import time

from astropy.io import fits

from gbpipe import gbparam, gbdir, utils

def read_tod(fname):
    ...

def accumulated_map(fname, nside=1024, intensity_only=True):
    module_idx = 2
    pix_idx_module = 0

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


    ix = data['tod_ix_mod{}'.format(module_idx)] 
    iy = data['tod_iy_mod{}'.format(module_idx)]
    #psi = data['tod_psi_mod{}'.format(module_idx)] # not used for actual observation.

    lst = gbdir.unixtime2lst_linear(ut, lon=param.lon, deg=True)

    # need a routine to get pointing matrix
    # using a single pixel 
    rmat = gbdir.Rot_matrix(el=el, az=az, lst=lst)
    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]

    # beam direction
    r_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    r_obs = gbdir.Rotate(r_det, rmat)

    pix_obs = hp.vec2pix(nside, r_obs[:,0,pix_idx_module], 
                r_obs[:,1,pix_idx_module], r_obs[:,2,pix_idx_module])


    # intensity tod (I + Ip)
    intensity = ix[:, pix_idx_module] + iy[:, pix_idx_module]

    # generating 3 maps - accumulated map, averaged map, hit map
    npix = hp.nside2npix(nside)
    acc_map = np.full(npix, 0.0)
    avg_map = np.full(npix, 0.0)
    hit_map = np.full(npix, 0.0)
    mask = np.full(npix, 0.0)

    for i, p in enumerate(pix_obs):
        acc_map[p] += intensity[i]
        hit_map[p] += 1
        mask[p] = 1

    hit_map_tmp = hit_map.copy()
    hit_map_tmp[hit_map_tmp==0] = 1
    avg_map = acc_map / hit_map_tmp

    #acc_map[acc_map==0] = hp.UNSEEN
    #avg_map[avg_map==0] = hp.UNSEEN

    return acc_map, avg_map, hit_map, mask
    
    # destriping with madam

def accumulated_maps_old(pathname, nside=1024, intensity_only=True):
    flist = os.listdir(pathname)
    flist.sort()

    module_idx = 1
    pix_idx_mod = 0

    ut = []
    az = [] 
    dec = [] 
    ra = [] 
    ix = [] 
    iy = [] 

    st = time.time()
    for i, fname in enumerate(flist):
        print (i, fname)
        hdu = fits.open(os.path.join(pathname,fname), memmap=True)

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
        
        ut.append(data['ut'])
        az.append(data['az'])
        dec.append(data['dec'])
        ra.append(data['ra'])

        ix = np.append(ix, data['tod_ix_mod{}'.format(module_idx)][:,pix_idx_mod])
        iy = np.append(iy, data['tod_iy_mod{}'.format(module_idx)][:,pix_idx_mod])
        #psi = data['tod_psi_mod{}'.format(module_idx)] # not used for actual observation.
        hdu.close()
        del hdu[1].data

    print ('elapsed time to load all the files = {} s'.format(time.time()-st))

    ut = np.array(ut).flatten()
    az = np.array(ut).flatten()
    dec = np.array(ut).flatten()
    ra = np.array(ut).flatten()
    ix = np.array(ut).flatten()
    iy = np.array(ut).flatten()

    st = time.time()
    lst = gbdir.unixtime2lst_1s(ut, lon=param.lon, deg=True)
    print ('elapsed time to get local sidereal time = {} s'.format(time.time()-st))
    st = time.time()

    # need a routine to get pointing matrix
    # using a single pixel 
    st = time.time()
    rmat = gbdir.Rot_matrix(el=el, az=az, lst=lst)
    print ('elapsed time to get rotation matrices = {} s'.format(time.time()-st))

    st = time.time()
    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]
    print ('elapsed time to get detector info = {} s'.format(time.time()-st))
    st = time.time()

    # beam direction
    st = time.time()
    r_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    print ('elapsed time to get beam vectors = {} s'.format(time.time()-st))
    st = time.time()
    r_obs = gbdir.Rotate(r_det, rmat)
    print ('elapsed time to rotate beam vectors = {} s'.format(time.time()-st))

    st = time.time()
    pix_obs = hp.vec2pix(nside, r_obs[:,0,pix_idx_mod], 
                r_obs[:,1,pix_idx_mod], r_obs[:,2,pix_idx_mod])
    print ('elapsed time to get observed sky pixels = {} s'.format(time.time()-st))


    # intensity tod (I + Ip)
    intensity = ix + iy

    # generating 3 maps - accumulated map, averaged map, hit map
    npix = hp.nside2npix(nside)
    acc_map = np.full(npix, 0.0)
    avg_map = np.full(npix, 0.0)
    hit_map = np.full(npix, 0.0)
    mask = np.full(npix, 0.0)


    st = time.time()
    for i, p in enumerate(pix_obs):
        acc_map[p] += intensity[i] 
        hit_map[p] += 1
        mask[p] = 1

    hit_map_tmp = hit_map.copy()
    hit_map_tmp[hit_map_tmp==0] = 1
    avg_map = acc_map / hit_map_tmp

    print ('elapsed time to get maps = {} s'.format(time.time()-st))
    #acc_map[acc_map==0] = hp.UNSEEN
    #avg_map[avg_map==0] = hp.UNSEEN

    return acc_map, avg_map, hit_map, mask
  

def accumulated_maps(pathname, nside=1024, module_idx=1, pix_idx_mod=0, intensity_only=True):
    flist = os.listdir(pathname)
    flist.sort()
    nfile = 1
    #flist = flist[:nfile]

    hdu = fits.open(os.path.join(pathname,flist[0]), memmap=True)
    
    nfile = len(flist)
    nsample = int(hdu[1].header['NAXIS2'])

    hdu.close()

    ut  = np.zeros(nfile * nsample) 
    az  = np.zeros(nfile * nsample) 
    dec = np.zeros(nfile * nsample) 
    ra  = np.zeros(nfile * nsample) 
    ix  = np.zeros(nfile * nsample) 
    iy  = np.zeros(nfile * nsample) 
    pix = np.zeros(nfile * nsample)

    st = time.time()
    for i, fname in enumerate(flist):
        print (i, fname)
        hdu = fits.open(os.path.join(pathname,fname), memmap=True)

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
        
        ut [i*nsample:(i+1)*nsample] = data['ut']
        az [i*nsample:(i+1)*nsample] = data['az']
        dec[i*nsample:(i+1)*nsample] = data['dec']
        ra [i*nsample:(i+1)*nsample] = data['ra']

        ix [i*nsample:(i+1)*nsample] = data['tod_ix_mod_{}'.format(module_idx)][:,pix_idx_mod]
        iy [i*nsample:(i+1)*nsample] = data['tod_iy_mod_{}'.format(module_idx)][:,pix_idx_mod]
        #psi = data['tod_psi_mod{}'.format(module_idx)] # not used for actual observation.
        pix[i*nsample:(i+1)*nsample] = data['tod_pix_mod_{}'.format(module_idx)][:,pix_idx_mod]
        del hdu[1].data
        hdu.close()

    print ('time to load all the files = {} s'.format(time.time()-st))

    """
    st = time.time()
    lst = gbdir.unixtime2lst_linear(ut, lon=param.lon, deg=True)
    print ('time to get local sidereal time = {} s'.format(time.time()-st))
    st = time.time()
    """

    # need a routine to get pointing matrix
    ## using a single pixel 
    st = time.time()
    #rmat = gbdir.Rot_matrix(el=el, az=az, lst=ra)
    rmat = gbdir.Rot_matrix(el=dec, az=ra, lst=0, psi=0, coord='H')
    print ('time to get rotation matrices = {} s'.format(time.time()-st))

    st = time.time()
    theta_det = gbpix['theta'][gbpix['mod']==module_idx]
    phi_det = gbpix['phi'][gbpix['mod']==module_idx]
    psi_det = gbpix['omtffr'][gbpix['mod']==module_idx]
    print ('time to get detector info = {} s'.format(time.time()-st))
    st = time.time()

    ## beam direction
    st = time.time()
    r_det = hp.ang2vec(np.radians(theta_det), np.radians(phi_det))
    print ('time to get beam vectors = {} s'.format(time.time()-st))

    st = time.time()
    r_obs = gbdir.Rotate(r_det, rmat)
    print ('time to rotate beam vectors = {} s'.format(time.time()-st))

    print(r_obs.shape)

    st = time.time()
    #pix_obs = hp.vec2pix(nside, r_obs[:, 0, pix_idx_mod], 
                #r_obs[:, 1, pix_idx_mod], r_obs[:, 2, pix_idx_mod])
    pix_obs = pix
    print ('time to get observed sky pixels = {} s'.format(time.time()-st))

    ## intensity tod (I + Ip)
    intensity = ix + iy
    avg_int = np.average(intensity)
    print ('average of intensity =', avg_int)
    #intensity -= 2.7255#avg_int 

    ## generating 3 maps - accumulated map, averaged map, hit map
    npix = hp.nside2npix(nside)
    acc_map = np.full(npix, 0.0)
    avg_map = np.full(npix, 0.0)
    hit_map = np.full(npix, 0.0)
    mask = np.full(npix, 0.0)

    arr_map = []
    pix_map = []
    for i in range(npix):
        arr_map.append([])
        pix_map.append([])

    # How to improve?
    st = time.time()
    for i, p in enumerate(pix_obs):
        p = int(pix_obs[i])
        acc_map[p] += intensity[i] 
        hit_map[p] += 1
        mask[p] = 1
        arr_map[p].append(intensity[i])
        pix_map[p].append(pix[i])

    hit_map_tmp = hit_map.copy()
    hit_map_tmp[hit_map_tmp==0] = 1
    avg_map = acc_map / hit_map_tmp
    print ('time to get maps = {} s'.format(time.time()-st))

    #acc_map[acc_map==0] = hp.UNSEEN
    #avg_map[avg_map==0] = hp.UNSEEN

    return acc_map, avg_map, hit_map, arr_map, pix_map, mask

if __name__=='__main__':
    arg = sys.argv[1]
    nside = 1024
    #acc_map, avg_map, hit_map, mask = accumulated_map(arg, nside)
    acc_map, avg_map, hit_map, arr_map, pix_map, mask = accumulated_maps(arg, nside)

    avg_map2 = np.zeros(hit_map.shape) 
    std_map2 = np.zeros(hit_map.shape)

    src_map = hp.read_map('/home/klee_ext/kmlee/maps/cmb_rseed42.fits', field=(0,1,2))
    src_map = src_map[0] + np.sqrt(src_map[1]**2 + src_map[2]**2)
    src_map = hp.ud_grade(src_map, nside_out=nside)

    for i, p in enumerate(arr_map):
        pix = pix_map[i]
        if not(p == []):
            avg = np.average(p)
            std = np.std(p)
            if std != 0:
                #print ('p=', p)
                #print ('pix=', pix)
                theta, phi = hp.pix2ang(nside, int(pix[0])) 
                neigh = hp.get_all_neighbours(nside, theta, phi) 
                #print ('values=', src_map[neigh])
                #print ('std=', std)
            avg_map2[i] = avg
            std_map2[i] = std

    hp.mollview(acc_map, xsize=3000)
    plt.savefig('acc_map.png')
    hp.mollview(avg_map, xsize=3000)
    plt.savefig('avg_map.png')
    hp.mollview(hit_map, xsize=3000)
    plt.savefig('hit_map.png')

    hp.mollview(avg_map2, xsize=3000)
    plt.savefig('avg_map2.png')
    hp.mollview(std_map2, xsize=3000)
    plt.savefig('std_map2.png')


    #src_map = src_map[0] + src_map[1] + src_map[2]

    idx = np.argwhere(hit_map>0).flatten()

    avg = avg_map[idx]
    src = src_map[idx]*mask[idx]
    #avg = avg_map
    #src = src_map*mask

    print(avg)
    print(src)
    diff = avg-src
    #diff = diff[np.argwhere(diff!=0)]
    print (diff)
    print (len(diff))
    print (sum(mask))
    print(np.average(diff))
    print(np.std(diff))
    
    #hp.mollview(avg-src, xsize=3000)
    #plt.savefig('diff_map.png')
    #plt.show()

