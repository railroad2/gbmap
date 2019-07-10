import numpy as np
import healpy as hp
import pylab as plt


def gen_strip_mask(nside, unmasked_region):
    mask = np.zeros(hp.nside2npix(nside))

    for r in unmasked_region:
        idx = hp.query_strip(nside, r[0], r[1])     
        mask[idx] = 1.0
        
    return mask


def mask_0(nside):
    unmasked_region = []
    unmasked_region.append([np.radians(90-64), np.radians(90+7)])

    mask = gen_strip_mask(nside, unmasked_region)

    return mask


def mask_1(nside):
    unmasked_region = []
    unmasked_region.append([np.radians(90-61.5), np.radians(90-36.25)])
    unmasked_region.append([np.radians(90-33.12), np.radians(90+5.07)])

    mask = gen_strip_mask(nside, unmasked_region)

    return mask


def mask_2(nside):
    unmasked_region = []
    unmasked_region.append([np.radians(90-60.), np.radians(90-35.)])
    unmasked_region.append([np.radians(90-32.), np.radians(90+4.)])

    mask = gen_strip_mask(nside, unmasked_region)

    return mask
 

def mask_3(nside):
    unmasked_region = []
    unmasked_region.append([np.radians(90-55.), np.radians(90-40.)])
    unmasked_region.append([np.radians(90-25.), np.radians(90+0.)])

    mask = gen_strip_mask(nside, unmasked_region)

    return mask


def main():
    nside = 1024
    mask0 = mask_0(nside)
    mask1 = mask_1(nside)
    mask2 = mask_2(nside)
    mask3 = mask_3(nside)

    hp.write_map(filename='mask_1024.fits', 
                 m=[mask0, mask1, mask2, mask3], 
                 column_names=('edge_mask', 'tight_mask', 'medium_mask', 'loose_mask'))

if __name__=='__main__':
    main()
