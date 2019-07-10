import numpy as np
import healpy as hp
from gbpipe.gbsim import sim_noise1f

def gen_1day_noise(wnl, fknee, fsample, alpha, rseed, outname):
    nsample = 86400000

    noisesim, (psdf, psdv) = sim_noise1f(nsample, wnl=wnl, fknee=fknee,
                                fsample=fsample, alpha=alpha, rseed=rseed, 
                                return_psd=True)
    
    np.savez_compressed(outname, noisesim=noisesim, psdf=psdf, psdv=psdv)

if __name__=='__main__':
    gen_1day_noise(
        wnl=300e-6, 
        fknee=1, 
        fsample=1000, 
        alpha=1, 
        rseed=0, 
        outname='/home/klee_ext/kmlee/gb_noise1f_1day')

    gen_1day_noise(
        wnl=300e-6, 
        fknee=0, 
        fsample=1000, 
        alpha=0, 
        rseed=0, 
        outname='/home/klee_ext/kmlee/gb_wnoise_1day')

