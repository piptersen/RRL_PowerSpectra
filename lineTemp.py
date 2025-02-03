import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def rest_freq(n):
    '''
    Calculates H transition frequency in MHz
    ------------------
    n            (int)   : Transition level
    ------------------
    return [MHz] (float) : Frequency of n transition
    '''
    return(((1 / (n)**2) - (1 / (n+1)**2)) * 1.1e5 * 3e10 * 1e-6)



def freq_z(nu_emit, z):
    '''
    Calculates cosmological redshift of ground state H frequency
    ------------------
    nu_emit (float) : Ground state frequency of transition
    z       (array) : Redshift space of 21cm emission
    ------------------
    return  (array) : Redshifted frequency of H transition
    '''
    return(nu_emit/(1+z))


def meanT_shifted(tau_L, N_HII, nu_emit, linewidth, bandwidth, z):
    '''
    Calculates the Line temperature of each RRL at the associated redshift. This function is only passed those RRLs (and associated redshifts) that are observable by 
    the 21cm observing experiment.
    ---------------------------------------
    E_SFR     [Solar Mass / yr / Mpc^3] (float) : Comoving Star Formation Rate
    tau_L                               (float) : Optical Depth of RRL in HII region
    N_HII                               (float) : Number of HII obscuring regions w/ optical depth, tau_L
    nu_emit   [MHz]                     (float) : Rest frame frequency of RRL
    linewidth [km / s]                  (float) : Linewidth of RRL, produced by HII region
    bandwidth [MHz]                     (float) : Bandwidth of 21cm observation
    z                                   (array) : Redshift where RRL is observable
    --------------------------------------
    return    [K]                       (array) : Line temperature of particular RRL
    '''
    E_SFR = madauFit(z)
    nu_shift = freq_z(nu_emit, z)
    T_L = 0.0000509244 * (E_SFR / 0.1) * ((tau_L * N_HII) / 0.1) * (nu_shift / 150)**(-2.8) * (linewidth / 20) * ((1+z)/3)**(-2.3)
    return(T_L)



def madauFit(z):
    '''
    Calculates the fit for SFR from redshift (Madau & Dickinson 2014)
    -----------------------------------
    z                                 (array) : Redshift 
    -----------------------------------
    returns [Solar Mass / yr / Mpc^3] (array) : Star formation rate Density
    '''
    return (0.015 * (1 + z)**2.7 / (1 + ((1 + z)/2.9)**5.6))


#Returns the 21cm brightness temperature in Kelvin from a homogeneous universe with xHI=0.01
def T21(z, xHI = .01): #xHI chooses expectation of post reioniation if not specified
    return 0.009*(1+z)**(1/2)*xHI  #eqn 18 in https://arxiv.org/pdf/astro-ph/0608032
    

def aFcn(nu, z_min, z_max, N = 10_000, n_min = 1, n_max = 450):
    '''
    Base 'analytical' calculation function:
    Underlying function takes single frequency and calculates total temperature contamination (from all n) at that frequency. Steps of this calculation are as follows:
    
    1. Find rest frame of each n
    2. Find redshift where n reaches frequency of interest
    3. Calculates temperature of n at calculated redshift
    4. For all n, if corresponding redshift is not 0 < z < 6, sets temperature to 0.
    5. Adds temperature for all remaining n
    -----------------------------------
    nu [MHz] (float) : Discrete frequency of interest
    -----------------------------------
    returns [K] (float) : Total RRL temperature at frequency, nu
    '''
    
    z_space = np.linspace(z_min, z_max, N)
    n_range = np.arange(n_min, n_max, 1)
    
    #Find rest frame frequency for each n transition
    nu_emitted = rest_freq(n_range)

    #Calculate redshift where each n would be shifted to given nu
    z_contam = (nu_emitted / nu) - 1
    
    #Creates a mask for all redshifts outside of observable band
    z_mask = (z_contam >= 0) & (z_contam <= 6)

    #Calculates temperature for each transition
    temp_line = meanT_shifted(1e-1, 1, nu_emitted, 20, 1, z_contam)
    
    #Masks out transitions that fall outside of z = (0,6) band
    temp_line = np.where(z_mask, temp_line, 0)

    #Combines all n temperatures for given nu
    return(np.sum(temp_line))