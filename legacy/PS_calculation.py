import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import lineTemp
import importlib
import itertools as it
importlib.reload(lineTemp)


def modelCalc(k, z_RRL, z_21):
    '''
    One time calculation of interpolated power spectrum from data. Generates evenly spaced regular and shifted matter power spectra 
    based on expansion of the universe (and location of contamiating RRL line).
    ----------------------------------------------
    k      (array)               : wavenumber space to generate power spectra for
    z_RRL  (float)               : Redshift of contaminating radio recombination line
    z_21   (float)               : Redshift of 21cm line being contaminated
    ----------------------------------------------
    return (array, array, array) : Distributed matter PS, shifted PS, and corresponding shifted wavenumber space
    '''
    
    file_path = 'initSB_matterpower_z0.dat'
    
    # Read the data from the file
    data = np.loadtxt(file_path)
    
    # Extract the columns
    k_data = data[:, 0]
    P = data[:, 1]
    
    model = interpolate.interp1d(k_data, P, fill_value="extrapolate")

    k_shift = k * (((1+z_RRL)/(1+z_21)) * (H_z_calc(z_21)/H_z_calc(z_RRL)))
    matterPS = model(k)
    shiftedPS = model(k_shift)

    
    return(matterPS, shiftedPS, k_shift)


def delX(z):
    '''
    Integrand in delta_x calculation to calculate distance between 21cm emitter and RRL emitter
    ---------------------------------------
    z            (float) : Redshift where line is emitted
    ---------------------------------------
    return [Mpc] (float) : Distance between redshift z and observer
    '''
    c = 3e5
    return(c / H_z_calc(z))


def H_z_calc(z):
    '''
    Function calculating the Hubble parameter w/ redshift dependence
    ---------------------------------------
    z                       (array) : redshift
    ---------------------------------------
    return [km / sec / Mpc] (array) : Hubble Constant at each redshift value
    '''
    return(70 * np.sqrt(0.73 + 0.27*(1+z)**3 + 8.24e-5*(1+z)**4))


def freq21RRL(nu21, nuRRL):
    return((nuRRL / nu21) - 1)
def ztofreq(x):
    return (1420 / (1+x))


def rest_freq(n):
    '''
    Calculates H transition frequency in MHz
    ------------------
    n            (int)   : Transition level
    ------------------
    return [MHz] (float) : Frequency of n transition
    '''
    return(((1 / (n)**2) - (1 / (n+1)**2)) * 1.1e5 * 3e10 * 1e-6)


def J_frac_calc(z, nu, x):
    '''
    Calculates the Jacobian for given x and nu
    ------------------------
    z      (float) : Redshift of measured line
    nu     (float) : Observed line frequency (21cm frequency)
    x      (float) : True position of emitting line
    ------------------------
    return (float) : Jacobian of given line at specific position
    '''
    H = H_z_calc(z)
    c = 3e5
    nu *= 1e6
    return(np.absolute((H * nu) / ((1+z) * c * x**2)))

def P_21_term(k):
    '''
    Unused until generalization
    '''
    return(P21)

def P_RRL_term(J_scale, matterPS):
    '''
    RRL Auto-power spectrum calculation
    ---------------------------------------
    J_scale  (float) : Jacobian for corrosponding PS
    matterPS (array) : General matter power spectrum
    ---------------------------------------
    return   (array) : Scaled power spectrum
    '''
    return(J_scale*matterPS)

def P_21_RRL_term(J_scale, matterPS, k, deltaX):
    '''
    RRL 21cm x RRL power spectrum calculation
    ---------------------------------------
    J_scale  (float) : Jacobian for corrosponding PS
    matterPS (array) : General matter power spectrum
    k        (array) : Wavenumber space
    deltaX   (float) : Physical distance between the 21cm line and RRL line
    ---------------------------------------
    return   (array) : Scaled cross power spectrum
    '''
    return(2*J_scale*matterPS*np.exp(1j*k*deltaX))


def P_RRL_RRL_term(J_scale, matterPS, k, deltaX):
    '''
    RRL 21cm x RRL power spectrum calculation
    ---------------------------------------
    J_scale  (float) : Jacobian for corrosponding PS
    matterPS (array) : General matter power spectrum
    k        (array) : Wavenumber space
    deltaX   (float) : Physical distance between the 21cm line and RRL line
    ---------------------------------------
    return   (array) : Scaled cross power spectrum
    '''
    return(2*J_scale*matterPS*np.exp(1j*k*deltaX))

def growth(z):
    '''
    Growth factor calculation
    ---------------------------
    z (float) : Redshift
    ---------------------------
    return (float) : Growth factor at specific redshift
    '''
    return(1 / (1+z))

def PowerScaling(P, k, b1, b2, z1, z2, T1, T2, T3):
    '''
    General Power Spectrum calculation based on scaling factors
    ---------------------------------
    P (array) : Power Spectrum, scaled by Jacobian
    k (array) : Wavenumber space
    b1 (float) : Bias factor for first line
    b2 (float) : Bias factor for second line
    z1 (float) : Redshift for first line
    z2 (float) : Redshift for second line
    T1 (float) : Temperature for first line
    T2 (float) : Temperature for second line
    T3 (float) : Scaling temperature (21cm Temp.)
    ----------------------------------
    return (array) : Scaled and unitless power spectrum
    '''
    bias = b1*b2
    G = growth(z1)*growth(z2)
    T = (T1 * T2) / T3**2
    P_full = bias * G * T * P
    return(k**3 * P_full / (2*np.pi**2))


def fullCalc_PS(outputX, k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, b21 = 1, bRRL = 3, bRRL_2 = 3, RRL_separation = 50):
    
    nu_21 = ztofreq(z_21)
    nu_RRL = rest_freq(n)
    z_RRL = freq21RRL(nu_21, nu_RRL)

    nu_2_RRL = rest_freq(n_2)
    z_2_RRL = freq21RRL(nu_21, nu_2_RRL)

    T_RRL = lineTemp.aFcn(nu_21, z_RRL - (bandwidth/2), z_RRL + (bandwidth/2), N, n, n+1)
    T_RRL_2 = lineTemp.aFcn(nu_21, z_2_RRL - (bandwidth/2), z_2_RRL + (bandwidth/2), N, n_2, n_2+1)
    #z_21_space = np.linspace(z_21-bandwidth/2, z_21+bandwidth/2, N)
    #T_21 = np.sum(lineTemp.T21(z_21_space))/N
    T_21 = lineTemp.T21(z_21)

    k = np.linspace(k_min, k_max, N)
    matterPS, shiftedPS, k_shift = modelCalc(k, z_RRL, z_21)

    #Calculates 21cm power spectrum
    P_21 = PowerScaling(matterPS, k, b21, b21, z_21, z_21, T_21, T_21, T_21)

    if (z_RRL < 0 or z_RRL > 6):
        return(k, P_21, np.zeros_like(k),np.zeros_like(k),np.zeros_like(k))
    
    # Can be in its own code and just pass in outputX since fixed
    x_21 = outputX(z_21)
    x_RRL = outputX(z_RRL)
    
    # Calculates the Jacobian for the 21cm and RRL power spectra
    J_21 = J_frac_calc(z_21, nu_21, x_21)
    J_RRL = J_frac_calc(z_RRL, nu_21, x_RRL)
    J_scale = J_RRL / J_21

    
    P_RRL = PowerScaling(P_RRL_term(J_scale, shiftedPS), k, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL, T_21)
    
    if np.absolute(z_21 - z_RRL) < bandwidth:
        P_21_RRL = PowerScaling(P_21_RRL_term(J_scale, matterPS, k, np.absolute(x_21 - x_RRL)), k, b21, bRRL, z_21, z_RRL, T_21, T_RRL, T_21)
    else:
        P_21_RRL = np.zeros_like(k)

    if (z_2_RRL < 0 or z_2_RRL > 6):
        P_RRL_RRL = np.zeros_like(k)
        return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)
        
    else:
        if np.absolute(nu_2_RRL - nu_RRL) < RRL_separation:
            x_RRL_2 = outputX(z_2_RRL)
            delX_RRL1x2 = np.absolute(x_RRL - x_RRL_2)
            P_RRL_RRL = PowerScaling(P_RRL_RRL_term(J_scale, shiftedPS, k_shift, delX_RRL1x2), k, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL_2, T_21)
        else:
            P_RRL_RRL = np.zeros_like(k)
    
        return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)


def singlePS_calc(k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, b21 = 1, bRRL = 3, bRRL_2 = 3, RRL_separation = 50):
    '''
    Wrapper Function for a single cross-line power spectrum
    -------------------------------------------------------
    k_min                              (float) : Minimum wavenumber (largest scale) for interpolated matter power spectra
    k_max                              (float) : Maximum wavenumber (smallest scale) for ...
    N                                  (float) : Number of points in wavenumber (k) spacing
    z_21                               (float) : Redshift of 21cm emission that the RRLs will be contaminating
    bandwidth                          (float) : Redshift resolution of 21cm experiment (if RRL falls in bandwidth it 'contaminates' signal)
    n                                  (float) : Minimum quantum number
    n_2                                (float) : Maximum quantum number
    b21                                (float) : 21cm bias factor
    bRRL                               (float) : RRL bias factor
    bRRL_2                             (float) : Cross term bias factor (should likely be equal to bRRL)
    RRL_separation                     (float) : Frequency resolution for cross RRL terms to corrolate
    --------------------------------------------------------
    return (array, array, array, array, array) : wavenumber space, 21cm auto-PS, RRL auto-PS, 21xRRL cross-PS, RRLxRRL_2 cross-PS
    '''
    z_thin = np.linspace(0, 6, N)
    delX_int = np.array([quad(delX, 0, z_rrl) for z_rrl in z_thin])[:,0]
    outputX = interpolate.interp1d(z_thin, delX_int)
    k, P_21, P_RRL, P_21_RRL, P_RRL_RRL = fullCalc_PS(outputX, k_min, k_max, N, z_21, bandwidth, n, n_2, b21, bRRL, bRRL_2, RRL_separation)
    return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)


def PS_calc_n_space(k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, b21 = 1, bRRL = 3, bRRL_2 = 3, RRL_separation = 50):
    '''
    Wrapper function for combining power spectrum for all RRL cross terms with delta_n >= 1 based on given parameters
    -------------------------------------------------------
    k_min                              (float) : Minimum wavenumber (largest scale) for interpolated matter power spectra
    k_max                              (float) : Maximum wavenumber (smallest scale) for ...
    N                                  (float) : Number of points in wavenumber (k) spacing
    z_21                               (float) : Redshift of 21cm emission that the RRLs will be contaminating
    bandwidth                          (float) : Redshift resolution of 21cm experiment (if RRL falls in bandwidth it 'contaminates' signal)
    n                                  (float) : Minimum quantum number
    n_2                                (float) : Maximum quantum number
    b21                                (float) : 21cm bias factor
    bRRL                               (float) : RRL bias factor
    bRRL_2                             (float) : Cross term bias factor (should likely be equal to bRRL)
    RRL_separation                     (float) : Frequency resolution for cross RRL terms to corrolate
    --------------------------------------------------------
    return (array, array, array, array, array) : wavenumber space, 21cm auto-PS, Combined RRL auto-PS, 21xRRL cross-PS, RRLxRRL_2 cross-PS
    '''
    n_range = np.arange(n, n_2, 1)
    
    z_thin = np.linspace(0, 6, N)

    # Generate the distance between all pairs of points
    delX_int = np.array([quad(delX, 0, z_rrl) for z_rrl in z_thin])[:,0]
    outputX = interpolate.interp1d(z_thin, delX_int)
    
    # Initilize the power spectra arrays
    comb_P_RRL = np.zeros(N, dtype=np.complex128)
    comb_P_21_RRL = np.zeros_like(comb_P_RRL)
    comb_P_RRL_RRL = np.zeros_like(comb_P_RRL)

    #Generate all line pairs
    pairs = it.combinations(n_range, 2)
    processed_n = set()

    #
    for n1, n2 in pairs:

        k, P_21, P_RRL, P_21_RRL, P_RRL_RRL = fullCalc_PS(outputX, k_min, k_max, N, z_21, bandwidth, n1, n2, b21, bRRL, bRRL_2, RRL_separation)

        # Only counts a single pair of lines
        if n1 not in processed_n:
            comb_P_RRL += P_RRL
            comb_P_21_RRL += P_21_RRL
            processed_n.add(n1)
        
        comb_P_RRL_RRL += P_RRL_RRL
        
    return(k, P_21, comb_P_RRL, comb_P_21_RRL, comb_P_RRL_RRL)




