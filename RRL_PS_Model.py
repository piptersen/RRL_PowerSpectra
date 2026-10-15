import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from scipy.signal import fftconvolve
import lineTemp
import importlib
import itertools as it
import pickle
import growthFactorCalc as Gz
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
    h=0.7
    # Read the data from the file
    data = np.loadtxt(file_path)
    
    # Extract the columns
    k_data = data[:, 0]*h
    P = data[:, 1] / h**3
    
    model = interpolate.interp1d(k_data, P, fill_value="extrapolate")

    k_shift = k / (((1+z_RRL)/(1+z_21)) * (H_z_calc(z_21)/H_z_calc(z_RRL)))
    matterPS = model(k)
    shiftedPS = model(k_shift)

    
    return(matterPS, shiftedPS, k_shift)


def bias_21(z_21):
    '''
    Interpolates bias values for 21cm line based on data from xxx
    '''
    file_path = 'bias_21.txt'
    data = np.loadtxt(file_path)
    
    z_interp = data[:, 0]
    b_21_interp = data[:, 1]
    
    model = interpolate.interp1d(z_interp, b_21_interp, fill_value="extrapolate")
    return(model(z_21))

def bias_RRL(z_rrl):    
    '''
    Interpolates bias values for RRL line based on data from xxx
    '''
    file_path = 'bias_rrl.txt'
    data = np.loadtxt(file_path)
    
    z_interp = data[:, 0]
    b_rrl_interp = data[:, 1]

    model = interpolate.interp1d(z_interp, b_rrl_interp, fill_value="extrapolate")
    return(model(z_rrl))

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
    return (1420.405751 / (1+x))


def rest_freq(n):
    '''
    Calculates H transition frequency in MHz
    ------------------
    n            (int)   : Transition level
    ------------------
    return [MHz] (float) : Frequency of n transition
    '''
    return(((1 / (n)**2) - (1 / (n+1)**2))*3.288051231582647*1e15*1e-6)

def sfr_schechter(psi, phi_star, psi_star, alpha):
    """
    dn/dpsi [Mpc^-3 (Msun/yr)^-1]
    """
    return phi_star * (psi / psi_star)**alpha * np.exp(-psi / psi_star) / psi_star


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
    return(matterPS)

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
    return(matterPS*np.exp(-1j*k*deltaX))


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
    return(matterPS*np.exp(-1j*k*deltaX))

def growth(z):
    '''
    Growth factor calculation
    ---------------------------
    z (float) : Redshift
    ---------------------------
    return (float) : Growth factor at specific redshift
    '''
    return(1 / (1+z))


def contam_1d(kpar, Pk, L, Delta_x):
    dk = kpar[1] - kpar[0]
    kernel = L * (np.sinc((kpar * L) / (2*np.pi)))**2
    y = np.real(fftconvolve(Pk, kernel, mode='same') * dk)/(2*np.pi)
    return y

def omega_m(z):
    '''
    Matter density parameter as a function of redshift
    ---------------------------------
    z (float) : Redshift
    ---------------------------------
    return (float) : Omega_m at redshift z
    '''
    Om0 = 0.27
    return(Om0 * (1+z)**3  *  (70 / H_z_calc(z))**2)

def convolve_sinc2(P_k, kpar, L):
    dk = kpar[1] - kpar[0]
    r = np.fft.fftshift(np.fft.fftfreq(len(kpar), dk)) * 2*np.pi
    P_r = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(P_k)))

    window_r = np.clip(1 - np.abs(r)/L, 0, 1)
    P_r *= window_r

    P_out = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(P_r)))
    return (P_out)

def make_wide_grids(k_min, k_max, N, k_perp):
    kpar_pos  = np.linspace(k_min, 2*k_max, 2*N)
    kpar_wide = np.concatenate((-kpar_pos[::-1], kpar_pos))

    k_pos  = np.sqrt(kpar_pos**2 + k_perp**2)
    k_wide = np.concatenate((k_pos[::-1], k_pos))

    mu_pos  = kpar_pos / k_pos
    mu_pos[np.isnan(mu_pos)] = 0
    mu_wide = np.concatenate((-mu_pos[::-1], mu_pos))

    return (kpar_wide, k_pos, k_wide, mu_wide)

def PowerScaling(P, k, growthModel, b1, b2, z1, z2, T1, T2, T3, mu = 0, shot21 = True, phase = False, deltaX = 0):
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
    T = (T1 * T2) / T3**2
    rsd1 = omega_m(z1)**0.6 * mu**2
    rsd2 = omega_m(z2)**0.6 * mu**2
    bias = (b1 + rsd1) * (b2 + rsd2)
    G = growthModel(z1) * growthModel(z2)
    
    n_eff_inv = 0
    if shot21 == False:
        n_eff_inv = 181

    phase_shift = 1
    if phase == True:
        phase_shift = np.exp(-1j*k*deltaX)

    P_full = T * ((bias * G  * P) + n_eff_inv) * phase_shift
    return(P_full)

def to_dimensionless(P, k):
    '''
    Converts power spectrum to dimensionless form
    ---------------------------------
    P (array) : Power Spectrum
    k (array) : Wavenumber space
    ---------------------------------
    return (array) : Dimensionless power spectrum
    '''
    return(k**3 * P / (2*np.pi**2))
    
def fullCalc_PS(outputX, outputGrowth, k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, b21 = 1, bRRL = 3, bRRL_2 = 3, k_perp = 0.0, topHat = True, tau_new=[0.1, 0.1]):
    #k_perp = 0.0
    bandwidth /= (1+z_21)
    nu_21 = ztofreq(z_21)
    nu_RRL = rest_freq(n)
    z_RRL = freq21RRL(nu_21, nu_RRL)

    nu_2_RRL = rest_freq(n_2)
    z_2_RRL = freq21RRL(nu_21, nu_2_RRL)

    bRRL = bias_RRL(z_RRL)
    bRRL_2 = bias_RRL(z_2_RRL)

    T_RRL = lineTemp.aFcn(nu_21, z_RRL - (bandwidth/2), z_RRL + (bandwidth/2), n, n+1, tau_new[0])
    T_RRL_2 = lineTemp.aFcn(nu_21, z_2_RRL - (bandwidth/2), z_2_RRL + (bandwidth/2), n_2, n_2+1, tau_new[1])
    T_21 = lineTemp.T21(z_21)
    
    k_parallel = np.linspace(k_min, k_max, N)
    k = np.sqrt(k_parallel**2 + k_perp**2)
    mu = k_parallel / k
    matterPS, shiftedPS, k_shift = modelCalc(k, z_RRL, z_21)

    #Calculates 21cm power spectrum
    #P_21 = PowerScaling(matterPS, k, outputGrowth, bias_21(z_21), bias_21(z_21), z_21, z_21, T_21, T_21, T_21, mu, True)
    P_21_dim = PowerScaling(matterPS, k_parallel, outputGrowth, bias_21(z_21), bias_21(z_21), z_21, z_21, T_21, T_21, T_21, mu)
    P_21 = to_dimensionless(P_21_dim, k)
    
    #Interupts function for non-physical RRL lines
    if (z_RRL < 0 or z_RRL > 6):
        return(k, P_21, np.zeros_like(k),np.zeros_like(k),np.zeros_like(k))
    
    # Distance of individual lines
    x_21 = outputX(z_21)
    if z_RRL < 0.05:
        x_RRL = np.max((outputX(z_RRL), outputX(bandwidth / 2)))
    else:
        x_RRL = outputX(z_RRL)

    # Calculates the Jacobian for the 21cm and RRL power spectra
    J_21 = J_frac_calc(z_21, nu_21, x_21)
    J_RRL = J_frac_calc(z_RRL, nu_21, x_RRL)
    J_scale = J_RRL / J_21

    #P_RRL = J_scale * PowerScaling(P_RRL_term(J_scale, shiftedPS), k, outputGrowth, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL, T_21, mu)
    P_RRL_dim = J_scale * PowerScaling(shiftedPS, k_parallel, outputGrowth, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL, T_21, mu, False, False)
    P_RRL = to_dimensionless(P_RRL_dim, k)
    nu_bandwidth = 1420.405751 * bandwidth / (1 + z_21)**2
    L = (3e5 * (1+z_21) * nu_bandwidth) / (H_z_calc(z_21) * nu_21)

    delta_x_21_RRL = np.absolute(x_21 - x_RRL)

    if (topHat == False):
        # 21cm-RRL Cross power spectrum calculation
        if delta_x_21_RRL < 1000:
            # 1) grids
            kpar_wide, k_pos, k_wide, mu_wide = make_wide_grids(k_min, k_max, N, k_perp)


            # critical sanity checks for FFT correctness
            dk = kpar_wide[1] - kpar_wide[0]

            # 2) underlying PS
            matterPS_pos, _, _ = modelCalc(k_pos, 0, 0)

            P_wide = np.concatenate((matterPS_pos[::-1], matterPS_pos))

            # 3) dimensional scaling (still before phase/FFT)
            P_dim_wide = PowerScaling(
                P_wide, k_wide, outputGrowth,
                bias_21(z_21), bRRL,
                z_21, z_RRL,
                T_21, T_RRL, T_21,
                mu=mu_wide,
                shot21=False,
                phase=False
            )

            # 4) phase
            phase = np.exp(-1j * kpar_wide * delta_x_21_RRL)

            P_pre = P_dim_wide * phase

            # 5) FFT “convolution”
            P_fft = convolve_sinc2(P_pre, kpar_wide, L)

            P_conv = np.real(P_fft)

            # 6) Jacobian and dimensionless conversion
            P_conv *= J_scale

            Delta2_wide = to_dimensionless(P_conv, k_wide)

            P_21_RRL = Delta2_wide[2*N:3*N]
            '''
            #Method 2 = Use the convolution function
            P_dim_wide = PowerScaling(
                P_wide, k_wide, outputGrowth,
                bias_21(z_21), bRRL,
                z_21, z_RRL,
                T_21, T_RRL, T_21,
                mu=mu_wide,
                shot21=True,     # no shot in 21×RRL (your convention)
                phase=True,
                deltaX=delta_x_21_RRL
            )
            P_21_RRL = contam_1d(kpar_wide, P_dim_wide, L, delta_x_21_RRL)[2*N:3*N]
            '''
        else:
            P_21_RRL = np.zeros_like(k)



       # RRL-RRL Cross power spectrum calculation 
        if (z_2_RRL < 0 or z_2_RRL > 6):
            P_RRL_RRL = np.zeros_like(k)
            return (k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)

        if z_2_RRL < 0.05:
            x_2_RRL = np.max((outputX(z_2_RRL), outputX(bandwidth / 2)))
        else:
            x_2_RRL = outputX(z_2_RRL)


        delX_RRL1x2 = np.absolute(x_RRL - x_2_RRL)
        if delX_RRL1x2 > 500:
            P_RRL_RRL = np.zeros_like(k)
            return (k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)
        


        kpar_wide, k_pos, k_wide, mu_wide = make_wide_grids(k_min, k_max, N, k_perp)
        # underlying PS on positive |k|
        _, shiftedPS_pos, _ = modelCalc(k_pos, z_RRL, z_21)
        P_wide = np.concatenate((shiftedPS_pos[::-1], shiftedPS_pos))

        P_dim_wide = PowerScaling(
            P_wide, k_wide, outputGrowth,
            bRRL, bRRL_2,
            z_RRL, z_2_RRL,
            T_RRL, T_RRL_2, T_21,
            mu=mu_wide,
            shot21=False,    # include shot term if you want it (your convention)
            phase=False
        )

        P_pre  = P_dim_wide * np.exp(-1j * kpar_wide * delX_RRL1x2)
        P_conv = np.real(convolve_sinc2(P_pre, kpar_wide, L))

        P_conv *= J_scale
        Delta2_wide = to_dimensionless(P_conv, k_wide)

        P_RRL_RRL = Delta2_wide[2*N:3*N]
        

        return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)
    
    if (topHat == True):
    # Both 21xRRL and RRLxRRL power spectrum, using if condition
        #21cm-RRL lines close in bandwidth?
        if (delta_x_21_RRL <= L and delta_x_21_RRL >= 0):
            
            P_21_RRL = J_scale * ((L-delta_x_21_RRL)/L)*PowerScaling(P_21_RRL_term(J_scale, np.ones_like(k), k, delta_x_21_RRL), k, outputGrowth, bias_21(z_21), bRRL, z_21, z_RRL, T_21, T_RRL, T_21)
        else:
            P_21_RRL = np.zeros_like(k)


        #RRL-RRL lines non-physical (or too close to us)?
        if (z_2_RRL < 0.05 or z_2_RRL > 6):
            P_RRL_RRL = np.zeros_like(k)
            return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)
            
        else:
            #RRL-RRL lines close in bandwidth?
            if np.absolute(z_2_RRL - z_RRL) < (bandwidth*(1+z_RRL)/(1+z_21)):
                x_RRL_2 = outputX(z_2_RRL)
                delX_RRL1x2 = np.absolute(x_RRL - x_RRL_2)
                #P_RRL_RRL = ((L-delX_RRL1x2)/L)*PowerScaling(P_RRL_RRL_term(J_scale, shiftedPS, k_shift, delX_RRL1x2), k, outputGrowth, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL_2, T_21)
                P_RRL_RRL = J_scale * ((L-delX_RRL1x2)/L)*PowerScaling(P_RRL_RRL_term(J_scale, np.ones_like(k_shift), k_shift, delX_RRL1x2), k, outputGrowth, bRRL, bRRL_2, z_RRL, z_RRL, T_RRL, T_RRL_2, T_21)
            else:
                P_RRL_RRL = np.zeros_like(k)
        
            return(k, P_21, P_RRL, P_21_RRL, P_RRL_RRL)


def P_single_test(outputX, outputGrowth, k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, b21 = 1, bRRL = 3, bRRL_2 = 3, k_perp = 0.0, topHat = True, tau_new=[0.1, 0.1], z_RRL_test = 2):
    #k_perp = 0.0
    bandwidth /= (1+z_21)
    nu_21 = ztofreq(z_21)
    nu_RRL = rest_freq(n)
    z_RRL = z_RRL_test

    bRRL = bias_RRL(z_RRL)

    T_RRL = lineTemp.meanT_shifted(tau_new[0], 1, nu_RRL, 20, bandwidth, z_RRL)
    T_21 = lineTemp.T21(z_21)
    
    k_parallel = np.linspace(k_min, k_max, N)
    k = np.sqrt(k_parallel**2 + k_perp**2)
    mu = k_parallel / k
    matterPS, shiftedPS, k_shift = modelCalc(k, z_RRL, z_21)

    if (z_RRL < 0 or z_RRL > 6):
        return(k, np.zeros_like(k))

    x_21 = outputX(z_21)
    x_RRL = np.max((outputX(z_RRL), outputX(bandwidth / 2)))
    print(x_RRL)
    # Calculates the Jacobian for the 21cm and RRL power spectra
    J_21 = J_frac_calc(z_21, nu_21, x_21)
    J_RRL = J_frac_calc(z_RRL, nu_21, x_RRL)
    J_scale = J_RRL / J_21
    P_RRL_dim = J_scale * PowerScaling(shiftedPS, k_parallel, outputGrowth, bRRL, bRRL, z_RRL, z_RRL, T_RRL, T_RRL, T_21, mu, False, False)
    P_RRL = to_dimensionless(P_RRL_dim, k)
    return(k, P_RRL)

def PS_calc_single(n=166, z_21=2, k_min=5e-1, k_max=1, N=10_000, bandwidth=0.05, bRRL=3, k_perp=0.0, topHat=True, fixed_Tau=True, tau=0.1, Te = 8000, ne = 1e1, EM = 1e5, z_RRL_test = 2, z_21_xaxis = True):
    '''
    Wrapper function for single RRL power spectrum calculation based on given parameters
    -------------------------------------------------------
    n                                  (float) : Quantum number
    z_21                               (float) : Redshift of 21cm emission that the RRLs will be contaminating
    k_min                              (float) : Minimum wavenumber (largest scale) for interpolated matter power spectra
    k_max                              (float) : Maximum wavenumber (smallest scale) for ...
    N                                  (float) : Number of points in wavenumber (k) spacing
    bandwidth                          (float) : Redshift resolution of 21cm experiment (if RRL falls in bandwidth it 'contaminates' signal)
    bRRL                               (float) : RRL bias factor
    --------------------------------------------------------
    return (array, array, array, array, array) : wavenumber space, 21cm auto-PS, RRL auto-PS, 21xRRL cross-PS
    '''
    z_thick = np.linspace(0, 6, N)
    # Generate the distance between all pairs of points
    delX_int = np.array([quad(delX, 0, z_rrl) for z_rrl in z_thick])[:,0]
    outputX = interpolate.interp1d(z_thick, delX_int)

    # Generate growth factor for all redshifts
    z_thin = np.linspace(0, 6, 100)
    growth_int = np.array([Gz.Dplus(z) for z in z_thin])
    outputGrowth = interpolate.interp1d(z_thin, growth_int)
    
    #Generate complex Tau
    if fixed_Tau == False:
        import tauCalc_model as tcm
        import importlib
        importlib.reload(tcm)
        tau_new = tcm.tau_out(Te, ne, EM)[1]*1e-6
        
    else:
        tau_new = np.ones(580)*tau

    if z_21_xaxis == True:
        k, P_RRL = P_single_test(outputX, outputGrowth, k_min, k_max, N, z_21, bandwidth, n, n+1, bias_21(z_21), bRRL, bRRL, k_perp, topHat, [tau_new[n-30], tau_new[n-30]], z_RRL_test)
    else:
        k, P_21, P_RRL = fullCalc_PS(outputX, outputGrowth, k_min, k_max, N, z_21, bandwidth, n, n+1, bias_21(z_21), bRRL, bRRL, k_perp, topHat, [tau_new[n-30], tau_new[n-30]])[0:3]
    return(k, P_RRL)

def PS_calc_n_space(k_min = 5e-1, k_max = 1, N = 10_000, z_21 = 2, bandwidth = 0.05, n=166, n_2=167, bRRL = 3, bRRL_2 = 3, k_perp=0.0, topHat = True, fixed_Tau = True, tau=0.1, Te = 8000, ne = 1e1, EM = 1e5):
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
    bRRL                               (float) : RRL bias factor
    bRRL_2                             (float) : Cross term bias factor (should likely be equal to bRRL)
    RRL_separation                     (float) : Frequency resolution for cross RRL terms to corrolate
    --------------------------------------------------------
    return (array, array, array, array, array) : wavenumber space, 21cm auto-PS, Combined RRL auto-PS, 21xRRL cross-PS, RRLxRRL_2 cross-PS
    '''
    n_range = np.arange(max(n, 30), n_2, 1)
    z_thick = np.linspace(0, 6, N)
    # Generate the distance between all pairs of points
    delX_int = np.array([quad(delX, 0, z_rrl) for z_rrl in z_thick])[:,0]
    outputX = interpolate.interp1d(z_thick, delX_int)

    # Generate growth factor for all redshifts
    z_thin = np.linspace(0, 6, 100)
    growth_int = np.array([Gz.Dplus(z) for z in z_thin])
    outputGrowth = interpolate.interp1d(z_thin, growth_int)
    
    # Initilize the power spectra arrays
    comb_P_RRL = np.zeros(N, dtype=np.complex128)
    comb_P_21_RRL = np.zeros_like(comb_P_RRL)
    comb_P_RRL_RRL = np.zeros_like(comb_P_RRL)

    #Generate complex Tau
    if fixed_Tau == False:
        #tau_new = complexTau(ztofreq(z_21), filename)
        #tau_new = optDepthInterp(1e3, 8000, z_21)
        #import tauModel_calc2 as tauMod
        import tauCalc_model as tcm
        import importlib
        importlib.reload(tcm)
        #tau_new = tauMod.optDepthSingleDensity(1e3, 8000, z_21)[1]
        #tau_new = tauMod.integratedOpticalDepth(1e-1, 1e4, 8000, z_21)[1]
        #tau_new = tauMod.optCalc(1e3, 8000)
        tau_new = tcm.tau_out(Te, ne, EM)[1]*1e-6
        
    else:
        tau_new = np.ones(580)*tau
    #Generate all line pairs
    

    pairs = it.combinations(n_range, 2)
    processed_n = set()
    P_z_Plot = []

    for n1, n2 in pairs:
        k, P_21, P_RRL, P_21_RRL, P_RRL_RRL = fullCalc_PS(outputX, outputGrowth, k_min, k_max, N, z_21, bandwidth, n1, n2, bias_21(z_21), bRRL, bRRL_2, k_perp, topHat, [tau_new[n1-30], tau_new[n2-30]])
        
        # Only counts a single pair of lines
        if n1 not in processed_n:
            P_z_Plot.append(np.absolute(P_RRL))
            comb_P_RRL += P_RRL
            comb_P_21_RRL += P_21_RRL
            processed_n.add(n1)
        
        comb_P_RRL_RRL += P_RRL_RRL

    return(k, P_21, comb_P_RRL, comb_P_21_RRL, comb_P_RRL_RRL)#, P_z_Plot)




