import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import interpolate
import numpy as np
import pickle

def rest_freq(n):
    '''
    Calculates H transition frequency in MHz
    ------------------
    n            (int)   : Transition level
    ------------------
    return [MHz] (float) : Frequency of n transition
    '''
    return(((1 / (n)**2) - (1 / (n+1)**2)) * 1.1e5 * 3e10 * 1e-6)
def intDepth(tau, n, V, ff=1):
    return((V/3e5) * rest_freq(n) * tau * 1.064467 * ff)


def tau_test(T, EM, n, bbeta):
    chi = 1.58e5 / (T * ((n)**2))
    return(2.046e6 * np.exp(chi)* (T**(-5/2)) * EM * bbeta)

def beta_n(b, n, T):
    bn = b
    bn_p = np.roll(b, -1)
    n_p = np.roll(n, -1)
    coeff = 3.168 * 1e-6 * n**3 * T * (bn_p - bn) / ((n_p - n) * bn)
    return (1-coeff)
def tau_C(Te, n, EM):
    #Formula from blue book eq 2.95
    return(0.08235 * Te**(-1.35) * (rest_freq(n)/1000)**(-2.1) * EM)

def carbonModel(n_vals):
    from scipy.optimize import curve_fit
    data = np.loadtxt("models/bbeta_carbon.txt")

    n_spaced = data[:, 0]
    bbeta_spaced = data[:, 1]

    sort_id = np.argsort(n_spaced)
    n_spaced = n_spaced[sort_id]
    bbeta_spaced = bbeta_spaced[sort_id]

    def asym_gaussian(n, amp, center, sig_left, sig_right):
        sig = np.where(n < center, sig_left, sig_right)
        return amp * np.exp(-0.5 * ((n - center) / sig)**2)

    def carbon_bbeta_model(n, C, Apos, npos, sigpos, Aneg, nneg, sigleft, sigright):
        pos = Apos * np.exp(-0.5 * ((n - npos) / sigpos)**2)
        neg = asym_gaussian(n, Aneg, nneg, sigleft, sigright)
        return C + pos - neg

    # Rough initial guess; tune based on your digitized points
    p0 = [
        0.0,    # C
        8.0,    # Apos
        170.0,  # npos
        35.0,   # sigpos
        28.0,   # Aneg
        320.0,  # nneg
        50.0,   # sigleft: sharp left side
        250.0   # sigright: broad right tail
    ]

    bounds = (
        [-10, 6,   120, 10,  26,   220, 20,  80],
        [ 10, 15,  220, 90, 40,  600, 120, 600]
    )
    sigma = np.ones_like(bbeta_spaced)

    # Care most about the trough
    trough_mask = (n_spaced > 240) & (n_spaced < 380)
    sigma[trough_mask] = 0.15

    # Some weight on the positive peak
    peak_mask = (n_spaced > 120) & (n_spaced < 230)
    sigma[peak_mask] = 0.35

    # Do not let the right tail dominate
    tail_mask = n_spaced > 500
    sigma[tail_mask] = 5.0

    popt, pcov = curve_fit(
        carbon_bbeta_model,
        n_spaced,
        bbeta_spaced,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=False,
        maxfev=10000
    )

    return(carbon_bbeta_model(n_vals, *popt))

def tau_out(Te, ne, EM = 1e5, i_n = 0):
    # Load models
    bn = np.load('models/bn_hrrl_v2.npy')
    beta = np.load('models/beta_hrrl_v2.npy')
    bn_head = pickle.load(open('models/bn_hrrl_head_v2.p', "rb" ))


    #Pulling all the T and n values
    Te_arr = np.array(bn_head['Te_v'])
    ne_arr = np.array(bn_head['ne_v'])
    
    #Finding the model index for the data table closest to the chosen temperature and density 
    model_close_id = np.argmin((Te_arr - Te)**2 + (ne_arr - ne)**2)

    #print('temp', Te_arr)
    #print('dens', ne_arr)
    #print('Model closest to chosen Te and ne: Te =', Te_arr[model_close_id], 'K, ne =', ne_arr[model_close_id], 'cm^-3')

    #Pulling the n values from the bn tables
    n_vals = bn[model_close_id, 0, :-1]
    
    #Choosing the first set of bn values from the model
    bn_vals = bn[model_close_id, 1, :-1]
    #betan_vals = beta_n(bn_vals, n_vals, Te)
    betan_vals = beta[model_close_id, :, 1]

    bbeta_vals = bn_vals * betan_vals

    if Te < 1000:
        '''
        file_path = 'models/bbeta_carbon_2.txt'
        
        data = np.loadtxt(file_path)
        
        n_spaced = data[:, 0]
        b_spaced = data[:, 1]
        bbeta_spaced = b_spaced * beta_n(b_spaced, n_spaced, Te)

        model = interpolate.interp1d(n_spaced, bbeta_spaced, fill_value="extrapolate")
        bbeta_vals = model(n_vals)
        #bbeta_vals = b_vals * beta_n(b_vals, n_vals, Te)'''

        file_path = 'models/carbon_ne005_te100.txt'
        
        data = np.loadtxt(file_path)
        
        n_spaced = data[:, 0]
        bbeta_spaced = data[:, 1]
        #bbeta_spaced = b_spaced * beta_n(b_spaced, n_spaced, Te)

        model = interpolate.interp1d(n_spaced, bbeta_spaced, fill_value="extrapolate")
        bbeta_vals = model(n_vals)
        #bbeta_vals = b_vals * beta_n(b_vals, n_vals, Te)
        #bbeta_vals = carbonModel(n_vals)
        
        #plt.plot(n_vals, bbeta_vals, label = 'Carbon', ls = '-', c = "#000000", lw = 2.5)
        
        #n_vals = np.concatenate((np.arange(30, 80, 1), data[:, 0]))
        '''
        data = np.loadtxt("models/crrl_bn_beta_T85K_n80-500.dat", comments="#")
        n_vals = data[:, 0]      # file column 0: n values
        b = data[:, i_n]      # file column 2: b_n for ne=1e-2
        beta = data[:, i_n+1]   # file column 3: beta_n for ne=1e-2
        bbeta_vals = b * beta
        '''
        #print(len(n_vals), len(bbeta_vals))
    tau_calc = tau_test(Te, EM, n_vals, bbeta_vals) * np.exp(-tau_C(Te, n_vals, EM))

    return(n_vals, tau_calc)