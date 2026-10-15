import numpy as np
from scipy.integrate import quad

def E(z, Omega_m=0.28, Omega_L=0.72, Omega_r=9.56e-5):
    """
    Dimensionless Hubble parameter:
      E(z) = H(z)/H0
    for a universe with matter, radiation, curvature, and a cosmological constant.

    E^2(z) = Omega_m*(1+z)^3 + Omega_r*(1+z)^4 + Omega_k*(1+z)^2 + Omega_L,
    where Omega_k = 1 - (Omega_m + Omega_L + Omega_r).
    """
    Omega_k = 1.0 - (Omega_m + Omega_L + Omega_r)
    return np.sqrt(
        Omega_m * (1+z)**3 +
        Omega_r * (1+z)**4 +
        Omega_k * (1+z)**2 +
        Omega_L
    )

def integrand_for_growth(zprime, Omega_m, Omega_L, Omega_r):
    """
    Integrand for the unnormalized growth factor:
      (1 + zprime) / [ E(zprime)^3 ].
    """
    return (1.0 + zprime) / E(zprime, Omega_m, Omega_L, Omega_r)**3

def Dplus_unnorm(z, Omega_m=0.3, Omega_L=0.7, Omega_r=0.0, zmax=10000.0):
    """
    Unnormalized growth factor in flat matter+Lambda cosmology:

      D_+^unnorm(z) = (5 * Omega_m / 2) * E(z)
                      * int_{z}^{zmax} [ (1+z') / E(z')^3 ] dz'.
    """
    val, _ = quad(integrand_for_growth, z, zmax, args=(Omega_m, Omega_L, Omega_r))
    return 2.5 * Omega_m * E(z, Omega_m, Omega_L, Omega_r) * val

def Dplus(z, Omega_m=0.3, Omega_L=0.7, Omega_r=0.0, zmax=10000.0):
    """
    Normalized growth factor D_+(z), imposing D_+(0) = 1.

      D_+(z) = D_+^unnorm(z) / D_+^unnorm(0).
    """
    D0 = Dplus_unnorm(0.0, Omega_m, Omega_L, Omega_r, zmax)
    return Dplus_unnorm(z, Omega_m, Omega_L, Omega_r, zmax) / D0