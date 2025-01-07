import numpy as np
from scipy.integrate import quad

# Hubbert function partial derivatives integration
def hubbert_partial_R(t, k, t0):
    return k * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))**2

def hubbert_partial_k(t, R, k, t0):
    term1 = (1 - k * (t - t0))
    term2 = 2 * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))
    return (R * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))**2) * (term1 - term2)

def hubbert_partial_t0(t, R, k, t0):
    term1 = 1 - 2 / (1 + np.exp(-k * (t - t0)))
    return R * k**2 * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))**2 * term1

# Integration of the partial derivatives over time
def integrate_hubbert_partial_R(t_min, t_max, k, t0):
    result, _ = quad(hubbert_partial_R, t_min, t_max, args=(k, t0))
    return result

def integrate_hubbert_partial_k(t_min, t_max, R, k, t0):
    result, _ = quad(hubbert_partial_k, t_min, t_max, args=(R, k, t0))
    return result

def integrate_hubbert_partial_t0(t_min, t_max, R, k, t0):
    result, _ = quad(hubbert_partial_t0, t_min, t_max, args=(R, k, t0))
    return result

# FEMP function partial derivatives integration
def femp_partial_R0(t, C):
    return -np.log(1 - C) * (1 - C)**t

def femp_partial_C(t, R0, C):
    term1 = (1 - C)**(t-1)
    term2 = t * (1 - C)**t * np.log(1 - C)
    return -R0 * (term1 + term2)

# Integration of the FEMP partial derivatives over time
def integrate_femp_partial_R0(t_min, t_max, C):
    result, _ = quad(femp_partial_R0, t_min, t_max, args=(C,))
    return result

def integrate_femp_partial_C(t_min, t_max, R0, C):
    result, _ = quad(femp_partial_C, t_min, t_max, args=(R0, C))
    return result

# Error propagation using Gaussian formula
def gaussian_error_propagation(partial_derivatives, variances, covariances):
    error_squared = sum((pd**2 * var) for pd, var in zip(partial_derivatives, variances))
    for i in range(len(partial_derivatives) - 1):
        for j in range(i + 1, len(partial_derivatives)):
            error_squared += 2 * partial_derivatives[i] * partial_derivatives[j] * covariances[i][j]
    return np.sqrt(error_squared)


