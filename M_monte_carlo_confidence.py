
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from plotnine import *
from itertools import product
from scipy.stats import shapiro, kstest, anderson, norm, lognorm, beta, truncnorm
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

from util import df_to_csv_int
from M_prod_model import hubbert_deriv, femp_deriv, hubbert_model, femp, safe_exp
from util import df_to_csv_int, save_fig_plotnine
###################################################Purpose#######################################################

# This script aims to estimate the confidence interval for the production and cumulative production values based on a
# a Monte Carlo simulation
# First data is loaded, parameter distribution initiated, and then the Monte Carlo simulation is run for 10000 iterations
# From the resulting production / integrals the confidence intervals are drawn

####################################################################################################################


#################################################Parameters#####################################################

iterations = 10000

sig = .05

lower_percentile = sig * 100
upper_percentile = (1-sig) *100

####################################################Functions#####################################################

def regularize_parameters(alpha, beta, epsilon=1e-5):
    """
    Regularize alpha and beta to ensure they are positive.
    
    Parameters:
    - alpha: Calculated alpha parameter.
    - beta: Calculated beta parameter.
    - epsilon: Small positive offset to ensure positivity.
    
    Returns:
    - Regularized alpha and beta.
    """
    return max(alpha, epsilon), max(beta, epsilon)


def trunc_norm(value, error, a = 0, b=np.inf):
     a_s, b_s = (a-value) / error, (b-value) / error
     return truncnorm.rvs(a_s, b_s, loc=value, scale=error, size=iterations)

def initiate_mc_per_mine(t_max, model, P1_value, P2_value, P3_value, P1_err, P2_err, P3_err):
     t = np.arange(t_max)

     if model == 'hubbert':
            p1 = trunc_norm(P1_value, P1_err)
            p2 = trunc_norm(P2_value, P2_err)
            p3 = trunc_norm(P3_value, P3_err)
               
            # calculate for each iteration the production and the integral. Estimate for every time step the 95% confidence interval interval
            f = np.zeros((iterations, t_max))
            F = np.zeros((iterations, t_max))

            for i in range(iterations):
                f[i, :], F[i, :] = hubbert_deriv(t, p1[i], p2[i], p3[i]), hubbert_model(t, p1[i], p2[i], p3[i])
            

            f_mean = np.zeros(t_max)
            F_mean = np.zeros(t_max)
            f_err = np.zeros(t_max)
            F_err = np.zeros(t_max)
            f_lower_ci = np.zeros(t_max)
            f_upper_ci = np.zeros(t_max)
            F_lower_ci = np.zeros(t_max)
            F_upper_ci = np.zeros(t_max)

            f_p_star = hubbert_deriv(t, P1_value, P2_value, P3_value)
            F_p_star = hubbert_model(t, P1_value, P2_value, P3_value)

            for i in range(t_max):
                 f_boot = bootstrap(data=f[:, i].reshape(-1, 1).T, statistic=np.std, n_resamples=1000, method='bca')
                 f_mean[i], f_err[i], f_lower_ci[i], f_upper_ci[i] =  (f_boot.confidence_interval.high + f_boot.confidence_interval.low) / 2, f_boot.standard_error, f_boot.confidence_interval.low, f_boot.confidence_interval.high
                 F_boot = bootstrap(data=F[:, i].reshape(-1, 1).T, statistic=np.std, n_resamples=1000, method='bca')
                 F_mean[i], F_err[i], F_lower_ci[i], F_upper_ci[i] =  (F_boot.confidence_interval.high + F_boot.confidence_interval.low) / 2, F_boot.standard_error, F_boot.confidence_interval.low, F_boot.confidence_interval.high


            

             # assert that the means are not nan
            assert not np.isnan(f_mean).any(), 'f_mean contains NaNs'
            assert not np.isnan(F_mean).any(), 'F_mean contains NaNs'

           

            f_lower_ci_norm = (f_lower_ci / f_mean) 
            f_upper_ci_norm = (f_upper_ci / f_mean) 

            F_lower_ci_norm = (F_lower_ci / F_mean) 
            F_upper_ci_norm = (F_upper_ci / F_mean) 



            return t, f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci, f_lower_ci_norm, f_upper_ci_norm, F_lower_ci_norm, F_upper_ci_norm, f_p_star, F_p_star

          
     elif model == 'femp':
           
            p1 = trunc_norm(P1_value, P1_err)
            p2 = trunc_norm(P2_value, P2_err, b=1)
    

            # calculate for each iteration the production and the integral. Estimate for every time step the 95% confidence interval interval
            f = np.zeros((iterations, t_max))
            F = np.zeros((iterations, t_max))

            for i in range(iterations):
                f[i, :], F[i, :] = femp_deriv(t, p1[i], p2[i]), femp(t, p1[i], p2[i])
            
            f_mean = np.zeros(t_max)
            F_mean = np.zeros(t_max)
            f_err = np.zeros(t_max)
            F_err = np.zeros(t_max)
            f_lower_ci = np.zeros(t_max)
            f_upper_ci = np.zeros(t_max)
            F_lower_ci = np.zeros(t_max)
            F_upper_ci = np.zeros(t_max)

            f_p_star = femp_deriv(t, P1_value, P2_value)
            F_p_star = femp(t, P1_value, P2_value)

            for i in range(t_max):
                 f_boot = bootstrap(data=f[:, i].reshape(-1, 1).T, statistic=np.std, n_resamples=1000, method='bca')
                 f_mean[i], f_err[i], f_lower_ci[i], f_upper_ci[i] =  (f_boot.confidence_interval.high + f_boot.confidence_interval.low) / 2, f_boot.standard_error, f_boot.confidence_interval.low, f_boot.confidence_interval.high
                 F_boot = bootstrap(data=F[:, i].reshape(-1, 1).T, statistic=np.std, n_resamples=1000, method='bca')
                 F_mean[i], F_err[i], F_lower_ci[i], F_upper_ci[i] =  (F_boot.confidence_interval.high + F_boot.confidence_interval.low) / 2, F_boot.standard_error, F_boot.confidence_interval.low, F_boot.confidence_interval.high


            f_lower_ci_norm = f_lower_ci / f_mean
            f_upper_ci_norm = f_upper_ci / f_mean

            F_lower_ci_norm = F_lower_ci / F_mean
            F_upper_ci_norm = F_upper_ci / F_mean

            return np.arange(t_max), f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci, f_lower_ci_norm, f_upper_ci_norm, F_lower_ci_norm, F_upper_ci_norm, f_p_star, F_p_star

     else:
            raise ValueError('Model not recognized')
          

#####################################################Main Function#################################################

def main_calc_error():

    res = pd.read_json(r'data\int\production_model_fits_trans.json')

    res['Start_up_year'] = res['Start_up_year'].astype(int)
    
    # Delta periods is the difference between the year of the data and 2022
    res['Delta_periods'] = 2022 - res['Start_up_year']



    grouped = res.groupby(['Prop_id', 'Target_var', 'Model'])

    # Prepare to store the results
    ua = []

    # Loop over the grouped DataFrame
    for (id, t_, m), group in tqdm(grouped, 'Calculating errors'):
            # If the group is empty, continue
            if group.empty:
                continue

            # Extract the relevant values
            t_max = group['Delta_periods'].iloc[0]

            # Pass the values to the initiation function
            t, f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci, f_lower_ci_norm, f_upper_ci_norm, F_lower_ci_norm, F_upper_ci_norm, f_p_star, F_p_star  = initiate_mc_per_mine(
                t_max, m, **group[['P1_value', 'P2_value', 'P3_value', 'P1_err', 'P2_err', 'P3_err']].iloc[0]
            )

            # Append the result to the errors list
            ua.append(pd.DataFrame({
                'Prop_id': np.array([id]*len(t)),
                'Target_var': np.array([t_]*len(t)),
                'Model': np.array([m] * len(t)),
                'Time_period': t,
                'f_mean': f_mean,
                'f_err': f_err,
                'f_lower_ci': f_lower_ci,
                'f_upper_ci': f_upper_ci,
                'F_mean': F_mean,
                'F_err': F_err,
                'F_lower_ci': F_lower_ci,
                'F_upper_ci': F_upper_ci,
                'f_lower_ci_norm': f_lower_ci_norm,
                'f_upper_ci_norm': f_upper_ci_norm,
                'F_lower_ci_norm': F_lower_ci_norm,
                'F_upper_ci_norm': F_upper_ci_norm,
                'f_p_star': f_p_star,
                'F_p_star': F_p_star,
                'Start_up_year': np.array([group['Start_up_year'].values[0]] * len(t))
            }))

    # Create the resulting DataFrame
    res = pd.concat(ua, ignore_index=True)

    df_to_csv_int(res, 'cumprod_mc_confidence')


    pass






if __name__ == '__main__':
    main_calc_error()
    
    pass

