
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from plotnine import *
from itertools import product
from scipy.stats import shapiro, kstest, anderson, norm, lognorm, beta, truncnorm


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


def initiate_mc_per_mine(t_max, model, P1_value, P2_value, P3_value, P1_err, P2_err, P3_err):
     t = np.arange(t_max)

     if model == 'hubbert':
            # P1 lognormal distributed
            p1_log_mean = np.log(P1_value / np.sqrt(1 + (P1_err / P1_value) ** 2))
            p1_log_std = np.sqrt(np.log(1 + (P1_err / P1_value) ** 2))

            p1_log = np.random.normal(p1_log_mean, p1_log_std, iterations)

            p1 = safe_exp(p1_log)

            # P2 log normal distributed
            p2_log_mean = np.log(P2_value / np.sqrt(1 + (P2_err / P2_value) ** 2))
            p2_log_std = np.sqrt(np.log(1 + (P2_err / P2_value) ** 2))
            p2_log = np.random.normal(p2_log_mean, p2_log_std, iterations)

            p2 = safe_exp(p2_log)

            # P3 normal distributed
            p3 = np.random.normal(P3_value, P3_err, iterations)

            # calculate for each iteration the production and the integral. Estimate for every time step the 95% confidence interval interval
            f = np.zeros((iterations, t_max))
            F = np.zeros((iterations, t_max))

            for i in range(iterations):
                f[i, :], F[i, :] = hubbert_deriv(t, p1[i], p2[i], p3[i]), hubbert_model(t, p1[i], p2[i], p3[i])
            

            # return std for f and F
            f_err = np.nanstd(f, axis=0)  # std ignoring NaNs
            F_err = np.nanstd(F, axis=0)  # std ignoring NaNs

            f_mean = np.nanmean(f, axis=0)  # mean ignoring NaNs
            F_mean = np.nanmean(F, axis=0)  # mean ignoring NaNs

             # assert that the means are not nan
            assert not np.isnan(f_mean).any(), 'f_mean contains NaNs'
            assert not np.isnan(F_mean).any(), 'F_mean contains NaNs'

            # Compute the percentiles while ignoring NaNs
            f_lower_ci = np.nanpercentile(f, lower_percentile, axis=0)  # Lower CI for f
            f_upper_ci = np.nanpercentile(f, upper_percentile, axis=0)  # Upper CI for f

            F_lower_ci = np.nanpercentile(F, lower_percentile, axis=0)  # Lower CI for F
            F_upper_ci = np.nanpercentile(F, upper_percentile, axis=0)  # Upper CI for F


            return t, f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci

          
     elif model == 'femp':
            # P1 lognormal distributed
            p1_log_mean = np.log(P1_value / np.sqrt(1 + (P1_err / P1_value) ** 2))
            p1_log_std = np.sqrt(np.log(1 + (P1_err / P1_value) ** 2))
            p1_log = np.random.normal(p1_log_mean, p1_log_std, iterations)
            p1 = safe_exp(p1_log)          

            # Truncation bounds in the original scale
            a, b = 0, 1

            # Scale the bounds to the standard normal scale
            a_scaled = (a - P2_value) / P2_err
            b_scaled = (b - P2_value) / P2_err

            # Generate samples from the truncated normal distribution
            p2 = truncnorm.rvs(a_scaled, b_scaled, loc=P2_value, scale=P2_err, size=iterations)
    

            # calculate for each iteration the production and the integral. Estimate for every time step the 95% confidence interval interval
            f = np.zeros((iterations, t_max))
            F = np.zeros((iterations, t_max))

            for i in range(iterations):
                f[i, :], F[i, :] = femp_deriv(t, p1[i], p2[i]), femp(t, p1[i], p2[i])
            
            # return std for f and F
            f_err = f.std(axis=0)
            F_err = F.std(axis=0)

            f_mean = f.mean(axis=0)
            F_mean = F.mean(axis=0)
            
            # assert that the means are not nan
            assert not np.isnan(f_mean).any(), 'f_mean contains NaNs'
            assert not np.isnan(F_mean).any(), 'F_mean contains NaNs'

            f_lower_ci = np.percentile(f, lower_percentile, axis=0)
            f_upper_ci = np.percentile(f, upper_percentile, axis=0)

            F_lower_ci = np.percentile(F, lower_percentile, axis=0)
            F_upper_ci = np.percentile(F, upper_percentile, axis=0)

            return np.arange(t_max), f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci

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
            t, f_mean, F_mean, f_err, F_err, f_lower_ci, f_upper_ci, F_lower_ci, F_upper_ci  = initiate_mc_per_mine(
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
                'Start_up_year': np.array([group['Start_up_year'].values[0]] * len(t))
            }))

    # Create the resulting DataFrame
    res = pd.concat(ua, ignore_index=True)

    df_to_csv_int(res, 'cumprod_mc_confidence')


    pass






if __name__ == '__main__':
    main_calc_error()
    
    pass

