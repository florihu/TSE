import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy import stats
from tqdm import tqdm

from util import df_to_latex, df_to_csv_int
from M_prod_model import hubbert_model, femp, hubbert_deriv, femp_deriv

###########################################Description############################################

# Calculate the error of production and integrals for femp and hubbert models


###########################################Parameter############################################

rename_dict ={'hubbert': {'P1_value': 'L', 'P2_value': 'k', 'P3_value': 't0', 'P1_err': 'L_err', 'P2_err': 'k_err', 'P3_err': 't0_err'},
                        'femp':{'P1_value': 'R0', 'P2_value': 'C', 'P1_err': 'R0_err', 'P2_err': 'C_err'}}

relevant_params = {'hubbert': ['L', 'k', 't0'], 'femp': ['R0', 'C']}


hubbert_corr_val = {
    ('hubbert', 'L'): {'L': 1.0, 'k': -0.17, 't0': 0.19},
    ('hubbert', 'k'): {'L': -0.17, 'k': 1.0, 't0': -0.30},
    ('hubbert', 't0'): {'L': 0.19, 'k': -0.30, 't0': 1.0},
}
femp_corr_val = {
    ('femp', 'R0'): {'R0': 1.0, 'C': -0.17},
    ('femp', 'C'): {'R0': -0.17, 'C': 1.0},
}
# Convert to DataFrame
corr_mat = {'hubbert': pd.DataFrame(hubbert_corr_val).T.fillna(0), 'femp': pd.DataFrame(femp_corr_val).T.fillna(0)}


###########################################Functions############################################
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


# FEMP function partial derivatives integration
def femp_partial_R0(t, C):
    return -np.log(1 - C) * (1 - C)**t

def femp_partial_C(t, R0, C):
    term1 = (1 - C)**(t-1)
    term2 = t * (1 - C)**t * np.log(1 - C)
    return -R0 * (term1 + term2)


def integrate_hubbert_partial_R(t_series, k, t0):
    return [quad(hubbert_partial_R, t_series[0], t_max, args=(k, t0))[0] for t_max in t_series]

def integrate_hubbert_partial_k(t_series, R, k, t0):
    return [quad(hubbert_partial_k, t_series[0], t_max, args=(R, k, t0))[0] for t_max in t_series]

def integrate_hubbert_partial_t0(t_series, R, k, t0):
    return [quad(hubbert_partial_t0, t_series[0], t_max, args=(R, k, t0))[0] for t_max in t_series]

def integrate_femp_partial_R0(t_series, C):
    return [quad(femp_partial_R0, t_series[0], t_max, args=(C,))[0] for t_max in t_series]

def integrate_femp_partial_C(t_series, R0, C):
    return [quad(femp_partial_C, t_series[0], t_max, args=(R0, C))[0] for t_max in t_series]

def integrate_femp_deriv(t_series, R0, C):
    return [quad(femp_deriv, t_series[0], t_max, args=(R0, C))[0] for t_max in t_series]

def integrate_hubbert_deriv(t_series, L, k, t0):
    return [quad(hubbert_deriv, t_series[0], t_max, args=(L, k, t0))[0] for t_max in t_series]



# Error propagation using Gaussian formula for a time series
def gaussian_error_propagation_ts(partial_derivatives_ts, variances, covariances):
    error_squared_ts = np.sum((partial_derivatives_ts**2) * variances, axis=1)
    for i in range(partial_derivatives_ts.shape[1] - 1):
        for j in range(i + 1, partial_derivatives_ts.shape[1]):
            error_squared_ts += (
                2 * partial_derivatives_ts[:, i] * partial_derivatives_ts[:, j] * covariances[i, j]
            )
    return np.sqrt(error_squared_ts)

def initiate_gauss_per_mine(t_max, model, **kwargs):

    t = np.arange(0, t_max )

    if model == 'hubbert':
        L = kwargs['P1_value']
        k = kwargs['P2_value']
        t0 = kwargs['P3_value']

        partial_derivatives = np.array([hubbert_partial_R(t, k, t0), 
                               hubbert_partial_k(t, L, k, t0), 
                               hubbert_partial_t0(t, L, k, t0)]).T
        
        partial_derivatives_integrated = np.array([integrate_hubbert_partial_R(t, k, t0), 
                                          integrate_hubbert_partial_k(t, L, k, t0), 
                                          integrate_hubbert_partial_t0(t, L, k, t0)]).T
        
        L_var = kwargs['P1_err']
        k_var = kwargs['P2_err']
        t0_var = kwargs['P3_err']

        variances = [L_var, k_var, t0_var]
        corr = corr_mat[model]
        covariances = np.array([[variances[i] * variances[j] * corr.iloc[i, j] for j in range(len(variances))] for i in range(len(variances))])

        f = hubbert_deriv(t, L, k, t0)
        F_a = hubbert_model(t, L, k, t0)
        F_n = integrate_hubbert_deriv(t, L, k, t0)

        f_err = gaussian_error_propagation_ts(partial_derivatives, variances, covariances)
        F_err = gaussian_error_propagation_ts(partial_derivatives_integrated, variances, covariances)

        
    
    elif model == 'femp':
        r0 = kwargs['P1_value']
        c = kwargs['P2_value']

        partial_derivatives = np.array([femp_partial_R0(t, c),
                                 femp_partial_C(t, r0, c)]).T
        
        partial_derivatives_integrated = np.array([integrate_femp_partial_R0(t, c),
                                            integrate_femp_partial_C(t, r0, c)]).T
        
        r0_var = kwargs['P1_err']
        c_var = kwargs['P2_err']

        variances = [r0_var, c_var]
        corr = corr_mat[model]
        covariances = np.array([[variances[i] * variances[j] * corr.iloc[i, j] for j in range(len(variances))] for i in range(len(variances))])
        

        f = femp_deriv(t, r0, c)
        F_a = femp(t, r0, c)
        F_n = integrate_femp_deriv(t, r0, c)


        f_err = gaussian_error_propagation_ts(partial_derivatives, variances, covariances)
        F_err = gaussian_error_propagation_ts(partial_derivatives_integrated, variances, covariances)

    return t, f, F_a, F_n, f_err, F_err




###########################################Main Tasks############################################
def main_calc_corr_tab():
    """
    Computes parameter correlations for every model standard deviation.

    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame containing the results for Hubbert model parameters,
        including columns ['Target_var', 'L', 'k', 't0', 'L_err', 'k_err', 't0_err'].

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the correlations, p-values, and parameter combinations for each target variable.
    """
    # Rename columns if necessary
    

    # List to store results

    results = pd.read_json(r'data\int\production_model_fits.json')


    results_list = []

    for m in ['hubbert', 'femp']:

        inter = results.rename(columns=rename_dict[m])
        # Loop over each target variable
        for t in ['Tailings_production', 'Ore_processed_mass', 'Waste_rock_production', 'Concentrate_production']:
            # Subset the DataFrame for the current target variable
            sub = inter[inter['Target_var'] == t][relevant_params[m]]

            # Compute the correlation matrix
            corr_matrix = sub.corr(method='pearson')
            p_matrix = sub.corr(method=lambda x, y: stats.pearsonr(x, y)[1])  # Compute p-values

            # Extract upper triangle of the correlation matrix (excluding diagonal)
            param_pairs = np.triu_indices_from(corr_matrix, k=1)
            for i, j in zip(*param_pairs):
                var1 = corr_matrix.index[i]
                var2 = corr_matrix.columns[j]

                results_list.append({
                    'Model': m,
                    'Target_var': t,
                    'Var1': var1,
                    'Var2': var2,
                    'Correlation': corr_matrix.iloc[i, j],
                    'P_value': p_matrix.iloc[i, j]
                })

    # Convert results list to DataFrame
    corr_df = pd.DataFrame(results_list)


    df_to_latex(corr_df, 'corr_table')

    return corr_df


def main_calc_error():

    res = pd.read_json(r'data\int\production_model_fits.json')

    prio =pd.read_csv(r'data\int\D_target_prio_prep\target_vars_prio_source.csv')

    prio = prio[['Prop_id', 'Start_up_year']].drop_duplicates()

    res = res.merge(prio, on='Prop_id', how='left')

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
            t, f, F_a, F_n, f_err, F_err = initiate_gauss_per_mine(
                t_max, m, **group[['P1_value', 'P2_value', 'P3_value', 'P1_err', 'P2_err', 'P3_err']].iloc[0]
            )

            # Append the result to the errors list
            ua.append(pd.DataFrame({
                'Prop_id': np.array([id]*len(t)),
                'Target_var': np.array([t_]*len(t)),
                'Model': np.array([m] * len(t)),
                'Time_period': t,
                'f': f,
                'F_a': F_a,
                'F_n': F_n,
                'f_err': f_err,
                'F_err': F_err,
                'Start_up_year': np.array([group['Start_up_year'].values[0]] * len(t))
            }))

    # Create the resulting DataFrame
    res = pd.concat(ua, ignore_index=True)

    df_to_csv_int(res, 'cumprod_ua')


    pass


if __name__ == '__main__':
    main_calc_error()



    

