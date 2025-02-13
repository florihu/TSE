import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from D_load_werner import merge_werner
from scipy import stats
from tqdm import tqdm
from D_sp_data_clean import get_data, var_exp_path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


############################################################Parameters############################################################
targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']

lower_bounds = {
    'hubbert': (0, 0, 0),  # log-transformed L and k
    'femp': (0, 0)  # log-transformed R0, logit-transformed C
}

upper_bounds = {
    'hubbert': (np.inf, np.inf, np.inf),
    'femp': (np.inf, 1)  # No upper bound needed for log-transformed variables
}
min_sample_size = 10

np.random.seed(42)

############################################################Functions############################################################
def safe_exp(x):
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf  # Or a reasonable max value
    
def hubbert_transformed(t, log_L, log_k, t0):
    """
    Hubbert model with log-transformed parameters.
    
    Parameters:
        t (np.array): Time
        log_L (float): Log of maximum production
        log_k (float): Log of growth rate
        t0 (float): Time of peak production
    
    Returns:
        np.array: Production at time
    """
    L = safe_exp(log_L)  # Ensure L is strictly positive
    k = safe_exp(log_k)  # Ensure k is strictly positive
    return L / (1 + np.exp(-k * (t - t0)))

def femp_transformed(t, log_R0, C_transformed):
     
     """
    FEMP model with log and logit-transformed parameters.
    
    Parameters:
        t (np.array): Time
        log_R0 (float): Log of initial reserves
        C_transformed (float): Logit of production-to-reserve ratio
    
    Returns:
        np.array: Production at time
    """
     R0 = safe_exp(log_R0)  # Ensure R0 is strictly positiv
     C = 1 / (1 + safe_exp(-C_transformed))  # Ensure C stays between 0 and 1
     return R0 * (1 - (1 - C)**t)

def hubbert_deriv_transformed(t, log_L, log_k, t0):
    """
    Hubbert model derivative with log-transformed parameters.
    """
    L = safe_exp(log_L)  # Ensure L is strictly positive
    k = safe_exp(log_k)  # Ensure k is strictly positive
    return L * k * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))**2

def femp_deriv_transformed(t, log_R0, C_transformed):
    """
    FEMP model derivative with log and logit-transformed parameters.
    """
    R0 = safe_exp(log_R0)  # Ensure R0 is strictly positive
    C = 1 / (1 + safe_exp(-C_transformed))  # Ensure C stays between 0 and 1
    return -R0 * np.log(1 - C) * (1 - C)**t

def hubbert_model(t, L, k, t0):
    '''
    Hubbert model for production
    
    Parameters:
        t (np.array): Time
        L (float): Maximum production
        k (float): Growth rate
        t0 (float): Time of peak production
        
    Returns:
        np.array: Production at time
    
    '''
    return L / (1 + np.exp(-k * (t - t0)))

def femp(t, R0, C):
    '''
    Exponential model for production
    
    Parameters:
        t (np.array): Time
        R0 (float): Initial reserves
        C (float): Production to reserve ratio
        
    Returns:
        np.array: Production at time
    
    '''
    return R0 * (1-(1-C)**t)

def femp_deriv(t, R0, C):
    '''
    Derivative of the exponential model for production
    
    Parameters:
        t (np.array): Time
        R0 (float): Initial reserves
        C (float): Production to reserve ratio
        
    Returns:
        np.array: Production at time
    
    '''
    return - R0 * np.log(1-C) * (1-C)**t

def hubbert_deriv(t, L, k, t0):
    '''
    Derivative of the Hubbert model for production
    
    Parameters:
        t (np.array): Time
        L (float): Maximum production
        k (float): Growth rate
        t0 (float): Time of peak production
        
    Returns:
        np.array: Production at time
    
    '''
    return L * k * safe_exp(-k * (t - t0)) / (1 + safe_exp(-k * (t - t0)))**2


def prep_init_guesses(model_name, sample, t):
    if model_name == 'hubbert':
        L_guess = sample[t].cumsum().max()  
        k_guess = sample[t].mean() / L_guess  
        t0_guess = sample.loc[sample[t].idxmax(), 'Year_diff'].astype(int) 
        return (L_guess, k_guess, t0_guess)

    elif model_name == 'femp':
        R0_guess = sample[t].cumsum().max()
        C_guess = sample[t].mean() / R0_guess
        return (R0_guess, C_guess)


def fit_prod_model(mine, prod_data, models, targets, lower_bounds, upper_bounds, min_sample_size):
    results = []
    data_records = []

    for t in targets:
        sample = prod_data.dropna(subset=t)

        # Outlier filtering
        sample = prod_data.loc[prod_data[f'{t}_outlier'] == False]


        sample_size = len(sample)
        if sample_size < min_sample_size:
            continue

        for model_name, model in models.items():
            years = sample['Year_diff']
            init_guess = prep_init_guesses(model_name, sample, t)

            try:
                # Fit model with transformed parameters
                popt, pcov = curve_fit(
                    model,
                    years.astype(int),
                    sample[t],
                    p0=init_guess,
                    maxfev=10000,
                    bounds=(lower_bounds[model_name], upper_bounds[model_name])
                )

                perr = np.sqrt(np.diag(pcov))  # Standard errors

                # Transform parameters back to original scale
                if model_name == 'hubbert':
                    L_est, k_est, t0_est = popt[0], popt[1], popt[2]
                    L_err, k_err, t0_err = perr[0] , perr[1],  perr[2]  # Assuming t0 is not log-transformed

                else:  # FEMP model
                    R0_est, C_est = popt[0], popt[1]
                    R0_err, C_err = perr[0], perr[1]

                # Predictions on train and test data
                pred = model(years.astype(int), *popt)  # No need   to exponentiate again

                # p-values
                t_stats = popt / perr
                p_values = [2 * (1 - stats.t.cdf(np.abs(t), sample_size - len(popt))) for t in t_stats]
                
                r2 = np.corrcoef(sample[t], pred)[0, 1] ** 2
                rmse = np.sqrt(np.mean((sample[t] - pred) ** 2))
                nrmse = rmse / sample[t].max()

                # Store results
                results.append({
                    'Prop_id': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'Start_up_year': sample['Start_up_year'].iloc[0],
                    'R2': r2,
                    'RMSE': rmse,
                    'NRMSE': nrmse,
                    'P1_value': L_est if model_name == 'hubbert' else R0_est,
                    'P2_value': k_est if model_name == 'hubbert' else C_est,
                    'P3_value': t0_est if model_name == 'hubbert' else np.nan,
                    'P1_err': L_err if model_name == 'hubbert' else R0_err,
                    'P2_err': k_err if model_name == 'hubbert' else C_err,
                    'P3_err': t0_err if model_name == 'hubbert' else np.nan,
                    'P1_pval': p_values[0],
                    'P2_pval': p_values[1],
                    'P3_pval': p_values[2] if model_name == 'hubbert' else np.nan
                })

                # Collect training and test data
                data_records.append(pd.DataFrame({
                    'Prop_id': mine,
                    'Year': years,
                    'Target_var': t,
                    'Start_up_year': sample['Start_up_year'].iloc[0],
                    'Observed': sample[t],
                    'Residual': sample[t] - pred,
                    'Predicted': pred,
                    'Model': model_name,
                }))

            except RuntimeError:
                # Handle fitting failures
                results.append({
                    'Prop_id': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'R2': np.nan,
                    'RMSE': np.nan,
                    'NRMSE': np.nan,
                    'P1_value': np.nan,
                    'P2_value': np.nan,
                    'P3_value': np.nan,
                    'P1_err': np.nan,
                    'P2_err': np.nan,
                    'P3_err': np.nan,
                    'P1_pval': np.nan,
                    'P2_pval': np.nan,
                    'P3_pval': np.nan
                })

    return results, data_records


def prep_data(): 
    prod_data = pd.read_csv(r'data\int\E_targets_explo\target_sample_with_outlier_detection_trans.csv')

    prod_data['Year_diff'] = prod_data['Year'] - prod_data['Start_up_year']

    assert prod_data['Year_diff'].min() >= 0, 'Negative year difference'

    return prod_data



############################################################Main############################################################
def main_fitting_loop():
    """
    Fit models to production data with train-test split

    Returns:
        pd.DataFrame: DataFrame of model fits
    """
    models = {'hubbert': hubbert_deriv, 'femp': femp_deriv}
    prod_data = prep_data()

    p_group = prod_data.groupby('Prop_id')

    # Parallelize over mines
    results_data = Parallel(n_jobs=-1)(delayed(fit_prod_model)(
        mine, group, models, targets, lower_bounds, upper_bounds, min_sample_size
    ) for mine, group in tqdm(p_group, desc='Fitting models to production data'))

    # Unpack parallelized results
    all_results, all_data_records = zip(*results_data)

    # Combine results if not empty
    res_df = pd.concat([pd.DataFrame(res) for res in all_results if res])
    data_records = pd.concat([pd.concat(data) for data in all_data_records if data])
    # Save results
    res_df.to_json('data/int/production_model_fits_trans.json', orient='records')
    data_records.to_csv('data/int/data_records_trans.csv', index=False)
    return res_df


if __name__ == '__main__':
    main_fitting_loop()