import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from D_load_werner import merge_werner
from scipy import stats
from tqdm import tqdm
from D_sp_data_clean import get_data, var_exp_path
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

def safe_exp(x):
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf  # Or a reasonable max value
    

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

def hubbert_L_restrict(t, k, t0, *args):
    '''
    Hubbert model for production with L restricted to be less than the maximum production
    
    Parameters:
        t (np.array): Time
        k (float): Growth rate
        t0 (float): Time of peak production
        
    Returns:
        np.array: Production at time
    
    '''
    L = args[0]
    return L / (1 + np.exp(-k * (t - t0)))


def power_law(t, a, b):
    '''
    Power law model for production
    
    Parameters:
        t (np.array): Time
        a (float): Initial production
        b (float): rate of growth over time
        
    Returns:
        np.array: Production at time
    
    '''
    return a * t**b


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


# def fit_models():
#     '''
#     Fit models to production data with train-test split

#     Returns:
#         pd.DataFrame: DataFrame of model fits
#     '''

#     models = {'hubbert': hubbert_deriv, 'femp': femp_deriv}
#     prod_data = prep_data()

#     # Initialize lists to store results
#     mine_names, target_var_names, model_names = [], [], []
#     r2s_train, rmses_train, sample_sizes_train, r2s_test, rmses_test, sample_sizes_test = [], [], [], [], [], []
#     p1, p2, p3, p1_err, p2_err, p3_err = [], [], [], [], [], []
#     p1_pval, p2_pval, p3_pval = [], [], []
#     periods_train, periods_test = [], []
    
#     data_records = []

#     p_group = prod_data.groupby('Prop_id')
#     for mine, prod_data in tqdm(p_group, desc='Fitting models to production data'):
#         for t in targets:
#             sample = prod_data.dropna(subset=t)

#             sample_size = len(sample)
#             if sample_size < min_sample_size:
#                 continue

#             # Train-test split
#             train, test = train_test_split(sample, test_size=0.2, random_state=42)
#             sample_size_train = len(train)
#             sample_size_test = len(test)

#             for model_name, model in models.items():
#                 year_train = train['Year_diff']
#                 year_test = test['Year_diff']

#                 try:
#                     # Fit model on training data
#                     popt, pcov = curve_fit(
#                         model,
#                         year_train.astype(int),
#                         train[t],
#                         p0=init_guesses[model_name],
#                         maxfev=10000,
#                         bounds=(lower_bounds[model_name], upper_bounds[model_name])
#                     )
#                     perr = np.sqrt(np.diag(pcov))

#                     # Predictions on train and test data
#                     train_pred = model(year_train.astype(int), *popt)
#                     test_pred = model(year_test.astype(int), *popt)

#                     # p-values for parameters
#                     t_stats = popt / perr
#                     p_values = [2 * (1 - stats.t.cdf(np.abs(t), sample_size - len(popt))) for t in t_stats]

#                     # Metrics
#                     r2_test = np.corrcoef(test[t], test_pred)[0, 1] ** 2
#                     rmse_test = np.sqrt(np.mean((test[t] - test_pred) ** 2))
#                     r2_train = np.corrcoef(train[t], train_pred)[0, 1] ** 2
#                     rmse_train = np.sqrt(np.mean((train[t] - train_pred) ** 2))

#                     # Store training and test data for this fold
#                     data_records.append(pd.DataFrame({
#                         'Mine_ID': mine,
#                         'Year': train['Year'],
#                         'Target_var': t,
#                         'Observed': train[t],
#                         'Predicted': train_pred,
#                         'Set': 'Train'
#                     }))

#                     data_records.append(pd.DataFrame({
#                         'Mine_ID': mine,
#                         'Year': test['Year'],
#                         'Target_var': t,
#                         'Observed': test[t],
#                         'Predicted': test_pred,
#                         'Set': 'Test'
#                     }))

#                     # Append results
#                     p1.append(popt[0])
#                     p2.append(popt[1])
#                     p1_err.append(perr[0])
#                     p2_err.append(perr[1])
#                     p1_pval.append(p_values[0])
#                     p2_pval.append(p_values[1])

#                     if model_name in ['hubbert', 'hubbert_L_restrict']:
#                         p3.append(popt[2])
#                         p3_err.append(perr[2])
#                         p3_pval.append(p_values[2])
#                     else:
#                         p3.append(np.nan)
#                         p3_err.append(np.nan)
#                         p3_pval.append(np.nan)

#                     mine_names.append(mine)
#                     target_var_names.append(t)
#                     model_names.append(model_name)
#                     r2s_train.append(r2_train)
#                     rmses_train.append(rmse_train)
#                     sample_sizes_train.append(sample_size_train)

#                     r2s_test.append(r2_test)
#                     rmses_test.append(rmse_test)
#                     sample_sizes_test.append(sample_size_test)
#                     periods_train.append(train['Year'].tolist())
#                     periods_test.append(test['Year'].tolist())

#                 except RuntimeError:
                    
#                     data_records.append(pd.DataFrame({
#                         'Mine_ID': mine,
#                         'Year': train['Year'],
#                         'Target_var': t,
#                         'Observed': train[t],
#                         'Predicted': np.nan,
#                         'Set': 'Train'
#                     }))
#                     data_records.append(pd.DataFrame({
#                         'Mine_ID': mine,
#                         'Year': test['Year'],
#                         'Target_var': t,
#                         'Observed': test[t],
#                         'Predicted': np.nan,
#                         'Set': 'Test'
#                     }))

#                     mine_names.append(mine)
#                     target_var_names.append(t)
#                     model_names.append(model_name)
#                     r2s_train.append(np.nan)
#                     rmses_train.append(np.nan)
#                     sample_sizes_train.append(np.nan)
#                     r2s_test.append(np.nan)
#                     rmses_test.append(np.nan)
#                     sample_sizes_test.append(np.nan)
#                     periods_train.append(np.nan)
#                     periods_test.append(np.nan)
#                     p1.append(np.nan)
#                     p2.append(np.nan)
#                     p3.append(np.nan)
#                     p1_err.append(np.nan)
#                     p2_err.append(np.nan)
#                     p3_err.append(np.nan)
#                     p1_pval.append(np.nan)
#                     p2_pval.append(np.nan)
#                     p3_pval.append(np.nan)
#         break

#     # Create results DataFrame
#     res_df = pd.DataFrame({
#         'Mine_ID': mine_names,
#         'Target_var': target_var_names,
#         'Model': model_names,
#         'R2_train': r2s_train,
#         'RMSE_train': rmses_train,
#         'Sample_size_train': sample_sizes_train,
#         'R2_test': r2s_test,
#         'RMSE_test': rmses_test,
#         'Sample_size_test': sample_sizes_test,
#         'Period_train': periods_train,
#         'Period_test': periods_test,
#         'P1_value': p1, 'P2_value': p2, 'P3_value': p3,
#         'P1_err': p1_err, 'P2_err': p2_err, 'P3_err': p3_err,
#         'P1_pval': p1_pval, 'P2_pval': p2_pval, 'P3_pval': p3_pval
#     })




#     data_records = pd.concat(data_records)
    

#     data_records.to_csv('data/int/data_records.csv', index=False)

#     # Save results to JSON
#     res_df.to_json('data/int/production_model_fits.json', orient='records')
    

#     return res_df

def fit_mine_models(mine, prod_data, models, targets, init_guesses, lower_bounds, upper_bounds, min_sample_size):
    """
    Fit models for a single mine.
    """
    results = []
    data_records = []

    for t in targets:
        sample = prod_data.dropna(subset=t)

        sample_size = len(sample)
        if sample_size < min_sample_size:
            continue

        # Train-test split
        train, test = train_test_split(sample, test_size=0.2, random_state=42)
        sample_size_train = len(train)
        sample_size_test = len(test)

        for model_name, model in models.items():
            year_train = train['Year_diff']
            year_test = test['Year_diff']

            try:
                # Fit model on training data
                popt, pcov = curve_fit(
                    model,
                    year_train.astype(int),
                    train[t],
                    p0=init_guesses[model_name],
                    maxfev=10000,
                    bounds=(lower_bounds[model_name], upper_bounds[model_name])
                )
                perr = np.sqrt(np.diag(pcov))

                # Predictions on train and test data
                train_pred = model(year_train.astype(int), *popt)
                test_pred = model(year_test.astype(int), *popt)

                # p-values for parameters
                t_stats = popt / perr
                p_values = [2 * (1 - stats.t.cdf(np.abs(t), sample_size - len(popt))) for t in t_stats]

                # Metrics
                r2_test = np.corrcoef(test[t], test_pred)[0, 1] ** 2
                rmse_test = np.sqrt(np.mean((test[t] - test_pred) ** 2))
                r2_train = np.corrcoef(train[t], train_pred)[0, 1] ** 2
                rmse_train = np.sqrt(np.mean((train[t] - train_pred) ** 2))

                # Store results
                results.append({
                    'Mine_ID': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'R2_train': r2_train,
                    'RMSE_train': rmse_train,
                    'Sample_size_train': sample_size_train,
                    'R2_test': r2_test,
                    'RMSE_test': rmse_test,
                    'Sample_size_test': sample_size_test,
                    'Period_train': train['Year'].tolist(),
                    'Period_test': test['Year'].tolist(),
                    'P1_value': popt[0],
                    'P2_value': popt[1],
                    'P3_value': popt[2] if model_name in ['hubbert', 'hubbert_L_restrict'] else np.nan,
                    'P1_err': perr[0],
                    'P2_err': perr[1],
                    'P3_err': perr[2] if model_name in ['hubbert', 'hubbert_L_restrict'] else np.nan,
                    'P1_pval': p_values[0],
                    'P2_pval': p_values[1],
                    'P3_pval': p_values[2] if model_name in ['hubbert', 'hubbert_L_restrict'] else np.nan
                })

                # Collect training and test data
                data_records.append(pd.DataFrame({
                    'Prop_id': mine,
                    'Year': train['Year'],
                    'Target_var': t,
                    'Observed': train[t],
                    'Predicted': train_pred,
                    'Model': model_name,
                    'Set': 'Train'
                }))
                data_records.append(pd.DataFrame({
                    'Prop_id': mine,
                    'Year': test['Year'],
                    'Target_var': t,
                    'Observed': test[t],
                    'Predicted': test_pred,
                    'Model': model_name,
                    'Set': 'Test'
                }))

            except RuntimeError:
                # Handle model fitting failures
                results.append({
                    'Prop_id': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'R2_train': np.nan,
                    'RMSE_train': np.nan,
                    'Sample_size_train': np.nan,
                    'R2_test': np.nan,
                    'RMSE_test': np.nan,
                    'Sample_size_test': np.nan,
                    'Period_train': np.nan,
                    'Period_test': np.nan,
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


def fit_models():
    """
    Fit models to production data with train-test split

    Returns:
        pd.DataFrame: DataFrame of model fits
    """
    models = {'hubbert': hubbert_deriv, 'femp': femp_deriv}
    prod_data = prep_data()

    p_group = prod_data.groupby('Prop_id')

    # Parallelize over mines
    results_data = Parallel(n_jobs=-1)(delayed(fit_mine_models)(
        mine, group, models, targets, init_guesses, lower_bounds, upper_bounds, min_sample_size
    ) for mine, group in tqdm(p_group, desc='Fitting models to production data'))

    # Unpack parallelized results
    all_results, all_data_records = zip(*results_data)

    # Combine results if not empty
    res_df = pd.concat([pd.DataFrame(res) for res in all_results if res])
    data_records = pd.concat([pd.concat(data) for data in all_data_records if data])
    # Save results
    res_df.to_json('data/int/production_model_fits.json', orient='records')
    data_records.to_csv('data/int/data_records.csv', index=False)
    return res_df


def prep_data(): 
    prod_data = pd.read_csv(r'data\int\D_build_sample_sets\target_vars_prio_source.csv')
    
    prod_data['Year'] = pd.to_datetime(prod_data['Year'], errors='coerce')
    prod_data['Start_up_year'] = pd.to_datetime(prod_data['Start_up_year'], errors='coerce')

    prod_data = prod_data.dropna(subset=['Year', 'Start_up_year'])

    prod_data['Year_diff'] = (prod_data['Year'].dt.year - prod_data['Start_up_year'].dt.year).astype(int)
    return prod_data

targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']

init_guesses = {'hubbert': (10**8, 0.1, 20), 'power_law': (0, 0), 'femp': (10**8, 0.1)}
lower_bounds = {'hubbert': (0, 0, 0), 'power_law': (0, 0), 'femp': (0, 0)}
upper_bounds = {'hubbert': (np.inf, np.inf, 100), 'power_law': (np.inf, np.inf), 'femp': (np.inf, 1)}

min_sample_size = 10

if __name__ == '__main__':
    fit_models()