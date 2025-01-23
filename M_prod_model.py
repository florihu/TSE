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

def polynomial_3degree(t, a, b, c):
    return a + b*t + c*t**2

def polynomial_4degree(t, a, b, c, d):
    return a + b*t + c*t**2 + d*t**3

def polynomial_5degree(t, a, b, c, d, e):
    return a + b*t + c*t**2 + d*t**3 + e*t**4





#def fit_model():
    '''
    Fit models to production data with train-test split

    Returns:
        pd.DataFrame: DataFrame of model fits
    '''

    models = {'hubbert': hubbert_deriv, 'femp': femp_deriv}
    prod_data = prep_data()

    # Initialize lists to store results
    mine_names, target_var_names, model_names = [], [], []
    r2s_train, rmses_train, sample_sizes_train, r2s_test, rmses_test, sample_sizes_test = [], [], [], [], [], []
    p1, p2, p3, p1_err, p2_err, p3_err = [], [], [], [], [], []
    p1_pval, p2_pval, p3_pval = [], [], []
    periods_train, periods_test = [], []
    
    data_records = []

    p_group = prod_data.groupby('Prop_id')
    for mine, prod_data in tqdm(p_group, desc='Fitting models to production data'):
        for t in targets:
            sample = prod_data.dropna(subset=t)
            sample = prod_data[prod_data[f'{t}_outlier'] == False]

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

                    # Store training and test data for this fold
                    data_records.append(pd.DataFrame({
                        'Mine_ID': mine,
                        'Year': train['Year'],
                        'Target_var': t,
                        'Observed': train[t],
                        'Predicted': train_pred,
                        'Set': 'Train'
                    }))

                    data_records.append(pd.DataFrame({
                        'Mine_ID': mine,
                        'Year': test['Year'],
                        'Target_var': t,
                        'Observed': test[t],
                        'Predicted': test_pred,
                        'Set': 'Test'
                    }))

                    # Append results
                    p1.append(popt[0])
                    p2.append(popt[1])
                    p1_err.append(perr[0])
                    p2_err.append(perr[1])
                    p1_pval.append(p_values[0])
                    p2_pval.append(p_values[1])

                    if model_name in ['hubbert', 'hubbert_L_restrict']:
                        p3.append(popt[2])
                        p3_err.append(perr[2])
                        p3_pval.append(p_values[2])
                    else:
                        p3.append(np.nan)
                        p3_err.append(np.nan)
                        p3_pval.append(np.nan)

                    mine_names.append(mine)
                    target_var_names.append(t)
                    model_names.append(model_name)
                    r2s_train.append(r2_train)
                    rmses_train.append(rmse_train)
                    sample_sizes_train.append(sample_size_train)

                    r2s_test.append(r2_test)
                    rmses_test.append(rmse_test)
                    sample_sizes_test.append(sample_size_test)
                    periods_train.append(train['Year'].tolist())
                    periods_test.append(test['Year'].tolist())

                except RuntimeError:
                    
                    data_records.append(pd.DataFrame({
                        'Mine_ID': mine,
                        'Year': train['Year'],
                        'Target_var': t,
                        'Observed': train[t],
                        'Predicted': np.nan,
                        'Set': 'Train'
                    }))
                    data_records.append(pd.DataFrame({
                        'Mine_ID': mine,
                        'Year': test['Year'],
                        'Target_var': t,
                        'Observed': test[t],
                        'Predicted': np.nan,
                        'Set': 'Test'
                    }))

                    mine_names.append(mine)
                    target_var_names.append(t)
                    model_names.append(model_name)
                    r2s_train.append(np.nan)
                    rmses_train.append(np.nan)
                    sample_sizes_train.append(np.nan)
                    r2s_test.append(np.nan)
                    rmses_test.append(np.nan)
                    sample_sizes_test.append(np.nan)
                    periods_train.append(np.nan)
                    periods_test.append(np.nan)
                    p1.append(np.nan)
                    p2.append(np.nan)
                    p3.append(np.nan)
                    p1_err.append(np.nan)
                    p2_err.append(np.nan)
                    p3_err.append(np.nan)
                    p1_pval.append(np.nan)
                    p2_pval.append(np.nan)
                    p3_pval.append(np.nan)
        break

    # Create results DataFrame
    res_df = pd.DataFrame({
        'Mine_ID': mine_names,
        'Target_var': target_var_names,
        'Model': model_names,
        'R2_train': r2s_train,
        'RMSE_train': rmses_train,
        'Sample_size_train': sample_sizes_train,
        'R2_test': r2s_test,
        'RMSE_test': rmses_test,
        'Sample_size_test': sample_sizes_test,
        'Period_train': periods_train,
        'Period_test': periods_test,
        'P1_value': p1, 'P2_value': p2, 'P3_value': p3,
        'P1_err': p1_err, 'P2_err': p2_err, 'P3_err': p3_err,
        'P1_pval': p1_pval, 'P2_pval': p2_pval, 'P3_pval': p3_pval
    })




    data_records = pd.concat(data_records)
    

    data_records.to_csv('data/int/data_records.csv', index=False)

    # Save results to JSON
    res_df.to_json('data/int/production_model_fits.json', orient='records')
    

    return res_df





def prep_init_guesses(model_name, sample, t):
    '''
    
    Using the year with the maximum value for the Hubbert model instead of the mean could potentially improve the prediction in some cases, but it's not guaranteed to be better for all datasets. Here's an analysis of the potential impact:

    Pros of using the year with maximum value:
        It might better capture the peak production year, which is crucial for the Hubbert model.
        It could provide a more accurate initial guess for the t0 parameter, especially if the production data has a clear peak.
    Cons of using the year with maximum value:
        It might be more sensitive to outliers or anomalies in the data.
        If the production data is noisy or has multiple local maxima, it might not represent the true peak of the overall trend.
    In general, using the mean year as the initial guess is a more robust approach, as it's less sensitive to outliers and anomalies. However, if you have reason to believe that the year with the maximum value is a better estimate of the peak production year, you could use that instead.
    
    
    '''

    if model_name == 'hubbert':
        # Initial guess for Hubbert model
        L_guess = sample[t].cumsum().max()
        k_guess = sample[t].mean() / L_guess
        # Assume peak production is at the midpoint of the data
        t0_guess = sample['Year_diff'].mean().astype(int)
        # Assumption peak of data and peak of production are the same - could be sensitive to outlier togh
        # t0_guess = sample.loc[sample[t].idxmax(), 'Year_diff'].astype(int)

        # assert all estimates are within bounds
        assert 0 < L_guess < np.inf, 'Invalid initial guess for Hubbert model'
        assert 0 < k_guess < np.inf, 'Invalid initial guess for Hubbert model'
        assert 0 < t0_guess < np.inf, 'Invalid initial guess for Hubbert model'


        return (L_guess, k_guess, t0_guess)
    
    elif model_name == 'femp':
        # Initial guess for FEMP model
        R0_guess = sample[t].cumsum().max()
        C_guess = sample[t].mean() / R0_guess

        assert 0 < R0_guess < np.inf, 'Invalid initial guess for FEMP model'
        assert 0 < C_guess < np.inf, 'Invalid initial guess for FEMP model'

        return (R0_guess, C_guess)
    else:
        return None
    



def fit_prod_model(mine, prod_data, models, targets, lower_bounds, upper_bounds, min_sample_size):
    """
    Fit models for a single mine.
    """
    results = []
    data_records = []

    for t in targets:
        sample = prod_data.dropna(subset=t)

        # Outlier filtering
        sample = prod_data[prod_data[f'{t}_outlier'] == False]

        sample_size = len(sample)

        if sample_size < min_sample_size:
            continue

        for model_name, model in models.items():
            years = sample['Year_diff']
            init_guess = prep_init_guesses(model_name, sample, t)

            try:
                # Fit model on training data
                popt, pcov = curve_fit(
                    model,
                    years.astype(int),
                    sample[t],
                    p0=init_guess,
                    maxfev=10000,
                    bounds=(lower_bounds[model_name], upper_bounds[model_name])
                )
                perr = np.sqrt(np.diag(pcov))

                # Predictions on train and test data
                pred = model(years.astype(int), *popt)
                

                # p-values for parameters
                t_stats = popt / perr
                p_values = [2 * (1 - stats.t.cdf(np.abs(t), sample_size - len(popt))) for t in t_stats]

                # Metrics
                r2 = np.corrcoef(sample[t], pred)[0, 1] ** 2
                rmse = np.sqrt(np.mean((sample[t] - pred) ** 2))
                cov = rmse / sample[t].max()        

                # Store results
                results.append({
                    'Prop_id': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'R2': r2,
                    'RMSE': rmse,
                    'COV': cov,
                    'Sample_size': len(sample),
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
                    'Year': years,
                    'Target_var': t,
                    'Observed': sample[t],
                    'Residual': sample[t] - pred,
                    'Predicted': pred,
                    'Model': model_name,
                }))

            except RuntimeError:
                # Handle model fitting failures
                results.append({
                    'Prop_id': mine,
                    'Target_var': t,
                    'Model': model_name,
                    'R2': np.nan,
                    'RMSE': np.nan,
                    'COV': np.nan,
                    'Sample_size': np.nan,
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


def fitting_loop():
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
    res_df.to_json('data/int/production_model_fits.json', orient='records')
    data_records.to_csv('data/int/data_records.csv', index=False)
    return res_df


def prep_data(): 
    prod_data = pd.read_csv(r'data\int\E_targets_explo\target_sample_with_outlier_detection.csv')

    prod_data['Year_diff'] = prod_data['Year'] - prod_data['Start_up_year']

    assert prod_data['Year_diff'].min() >= 0, 'Negative year difference'

    return prod_data

targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']


lower_bounds = {'hubbert': (0, 0, 0), 'power_law': (0, 0), 'femp': (0, 0)}
upper_bounds = {'hubbert': (np.inf, np.inf, np.inf), 'power_law': (np.inf, np.inf), 'femp': (np.inf, np.inf)}

min_sample_size = 10

if __name__ == '__main__':
    fitting_loop()