import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from D_load_werner import merge_werner
from scipy import stats
from tqdm import tqdm
from D_sp_data_clean import get_data, var_exp_path

from sklearn.model_selection import train_test_split


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
    return L * k * np.exp(-k * (t - t0)) / (1 + np.exp(-k * (t - t0)))**2


def fit_models():
    '''
    Fit models to production data with train-test split
    
    Returns:
        pd.DataFrame: DataFrame of model fits
    '''
    
    models = {'hubbert': hubbert_deriv,  'femp': femp_deriv}

    # Initialize lists to store results
    mine_names, target_var_names, model_names = [], [], []
    r2s_train, rmses_train, sample_sizes_train, r2s_test, rmses_test, sample_sizes_test = [], [], [], [], [], []
    p1, p2, p3, p1_err, p2_err, p3_err = [], [], [], [], [], []
    p1_pval, p2_pval, p3_pval = [], [], []

    collect_test_train = {'id': [], 'target_var': [], 'y_train_pred': [], 'y_test_pred': [], 'y_train': [], 'y_test': []}

    p_group = prod_data.groupby('Prop_id')
    for mine, prod_data in tqdm(p_group, desc='Fitting models to production data'):
        for t in targets_cumsum:
            sample = prod_data.dropna(subset=t)
            
            sample_size = len(sample)
            if sample_size < min_sample_size:
                continue

            # Train-test split
            train, test = train_test_split(sample, test_size=0.2, random_state=42)
            sample_size_train = len(train)
            sample_size_test = len(test)

            for model_name, model in models.items():
                year_train = train['Year'].dt.year - train['Year'].dt.year.min()
                year_test = test['Year'].dt.year - train['Year'].dt.year.min()
                
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
                
                except RuntimeError:
                    # Handle fit failures
                    mine_names.append(mine)
                    target_var_names.append(t)
                    model_names.append(model_name)
                    r2s_train.append(np.nan)
                    rmses_train.append(np.nan)
                    sample_sizes_train.append(np.nan)

                    r2s_test.append(np.nan)
                    rmses_test.append(np.nan)
                    sample_sizes_test.append(np.nan)
                    p1.append(np.nan)
                    p2.append(np.nan)
                    p3.append(np.nan)
                    p1_err.append(np.nan)
                    p2_err.append(np.nan)
                    p3_err.append(np.nan)
                    p1_pval.append(np.nan)
                    p2_pval.append(np.nan)
                    p3_pval.append(np.nan)

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
        'P1_value': p1, 'P2_value': p2, 'P3_value': p3,
        'P1_err': p1_err, 'P2_err': p2_err, 'P3_err': p3_err,
        'P1_pval': p1_pval, 'P2_pval': p2_pval, 'P3_pval': p3_pval
    })

    # Save results to JSON
    res_df.to_json('data/int/production_model_fits.json', orient='records')
    
    return res_df


def prep_data(): 
    prod_data = pd.read_csv(r'data\int\D_build_sample_sets\target_vars_prio_source.csv')
    
    prod_data['Year'] = pd.to_datetime(prod_data['Year'], errors='coerce')
    prod_data['Start_up_year'] = pd.to_datetime(prod_data['Start_up_year'], errors='coerce')

    prod_data['Year_diff'] = (prod_data['Year'].dt.year - prod_data['Start_up_year'].dt.year).astype(int)
    return None

targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']
targets_cumsum = [i + '_cumsum' for i in targets]
init_guesses = {'hubbert': (10**6, 0.1, 20), 'power_law': (0, 0), 'femp': (10**8, 0.1)}
lower_bounds = {'hubbert': (0, 0, 0), 'power_law': (0, 0), 'femp': (0, 0)}
upper_bounds = {'hubbert': (10**8, np.inf, 100), 'power_law': (np.inf, np.inf), 'femp': (np.inf, 1)}

min_sample_size = 10

if __name__ == '__main__':
    prep_data()