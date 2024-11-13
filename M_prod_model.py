import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from D_load_werner import c_prod
from scipy import stats
from tqdm import tqdm


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


def fit_models():
    '''
    Fit models to production data
    
    Parameters:
        prod_data (pd.DataFrame): Production data
        
    Returns:
        dict: Dictionary of model fits
    '''
    prod_data = prep_data(c_prod)
    models = {'hubbert': hubbert_model, 'power_law': power_law, 'femp': femp}

   

    mine_names = []
    target_var_names = []
    model_names = []
    r2s = []
    rmses = []
    params = []
    sample_sizes = []
    param_uncertainties = []
    param_pvalues = []

    p_group = prod_data.groupby('Prop_name')
    for mine, prod_data in tqdm(p_group, desc='Fitting models to production data'):
        for t in targets_cumsum:
            sample = prod_data.dropna(subset=t)
            sample_size = len(sample)
            if sample_size < min_sample_size:
                continue

            for model_name, model in models.items():
                    # if model_name == 'hubbert_L_restrict':
                    #     popt, pcov = curve_fit(model, prod_data['Year'], prod_data[t], args=(prod_data[t].max()))
                    # else:
                    #     popt, pcov = curve_fit(model, prod_data['Year'], prod_data[t], args=(prod_data[t].max()))
                year_trans = sample['Year'].dt.year - sample['Year'].dt.year.min()
                year_trans_int = year_trans.astype(int)

                try: 
                    popt, pcov = curve_fit(model, year_trans_int, sample[t], p0=init_guesses[model_name], maxfev=10000, bounds=(lower_bounds[model_name], upper_bounds[model_name]))
                    perr = np.sqrt(np.diag(pcov))
                    pred = model(year_trans_int, *popt)

                    t_stats = popt / perr
                    p_values = [2 * (1 - stats.t.cdf(np.abs(t), sample_size - len(popt))) for t in t_stats]

                    r2 = np.corrcoef(sample[t], pred)[0, 1]**2
                    rmse = np.sqrt(np.mean((sample[t] - pred)**2))

                    mine_names.append(mine)
                    target_var_names.append(t)
                    model_names.append(model_name)
                    r2s.append(r2)
                    rmses.append(rmse)
                    params.append(popt)
                    param_uncertainties.append(perr)
                    param_pvalues.append(p_values)
                    sample_sizes.append(sample_size)
                
                except RuntimeError:
                    # Append NaNs if fitting fails
                    mine_names.append(mine)
                    target_var_names.append(t)
                    model_names.append(model_name)
                    r2s.append(np.nan)
                    rmses.append(np.nan)
                    params.append([np.nan] * len(init_guesses[model_name]))
                    param_uncertainties.append([np.nan] * len(init_guesses[model_name]))
                    param_pvalues.append([np.nan] * len(init_guesses[model_name]))
                    sample_sizes.append(sample_size)
        


    res_df = pd.DataFrame({'Mine_ID': mine_names, 'Target_var': target_var_names, 'Model': model_names, 'R2': r2, 
                           'RMSE': rmse, 'Params': params, 'Sample_size': sample_sizes, 'Param_uncertainties': param_uncertainties, 'Param_pvalues': param_pvalues})

    # to json
    res_df.to_json('data\int\production_model_fits.json', orient='records')
    
    return res_df


def prep_data(data):
    data['Concentrate_production'] = data[['Cu_concentrate_production', 'Mo_concentrate_production', 'Pb_concentrate_production', 'Zn_concentrate_production']].sum( axis=1)
    data['Concentrate_production'] = data['Concentrate_production'].fillna(0)
    drop_na = data.dropna(subset=['Year', 'Prop_name'])
    
    drop_na[[i +'_cumsum' for i in targets]] = (
        drop_na.sort_values(['Prop_name', 'Year'])
            .groupby('Prop_name')[targets]
            .cumsum()
    )
    return drop_na


targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']
targets_cumsum = [i + '_cumsum' for i in targets]
init_guesses = {'hubbert': (10**6, 0.1, 20), 'power_law': (0, 0), 'femp': (10**8, 0.1)}
lower_bounds = {'hubbert': (0, 0, 0), 'power_law': (0, 0), 'femp': (0, 0)}
upper_bounds = {'hubbert': (10**8, np.inf, 100), 'power_law': (np.inf, np.inf), 'femp': (np.inf, np.inf)}

min_sample_size = 10

if __name__ == '__main__':
    fit_models()