'''
This script is used for the Monte Carlo simulation of the prediction of concentrate and tailing production.

1. Select model input parameters and sample from the distribution
2. Sample from the distribution of the model input parameters
3. Train the best ML model with the sampled data
4. Store the model and the sampled data. Do this 1000 times

'''

import numpy as np
import pandas as pd
import os
from scipy.stats import norm, lognorm, uniform

from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import StackingRegressor

import geopandas as gpd
import joblib
from tqdm import tqdm

import warnings


from R_prod_analysis import identify_cum_model
from M_prod_model import hubbert_model, femp
from M_ml_train_loop import immpute_vars, geography_similarity
from util import data_to_csv_int

from sklearn.base import BaseEstimator, RegressorMixin

cat_vars =  ['Primary_Chromium',
       'Byprod_Chromium', 'Primary_Cobalt', 'Byprod_Cobalt', 'Primary_Copper',
       'Byprod_Copper', 'Primary_Crude Oil', 'Byprod_Crude Oil',
       'Primary_Gold', 'Byprod_Gold', 'Primary_Indium', 'Byprod_Indium',
       'Primary_Iron', 'Byprod_Iron', 'Primary_Lead', 'Byprod_Lead',
       'Primary_Manganese', 'Byprod_Manganese', 'Primary_Molybdenum',
       'Byprod_Molybdenum', 'Primary_Nickel', 'Byprod_Nickel',
       'Primary_Palladium', 'Byprod_Palladium', 'Primary_Platinum',
       'Byprod_Platinum', 'Primary_Rhenium', 'Byprod_Rhenium',
       'Primary_Silver', 'Byprod_Silver', 'Primary_Tin', 'Byprod_Tin',
       'Primary_Titanium', 'Byprod_Titanium', 'Primary_Tungsten',
       'Byprod_Tungsten', 'Primary_Uranium', 'Byprod_Uranium',
       'Primary_Vanadium', 'Byprod_Vanadium', 'Primary_Zinc', 'Byprod_Zinc',
       'ev', 'mt', 'nd', 'pa', 'pb', 'pi', 'py', 'sc', 'sm',
       'ss', 'su', 'va', 'vb', 'vi', 'wb']
    
num_vars = ['Tailings_production', 'Concentrate_production', 'Active_years',
       'Polygon_count', 'Weight', 'Area_mine', 'Area_mine_weighted',
       'Convex_hull_area', 'Convex_hull_area_weighted',
       'Convex_hull_perimeter', 'Convex_hull_perimeter_weighted',
       'Compactness', 'Compactness_weighted', 'EPS_mean', 'EPS_slope']

warnings.filterwarnings("ignore")

def log_pipeline():
    # standard scale plus log
    return make_pipeline(FunctionTransformer(np.log1p), MinMaxScaler())
    
def no_log_pipeline():
    # standard scale plus log
    return make_pipeline(MinMaxScaler())

# if one of the scale params is not positive take a small positive value
def check_positive(x):
    if x <= 0:
        return 1e-8
    else:
        return x


var_selection = {'Tailings_production': ['Tailings_production', 'Area_mine', 'Weight', 'Compactness', 'Primary_Copper', 'Primary_Nickel', 'Primary_Silver', 'Primary_Zinc', 'va', 'max_similarity'],
                    'Concentrate_production': ['Concentrate_production','Area_mine', 'Compactness', 'EPS_mean' , 'Primary_Copper', 'Primary_Silver',  'Primary_Zinc', 'pb', 'va', 'py', 'max_similarity'],
                    }

models = {'Tailings_production': joblib.load(r'models\SVR_Tailings_production.pkl'),'Concentrate_production': joblib.load(r'models\SVR_Concentrate_production.pkl')}


transfo = {'Tailings_production': ColumnTransformer([('Tailings_production', log_pipeline(), ['Tailings_production']), 
                                                        ('Area_mine', log_pipeline(), ['Area_mine'])]
                                                        , remainder=no_log_pipeline()), 
            'Concentrate_production': ColumnTransformer([('Concentrate_production', log_pipeline(), ['Concentrate_production']), 
                                                        ('Area_mine', log_pipeline(), ['Area_mine'])  ], remainder=no_log_pipeline()) 
            }

def return_integrated_values(prop_ids, t, df_res, df_class, sample_size=1000, random_state=42):
    '''
    This function returns integrated values for the production model based on 
    given property IDs and sample size. The output dataframe will have the 
    property IDs as the index and columns for each sample iteration.

    Parameters:
    - prop_ids: List of property IDs.
    - t: Target variable for the model.
    - df_res: DataFrame containing the model parameters and error values.
    - df_class: DataFrame containing the property class information.
    - sample_size: Number of samples to draw from the distributions (default is 1000).

    Returns:
    - collect: DataFrame with the integrated values for each sample iteration.
    '''
    
    # Initialize the result DataFrame with property IDs as the index
    collect = pd.DataFrame(index=prop_ids, columns=[i for i in range(sample_size)])

    df_res['Prop_id'] = df_res['Prop_id'].astype(str)
    df_class['Prop_id'] = df_class['Prop_id'].astype(str)

    # Loop through each property ID to process the relevant data
    for id in prop_ids:
        
        # Determine the period based on the startup year for each property
        period = 2019 - int(df_res[df_res.Prop_id == id]['Start_up_year'].values[0])
        period = np.repeat(period, sample_size)

        # Get the class of the property for the target variable
        class_ = df_class[(df_class.Prop_id == id) & (df_class.Target_var == t)]['Class'].values[0]

        if class_ == 'H':  # Hubbert model
        
            # Extract parameters and errors for the Hubbert model
            p1, p2, p3 = df_res[(df_res.Prop_id == id) & 
                                (df_res.Target_var == t) &
                                (df_res.Model == 'hubbert')][['P1_value', 'P2_value', 'P3_value']].values.flatten()
            p1_err, p2_err, p3_err = df_res[(df_res.Prop_id == id) & 
                                            (df_res.Target_var == t) &
                                            (df_res.Model == 'hubbert')][['P1_err', 'P2_err', 'P3_err']].values.flatten()


            # Log-normal distribution requires log-transformed parameters for initialization
            p1_mean_log = check_positive(np.log(p1))
            p1_std_log = check_positive(p1_err / p1)  # Approximation assuming small relative error

            p3_mean_log = check_positive(np.log(p3))
            p3_std_log = check_positive(p3_err / p3)  # Approximation assuming small relative error

            
            # Initialize distributions
            p1_distrib = lognorm(s=p1_std_log, scale=np.exp(p1_mean_log))
            p2_distrib = norm(loc=p2, scale=p2_err)  # Normal distribution
            p3_distrib = lognorm(s=p3_std_log, scale=np.exp(p3_mean_log))

            # Draw samples for each parameter
            p1_sample = p1_distrib.rvs(sample_size, random_state=random_state)
            p2_sample = p2_distrib.rvs(sample_size, random_state=random_state)
            p3_sample = p3_distrib.rvs(sample_size, random_state=random_state)

            # Apply the Hubbert model and store the results
            collect.loc[id,:] = hubbert_model(period, p1_sample, p2_sample, p3_sample).flatten()

        elif class_ == 'F':  # FEM model

            # Extract parameters and errors for the FEM model
            p1, p2 = df_res[(df_res.Prop_id == id) & 
                            (df_res.Target_var == t) &
                            (df_res.Model == 'femp')][['P1_value', 'P2_value']].values.flatten()
            p1_err, p2_err = df_res[(df_res.Prop_id == id) & 
                                    (df_res.Target_var == t) &
                                    (df_res.Model == 'femp')][['P1_err', 'P2_err']].values.flatten()

            # Log-normal distribution for p1, normal distribution for p2
            p1_mean_log = check_positive(np.log(p1))
            p1_std_log = check_positive(p1_err / p1)

            p1_distrib = lognorm(s=p1_std_log, scale=np.exp(p1_mean_log))
            p2_distrib = norm(loc=p2, scale=p2_err)

            # Draw samples for each parameter
            p1_sample = p1_distrib.rvs(sample_size, random_state=random_state)
            p2_sample = p2_distrib.rvs(sample_size, random_state=random_state)

            # Apply the FEM model and store the results
            collect.loc[id,:] = femp(period, p1_sample, p2_sample).flatten()

    # negative instances 

    sample_df = collect.copy()
    print(f'Negative instances: {sample_df[sample_df < 0].count().sum()}')
    #mark negative values as nan
    sample_df[sample_df < 0] = np.nan

    sample_df = sample_df.astype(float)
    imputer = KNNImputer(n_neighbors=4)
    sample_df = pd.DataFrame(imputer.fit_transform(sample_df), columns=sample_df.columns)

    # Return the DataFrame with all the integrated values for each sample iteration
    return sample_df




class ConcatenatedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, names, models):
        """
        models: List of trained regressors to be concatenated.
        """
        self.names = names
        self.models = models

    def fit(self, X, y):
        """
        Fits each regressor in the models list.
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Makes predictions by averaging the predictions from all models.
        """
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Average the predictions from each model
        return np.mean(predictions, axis=1)

    def score(self, X, y):
        """
        Returns the R2 score of the concatenated predictions.
        """
        predictions = self.predict(X)
        return np.corrcoef(predictions, y)[0, 1] ** 2  # R^2 score


def monte_carlo(sample_size = 1000):
    

    targets_fit_p = r'data\int\D_target_prio_prep\target_vars_prio_source.csv'
    tailings_p = r'data\int\D_train_tetst_df\tailings.gpkg'
    modelres_p = r'data\int\production_model_fits.json'


    targets_fit = pd.read_csv(targets_fit_p)
    model_res = pd.read_json(modelres_p)
    model_res = pd.merge(model_res, targets_fit[['Prop_id', 'Start_up_year']].drop_duplicates(), on=['Prop_id'], how='left')   
    class_ = identify_cum_model(model_res)   
    df = gpd.read_file(tailings_p)
    df = immpute_vars(df, cat_vars, num_vars)
    df = geography_similarity(df, make_plot=False)


    # Feature for stratisfied sampling
    df['Copper'] = df['Primary_Copper'] + df['Byprod_Copper']
    df['Zinc'] = df['Primary_Zinc'] + df['Byprod_Zinc']
    df['Nickel'] = df['Primary_Nickel'] + df['Byprod_Nickel']
    production_features = ['Copper', 'Zinc', 'Nickel']
    df[production_features] = df[production_features].astype(int)
    df['Combination'] = df[production_features].astype(str).agg('-'.join, axis=1)
           

    res = []
    


    for t in tqdm(['Tailings_production', 'Concentrate_production']):

        sample_df = return_integrated_values(df.Prop_id, t, model_res, class_, sample_size=sample_size)

        df_reg = df[var_selection[t]]
        
        estimators = []
        estimator_names = []

        for i in tqdm(range(sample_size)):

            df_reg[t] = sample_df.loc[:,i].values
            df_reg[t] = df_reg[t].astype(float)

            df_trans = pd.DataFrame(transfo[t].fit_transform(df_reg), columns=df_reg.columns)
            
            y = df_trans[t]
            X = df_trans.drop(columns=[t])
            
            for j, (train_idx, test_idx)  in  enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X.index, df['Combination'])):
                
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = models[t]
                
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                r2_train = model.score(X_train, y_train)
                r2_test = model.score(X_test, y_test)
                rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
                mae_train = np.mean(np.abs(y_train - y_train_pred))
                mae_test = np.mean(np.abs(y_test - y_test_pred))
                cv_train = rmse_train / np.mean(y_train)
                cv_test = rmse_test / np.mean(y_test)

                res.append({ 'Model':model.__class__.__name__, 'Variable': t, 'Fold':j, 'Sample':i, 'R2_train': r2_train, 'R2_test': r2_test, 'RMSE_train': rmse_train, 'RMSE_test': rmse_test, 'MAE_train': mae_train, 'MAE_test': mae_test, 'CV_train': cv_train, 'CV_test': cv_test})

                estimators.append(model)
                estimator_names.append(f'{model.__class__.__name__}_{i}_{j}')

        # Concatenate the regressors
        concatenated = ConcatenatedRegressor(estimator_names, estimators)

        # Save the concatenated regressor
        joblib.dump(concatenated, f'models\SVR_ensemble_{t}.pkl')

    res_df = pd.DataFrame(res)
    
    data_to_csv_int(res_df, 'monte_carlo_ensemble_results.csv')
       
    return 


if __name__ == '__main__':
    monte_carlo()