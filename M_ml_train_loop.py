'''
This script is for 
* identifying the sample realtions copper-zinc-nickel to ensure that no commodity is overproportionally represented.
* Engineering of a geography similarity feature
* Feature selection, scaling, pca, 
* Fitting models
* Hyperparameter integration

Output:
* df with th model name, metrics
* optimal models are saved to models folder

'''
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.base import BaseEstimator, TransformerMixin

from matplotlib import pyplot as plt

from E_ml_explo import immpute_vars, unit_rename
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from util import save_fig, save_fig_plotnine

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from plotnine import *
import joblib

import smogn

# import pca
from sklearn.decomposition import PCA

from tqdm import tqdm

from util import df_to_csv_int, save_fig, save_fig_plotnine
from E_pca import get_data_per_var
from E_ml_explo import log_vars



##################################################Purpose####################################################

##################################################Params####################################################

models = { 'ElasticNet': ElasticNet(), 'Lasso': Lasso(), 'LinearRegression': LinearRegression(), 'MLPRegressor': MLPRegressor(), 'Ridge': Ridge(), 'SVR': SVR(),   'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}
random_state = 42
out_remove = False
synth = True

n_splits = 5
thres_out = 5
pre_pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))

y_pipe = make_pipeline(FunctionTransformer(np.log, np.exp), StandardScaler())


##################################################Functions####################################################

   
def stratified_train_test_split(df, test_size=0.2, random_state=None):
    """
    Splits the dataset into train and test subsets with stratification based on production variables.

    Args:
    - df (pd.DataFrame): Input dataframe containing binary variables for production.
    - test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
    - random_state (int, optional): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Training subset.
    - pd.DataFrame: Testing subset.
    """
    # Create a unique label for stratification
    df['Copper'] = df['Primary_Copper'] + df['Byprod_Copper']
    df['Zinc'] = df['Primary_Zinc'] + df['Byprod_Zinc']
    df['Nickel'] = df['Primary_Nickel'] + df['Byprod_Nickel']
    production_features = ['Copper', 'Zinc', 'Nickel']
    
    df[production_features] = df[production_features].astype(int)
    df['Combination'] = df[production_features].astype(str).agg('-'.join, axis=1)

    index = df.index
    # Perform stratified train-test split
    train_idx, test_idx = train_test_split(index, test_size=test_size, stratify=df['Combination'], random_state=random_state)
   # only return index of dfs
    return train_idx, test_idx

def get_comb(df):

    # only if the columns are present

    # Feature for stratisfied sampling
    df['Copper'] = df['Primary_Copper'] + df['Byprod_Copper']
    df['Zinc'] = df['Primary_Zinc'] + df['Byprod_Zinc']
    df['Nickel'] = df['Primary_Nickel'] + df['Byprod_Nickel']
    production_features = ['Copper', 'Zinc', 'Nickel']
    
    df[production_features] = df[production_features].astype(int)
    comb= df[production_features].astype(str).agg('-'.join, axis=1)
    return comb

def get_synth_samp(data):
    data['Cum_prod'] = np.log(data['Cum_prod'])
    data.reset_index(inplace=True, drop=True)
    

    rg_mtrx = [
            [data['Cum_prod'].quantile(0.50), 0, 0],   # Below median: No oversampling
            [data['Cum_prod'].quantile(0.75), 1, 0],  # Moderate relevance in upper quartile
            [data['Cum_prod'].quantile(0.90), 1, 0],  # High relevance above 90% quantile
            [data['Cum_prod'].max(), 0, 0]  # Ensure extreme max values are captured
        ]
        
    smog = smogn.smoter(
            data, 
            y='Cum_prod', 
            k=5, 
            samp_method='balance', 
            rel_method="manual", 
            rel_thres=0.9,
            rel_ctrl_pts_rg = rg_mtrx

        )
    
    smog['Cum_prod'] = np.exp(smog['Cum_prod'])

    return smog


def r2_calc(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def train_loop():
    
    res_df = pd.DataFrame(columns=['Model', 'Variable', 'Fold',  'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'NRMSE_train', 'NRMSE_test'])


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        d = get_data_per_var(name, out_remove=out_remove, thres_out=thres_out)
        
        if synth:
            d = get_synth_samp(d)

        y = d['Cum_prod']
        X = d.drop('Cum_prod', axis=1)

        idx = np.arange(X.shape[0])

        comb = get_comb(d)

        for train_idx, test_idx  in  StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(idx, comb):
            
            count = 0
            
            for model_name, model in tqdm(models.items()):

                # add model to pipe
                model = make_pipeline(pre_pipe, model)
            
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                y_train = y_pipe.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                y_test = y_pipe.transform(y_test.values.reshape(-1, 1)).flatten()

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                y_train_pred = y_pipe.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
                y_test_pred = y_pipe.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
                y_train = y_pipe.inverse_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_pipe.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                r2_train = r2_calc(y_train, y_train_pred)
                r2_test = r2_calc(y_test, y_test_pred)

                rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
                mae_train = np.mean(np.abs(y_train - y_train_pred))
                mae_test = np.mean(np.abs(y_test - y_test_pred))
                nrmse_train = rmse_train / np.mean(y_train)
                nrmse_test = rmse_test / np.mean(y_test)

                res_df = pd.concat([res_df, pd.DataFrame([{ 'Model': model_name, 'Variable': name, 'Fold':count,'R2_train': r2_train, 'R2_test': r2_test, 'RMSE_train': rmse_train, 'RMSE_test': rmse_test, 'MAE_train': mae_train, 'MAE_test': mae_test, 'NRMSE_train': nrmse_train, 'NRMSE_test': nrmse_test}])], ignore_index=True)
            
            count += 1
    
    if synth:
        d_name = f'ml_train_loop_result_synth'
    else:
        d_name = f'ml_train_loop_result'

    df_to_csv_int(res_df, d_name)

    pass
                
def plot_train_results():
    # Melt the DataFrame

    if synth:
        p = r'data\int\M_ml_train_loop\ml_train_loop_result_synth.csv'
        color_dict = {'train': '#c51b7d', 'test': '#7fbc41'}
    else:
        p = r'data\int\M_ml_train_loop\ml_train_loop_result.csv'
        
        color_dict = {'train': '#7570b3', 'test': '#d95f02'}

    df = pd.read_csv(p)

    model_order = [
        'ElasticNet', 'Lasso', 'LinearRegression', 'Ridge', 'MLPRegressor', 
        'SVR', 'RandomForestRegressor', 'GradientBoostingRegressor'
    ]


    # Ensure the 'Model' column is ordered
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

    df[['RMSE_train', 'RMSE_test']] = df[['RMSE_train', 'RMSE_test']] / 10**6

    
    melt = df.melt(
        id_vars=['Model', 'Variable', 'Fold'], 
        value_vars=['R2_test', 'R2_train', 'RMSE_train', 'RMSE_test', 
                    'MAE_train', 'MAE_test', 'NRMSE_train', 'NRMSE_test'], 
        var_name='Metric', 
        value_name='Value'
    )

   
    # Split 'Metric' into 'Metric_Type' and 'Data_Split'
    melt[['Metric_Type', 'Data_Split']] = melt['Metric'].str.extract(r'(\w+)_(train|test)')
    
    split_order = ['train', 'test']

    # Ensure the 'Data_Split' column is ordered
    melt['Data_Split'] = pd.Categorical(melt['Data_Split'], categories=split_order, ordered=True)

    # color for train and test
    

    melt = melt[melt['Metric_Type'].isin(['R2', 'RMSE'])]

    melt['Metric_Type'] = melt['Metric_Type'].replace({'RMSE': 'RMSE (Mt)'})


    for v in ['Concentrate_production', 'Tailings_production', 'Ore_processed_mass']:
        
        melt_v = melt[melt['Variable'] == v]
   
        # Create a facet plot for each metric, coloring by 'Data_Split'
        p = (ggplot(melt_v, aes(x='Model', y='Value', color='Data_Split')) 
             + geom_boxplot() 
             + facet_wrap('Metric_Type', scales='free_y', ncol = 2) 
             + labs(x='Model', y='Value')
            + theme_minimal()
             + theme(
                 axis_text_x=element_text(rotation=45, hjust=1, vjust=1, size=10),
                 figure_size=(12, 6)
             )
        )

        #use color_dict
        p += scale_color_manual(values=color_dict)

        if synth:
            save_fig_plotnine(p, f'{v}_ml_train_res_out_synth', dpi=800, w=12, h=6)
        else:
            # Save the plot
            save_fig_plotnine(p, f'{v}_ml_train_res', dpi=800, w=12, h=6)

    pass

def descriptive_analysis(p = r'data\int\M_ml_train_loop\ml_train_loop_result.csv'):
    d = pd.read_csv(p)
    desc = d.groupby(['Variable', 'Model']).describe()
    desc = desc.round(4)
    pass



if __name__ == '__main__':

    plot_train_results()
    
    





