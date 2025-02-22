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
from sklearn.model_selection import train_test_split
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
pre_pipe = make_pipeline(MinMaxScaler(), PCA(n_components=0.95))


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

def train_loop():
    
    res_df = pd.DataFrame(columns=['Model', 'Variable', 'Fold',  'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'NRMSE_train', 'NRMSE_test'])


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        d = get_data_per_var(name, out_remove=out_remove)

        y = d['Cum_prod']
        X = d.drop('Cum_prod', axis=1)

        comb = get_comb(d)
        
        

        idx = np.arange(X.shape[0])

        for train_idx, test_idx  in  StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(idx, comb):
            
            count = 0
            
            for model_name, model in tqdm(models.items()):

                # add model to pipe
                model = make_pipeline(pre_pipe, model)
            
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                r2_train = model.score(X_train, y_train)
                r2_test = model.score(X_test, y_test)
                rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
                mae_train = np.mean(np.abs(y_train - y_train_pred))
                mae_test = np.mean(np.abs(y_test - y_test_pred))
                nrmse_train = rmse_train / np.mean(y_train)
                nrmse_test = rmse_test / np.mean(y_test)

                res_df = pd.concat([res_df, pd.DataFrame([{ 'Model': model_name, 'Variable': name, 'Fold':count,'R2_train': r2_train, 'R2_test': r2_test, 'RMSE_train': rmse_train, 'RMSE_test': rmse_test, 'MAE_train': mae_train, 'MAE_test': mae_test, 'NRMSE_train': nrmse_train, 'NRMSE_test': nrmse_test}])], ignore_index=True)
            
            count += 1
    
    if out_remove:
        d_name = 'ml_train_loop_result_out_removed'
    else:
        d_name = 'ml_train_loop_result'

    df_to_csv_int(res_df, d_name)

    pass
                
def plot_train_results():
    # Melt the DataFrame

    if out_remove:
        p = r'data\int\M_ml_train_loop\ml_train_loop_result_out_removed.csv'
    else:
        p = r'data\int\M_ml_train_loop\ml_train_loop_result.csv'

    df = pd.read_csv(p)

    model_order = [
        'ElasticNet', 'Lasso', 'LinearRegression', 'Ridge', 'MLPRegressor', 
        'SVR', 'RandomForestRegressor', 'GradientBoostingRegressor'
    ]


    # Ensure the 'Model' column is ordered
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)



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
    color_dict = {'train': '#7570b3', 'test': '#d95f02'}

    melt = melt[(melt.Model != 'MLPRegressor')]

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

        if out_remove:
            save_fig_plotnine(p, f'ml_train_res__out_removed', dpi=800)
        else:
            # Save the plot
            save_fig_plotnine(p, f'{v}_ml_train_res', dpi=800)

    pass

def descriptive_analysis(p = r'data\int\M_ml_train_loop\ml_train_loop_result.csv'):
    d = pd.read_csv(p)
    desc = d.groupby(['Variable', 'Model']).describe()
    desc = desc.round(4)

    
    pass



if __name__ == '__main__':
    
    descriptive_analysis()
    





