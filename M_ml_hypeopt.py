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

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.compose import TransformedTargetRegressor
from plotnine import *
import joblib


from sklearn.metrics import make_scorer


# import pca
from sklearn.decomposition import PCA

from tqdm import tqdm

from util import df_to_csv_int, save_fig, save_fig_plotnine
from E_pca import get_data_per_var
from E_ml_explo import log_vars
from M_ml_train_loop import get_comb, pre_pipe, y_pipe, r2_calc, get_synth_samp
import config
###############################################################Purpose#####################################################################################################





###############################################################Params#####################################################################################################


numer_of_samples = {'ElasticNet': 100, 'Lasso': 100, 'MLPRegressor': 300, 'Ridge': 100, 'SVR': 200, 'RandomForestRegressor': 300, 'GradientBoostingRegressor': 300}
    
    # Parameter distributions
param_dist = {
        'ElasticNet': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [100, 500, 1000, 5000]
        },
        'Lasso': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [100, 500, 1000, 5000]
        },
        
        'MLPRegressor': {
            'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 64), (128, 128)],  # Keeping it shallow but diverse
            'activation': ['relu', 'tanh'],  # 'identity' and 'logistic' are less common for regression
            'solver': ['adam'],  # 'sgd' often struggles to converge without tuning momentum
            'alpha': np.logspace(-5, -1, 6).tolist(),  # Reduce range to prevent excessive regularization
            'learning_rate': ['constant', 'adaptive'],  # 'invscaling' is rarely beneficial in practice
            'max_iter': [500, 1000, 1500]  # 2000 is excessive for shallow models
        },
        'Ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        },
        'SVR': {
            'C': np.logspace(-2, 3, 10).tolist(),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'epsilon': np.logspace(-3, 0, 10).tolist(),
            'gamma': ['scale', 'auto']
        },
        'RandomForestRegressor': {
            'n_estimators': np.arange(50, 600, 50).tolist(),
            'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
            'max_features': np.arange(1,80,5).tolist()
        },
        'GradientBoostingRegressor': {
            'n_estimators': np.arange(50, 600, 50).tolist(),
            'learning_rate': np.logspace(-3, 0, 10).tolist(),
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 4, 8, 16, 32],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
            'max_features': np.arange(1,80,5).tolist()
        }
    }


synth = True

models = { 'MLPRegressor': MLPRegressor(), 'SVR': SVR(), 'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}

rename_dict = {'Tailings_production': 'CTP', 'Concentrate_production': 'CCP', 'Ore_processed_mass': 'COP'}


###############################################################Main#####################################################################################################


def hype_results_plot():
    # Melt the DataFrame

    if synth:
        p = r'data\int\M_ml_hypeopt\ml_hype_opt_results_synth.csv'
        d_name = 'ml_hype_opt_results_synth'
        color_dict = {'train': '#c51b7d', 'test': '#7fbc41'}
    else:
        p = r'data\int\M_ml_hypeopt\ml_hype_opt_results.csv'
        d_name = 'ml_hype_opt_results'
        
        color_dict = {'train': '#7570b3', 'test': '#d95f02'}

    df = pd.read_csv(p)
    df.dropna(axis=1, inplace=True)

    model_order = [
        'ElasticNet', 'Lasso', 'LinearRegression', 'Ridge',
        'SVR', 'RandomForestRegressor', 'GradientBoostingRegressor'
    ]

    # Ensure the 'Model' column is ordered
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

    # melt the df
    melt = df.melt(id_vars=['Model', 'Variable'], value_vars=['R2_mean_train', 'R2_mean_test', 'R2_std_train', 'R2_std_test'], var_name='Metric', value_name='Value')

   
    # Split 'Metric' into 'Metric_Type' and 'Data_Split'
    melt[['Metric', 'Data_split']] = melt['Metric'].str.extract(r'(\w+)_(train|test)')
    
    split_order = ['train', 'test']
    # Ensure the 'Data_Split' column is ordered
    melt['Data_split'] = pd.Categorical(melt['Data_split'], categories=split_order, ordered=True)

    melt_piv = melt.pivot_table(index=['Model', 'Variable', 'Data_split'], columns=['Metric'], values='Value').reset_index()

        
    # make a scatter plot with model on x

    melt_piv['Variable'] = melt_piv['Variable'].replace(rename_dict)

    p = (ggplot(melt_piv, aes(x='Model', y='R2_mean', color='Data_split'))
            
        + geom_point(position=position_dodge(width=0.3), size=3)  # Shift points
        + geom_errorbar(aes(ymin='R2_mean - R2_std', ymax='R2_mean + R2_std'),
                                position=position_dodge(width=0.3), width=0.2)
        + labs(x='Model', y='R2') 
        + theme_minimal() 
        + facet_wrap('~ Variable', scales='free_y', ncol=3)
        + theme(axis_text_x=element_text(rotation=45, hjust=1, vjust=1, size=10), 
                axis_text_y=element_text(size=10),
                axis_title_x=element_text(size=10),
                axis_title_y=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=10),
                figure_size=(16, 8))
        + scale_color_manual(values=color_dict)

        )
  
    # Save the plot
    save_fig_plotnine(p, d_name, dpi=1000, h=8, w=14)

    pass

def hype_loop():
    

    res_df = pd.DataFrame(columns=['Model', 'Variable', 'RMSE_mean_train', 'RMSE_mean_test', 'RMSE_std_train', 'RMSE_std_test'])

    for name in tqdm(['Concentrate_production', 'Tailings_production', 'Ore_processed_mass'], desc='Variables'):

        d = get_data_per_var(name)

        if synth:
            d = get_synth_samp(d)

        y = d['Cum_prod']
        X = d.drop('Cum_prod', axis=1)

        comb = get_comb(d)
        
        strat_kflold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

        for model_name, model in tqdm(models.items(), desc='Models'):
            # add random state to the model
            model.random_state = config.RANDOM_STATE

            model_pipe = Pipeline([('Preprocessing', pre_pipe), (model_name, model)])
            model_ttr = TransformedTargetRegressor(regressor=model_pipe, transformer=y_pipe)

            #add model_name to params keys
            param_dist_ = {f'{model_name}__{k}': v for k, v in param_dist[model_name].items()}

            # Instantiate RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model_ttr,
                param_distributions=param_dist_,
                n_iter=numer_of_samples[model_name],
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                cv=strat_kflold.split(X, comb),  # Pass stratified groups here
                random_state=config.RANDOM_STATE,
                return_train_score=True
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            # get r2 scores for the best estimator
            rmse_mean_train = random_search.cv_results_['mean_train_score'][random_search.best_index_] * -1
            rmse_mean_test = random_search.cv_results_['mean_test_score'][random_search.best_index_] * -1
            rmse_std_train = random_search.cv_results_['std_train_score'][random_search.best_index_]
            rmse_std_test = random_search.cv_results_['std_test_score'][random_search.best_index_]

            if synth:
                model_name = f'{model_name}_synth'

            # save best estimater to path
            model_path = f'models/{model_name}_{name}.pkl'
            joblib.dump(random_search.best_estimator_, model_path)

            print('Done with ', model_name, ' for ', name, 'rmse mean test', rmse_mean_test)


            res_df = pd.concat([res_df, pd.DataFrame([{ 'Model': model_name, 'Variable': name, 'RMSE_mean_train': rmse_mean_train, 'RMSE_mean_test': rmse_mean_test, 'RMSE_std_train': rmse_std_train, 'RMSE_std_test': rmse_std_test}])], ignore_index=True)

    if synth:
        d_name = 'ml_hype_opt_results_synth'
    else:
        d_name = 'ml_hype_opt_results'    
    
    df_to_csv_int(res_df, d_name)

    return res_df


if __name__ =='__main__':
    hype_results_plot()



    # def safe_exp(x, max_val=1e18):
        #     x = np.asarray(x)  # Ensure input is an array
        #     return np.where(x > np.log(max_val), np.inf, np.exp(x))

        # def safe_sqrt(x, max_val=1e18):
        #     x = np.asarray(x)  # Ensure input is an array
        #     return np.where(x > max_val, np.inf, np.sqrt(x))

        # def rmse_inv(y_true, y_pred):
        #         y_true_orig = y_pipe.inverse_transform(y_true.reshape(-1, 1)).flatten()
        #         y_pred_orig = y_pipe.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        #         return safe_sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
        
        # def r2_inv(y_true, y_pred):
        #     y_true_orig = y_pipe.inverse_transform(y_true.reshape(-1, 1)).flatten()
        #     y_pred_orig = y_pipe.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        #     return r2_calc(y_true_orig, y_pred_orig)	
            
        # #rmse_scorer = make_scorer(rmse_inv, greater_is_better=False)
        # r2_scorer = make_scorer(r2_inv, greater_is_better=True)