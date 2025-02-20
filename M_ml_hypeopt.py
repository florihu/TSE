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
from plotnine import *
import joblib


# import pca
from sklearn.decomposition import PCA

from tqdm import tqdm

from util import df_to_csv_int, save_fig, save_fig_plotnine
from E_pca import get_data_per_var
from E_ml_explo import log_vars
from M_ml_train_loop import get_comb, pre_pipe

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
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [1000, 1500, 2000]
        },
        'Ridge': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
        },
        'SVR': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'epsilon': [0.001, 0.01, 0.1, 0.5, 1.0],
            'gamma': ['scale', 'auto']
        },
        'RandomForestRegressor': {
            'n_estimators': [10, 50, 100, 200, 300, 400, 500],
            'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
            'max_features': np.arange(1,80).tolist()
        },
        'GradientBoostingRegressor': {
            'n_estimators': [10, 50, 100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
            'max_features': np.arange(1,80).tolist()
        }
    }

random_state = 42
models = { 'ElasticNet': ElasticNet(), 'Lasso': Lasso(),  'Ridge': Ridge(), 'SVR': SVR(),   'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}

rename_dict = {'Tailings_production': 'CTP', 'Concentrate_production': 'CCP', 'Ore_processed_mass': 'COP'}

###############################################################Main#####################################################################################################




def hype_results_plot(p = r'data\int\M_ml_hypeopt\ml_hype_opt_results.csv'):
    # Melt the DataFrame

    df = pd.read_csv(p)

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

    # color for train and test
    color_dict = {'train': '#7570b3', 'test': '#d95f02'}
        
        # make a scatter plot with model on x

    melt_piv['Variable'] = melt_piv['Variable'].replace(rename_dict)

    p = (ggplot(melt_piv, aes(x='Model', y='R2_mean', color='Data_split'))
            
        + geom_point(position=position_dodge(width=0.3), size=3)  # Shift points
        + geom_errorbar(aes(ymin='R2_mean - R2_std', ymax='R2_mean + R2_std'),
                                position=position_dodge(width=0.3), width=0.2)
        + labs(x='Model', y='R2') 
        + theme_minimal() 
        + facet_wrap('~ Variable', scales='fixed', ncol=3)
        + theme(axis_text_x=element_text(rotation=45, hjust=1, vjust=1, size=10), 
                axis_text_y=element_text(size=10),
                axis_title_x=element_text(size=10),
                axis_title_y=element_text(size=10),
                legend_title=element_text(size=10),
                legend_text=element_text(size=10),
                figure_size=(12, 6))
        + scale_color_manual(values=color_dict)

        )
  
    # Save the plot
    save_fig_plotnine(p, f'hype_opt_results', dpi=800)

    pass

def hype_loop():
    

    res_df = pd.DataFrame(columns=['Model', 'Variable', 'R2_mean_train', 'R2_mean_test', 'R2_std_train', 'R2_std_test'])

    for name in tqdm(['Concentrate_production', 'Tailings_production', 'Ore_processed_mass'], desc='Variables'):

        d = get_data_per_var(name)

        y = d['Cum_prod']
        X = d.drop('Cum_prod', axis=1)

        comb = get_comb(d)
        
        strat_kflold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for model_name, model in models.items():
            # add random state to the model
            model.random_state = random_state

            model = Pipeline([('Preprocessing', pre_pipe), (model_name, model)])

            #add model_name to params keys
            param_dist_ = {f'{model_name}__{k}': v for k, v in param_dist[model_name].items()}


            # Instantiate RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist_,
                n_iter=numer_of_samples[model_name],
                scoring='r2',
                n_jobs=-1,
                cv=strat_kflold.split(X, comb),  # Pass stratified groups here
                random_state=random_state,
                return_train_score=True
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            # get r2 scores for the best estimator
            r2_mean_train = random_search.cv_results_['mean_train_score'][random_search.best_index_]
            r2_mean_test = random_search.cv_results_['mean_test_score'][random_search.best_index_]
            r2_std_train = random_search.cv_results_['std_train_score'][random_search.best_index_]
            r2_std_test = random_search.cv_results_['std_test_score'][random_search.best_index_]



            # save best estimater to path
            model_path = f'models/{model_name}_{name}.pkl'
            joblib.dump(random_search.best_estimator_, model_path)


            res_df = pd.concat([res_df, pd.DataFrame([{ 'Model': model_name, 'Variable': name, 'R2_mean_train': r2_mean_train, 'R2_mean_test': r2_mean_test, 'R2_std_train': r2_std_train, 'R2_std_test': r2_std_test}])], ignore_index=True)
        
    df_to_csv_int(res_df, 'ml_hype_opt_results')

    return res_df


if __name__ =='__main__':
    hype_results_plot()