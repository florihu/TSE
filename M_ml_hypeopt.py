import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.base import BaseEstimator, TransformerMixin

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

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

#models = { 'MLPRegressor': MLPRegressor(), 'SVR': SVR(), 'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}
models = {'GradientBoostingRegressor': GradientBoostingRegressor()}
rename_dict = {'Tailings_production': 'CTP', 'Concentrate_production': 'CCP', 'Ore_processed_mass': 'COP'}
n_splits = 5


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

    for name in tqdm(['Ore_processed_mass'], desc='Variables'):

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
            param_dist_ = {f'regressor__{model_name}__{k}': v for k, v in param_dist[model_name].items()}

            # Instantiate RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model_ttr,
                param_distributions=param_dist_,
                n_iter=numer_of_samples[model_name],
                scoring='neg_root_mean_squared_error',
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
    
    #df_to_csv_int(res_df, d_name)

    return res_df

def model_eval():

    models_path_per_var = {
        'Concentrate_production': ['models/MLPRegressor_synth_Concentrate_production.pkl', 'models/SVR_synth_Concentrate_production.pkl', 'models/RandomForestRegressor_synth_Concentrate_production.pkl', 'models/GradientBoostingRegressor_synth_Concentrate_production.pkl'],
        'Tailings_production': ['models/MLPRegressor_synth_Tailings_production.pkl', 'models/SVR_synth_Tailings_production.pkl', 'models/RandomForestRegressor_synth_Tailings_production.pkl', 'models/GradientBoostingRegressor_synth_Tailings_production.pkl'],
        'Ore_processed_mass': ['models/MLPRegressor_synth_Ore_processed_mass.pkl', 'models/SVR_synth_Ore_processed_mass.pkl', 'models/RandomForestRegressor_synth_Ore_processed_mass.pkl', 'models/GradientBoostingRegressor_synth_Ore_processed_mass.pkl']
    }

    res_df = pd.DataFrame(columns=['Model', 'Variable', 'Fold',  'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'NRMSE_train', 'NRMSE_test'])


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        d = get_data_per_var(name, out_remove=False, thres_out=None)
        
        if synth:
            d = get_synth_samp(d)

        y = d['Cum_prod']
        X = d.drop('Cum_prod', axis=1)

        idx = np.arange(X.shape[0])

        comb = get_comb(d)

        for train_idx, test_idx  in  StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE).split(idx, comb):
            
            count = 0
            
            for model_path in tqdm(models_path_per_var[name], desc='Models'):
                model_ttr = joblib.load(model_path)

                model_ttr.regressor.random_state = config.RANDOM_STATE

                model_name = model_path.split('_')[0].split('/')[-1]

            
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    
                model_ttr.fit(X_train, y_train)

                y_train_pred = model_ttr.predict(X_train)
                y_test_pred = model_ttr.predict(X_test)
                
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
        d_name = f'ml_hype_loop_eval_synth'
    else:
        d_name = f'ml_hype_loop_eval'

    df_to_csv_int(res_df, d_name)

    pass

def plot_model_eval():
    # File path
    p_file = r'data\int\M_ml_hypeopt\ml_hype_loop_eval_synth.csv'

    color_dict = {'train': '#b2df8a', 'test': '#33a02c'}

    # Read the data
    df = pd.read_csv(p_file)

    # Define the model order
    model_order = [
        'MLPRegressor', 'SVR', 
        'RandomForestRegressor', 'GradientBoostingRegressor'
    ]
    df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)

    # Convert RMSE to Mt (millions)
    df[['RMSE_train', 'RMSE_test']] /= 10**6

    # filter out r2_tests smaller -.5
    #df = df[df['R2_test'] > -0.5]

    # Melt the DataFrame
    melt = df.melt(
        id_vars=['Model', 'Variable', 'Fold'], 
        value_vars=['R2_test', 'R2_train', 'RMSE_train', 'RMSE_test'],
        var_name='Metric', 
        value_name='Value'
    )

    # Split metric into Metric_Type and Data_Split
    melt[['Metric_Type', 'Data_Split']] = melt['Metric'].str.extract(r'(\w+)_(train|test)')

    # Ensure consistent ordering
    melt['Data_Split'] = pd.Categorical(melt['Data_Split'], categories=['train', 'test'], ordered=True)

    # Filter for R² and RMSE only
    melt = melt[melt['Metric_Type'].isin(['R2', 'RMSE'])]
    melt['Metric_Type'] = melt['Metric_Type'].replace({'RMSE': 'RMSE (Mt)'})

    melt['Variable'] = melt['Variable'].replace({'Ore_processed_mass': 'COP', 'Concentrate_production': 'CCP', 'Tailings_production': 'CTP'})

    # Initialize a FacetGrid
    g = sns.FacetGrid(
        melt, col='Metric_Type', row='Variable', hue='Data_Split',
        sharey=False, height=4, aspect=1.5, sharex=True,  
    )

    # Plot boxplots in each facet
    g.map_dataframe(sns.boxplot, x='Model', y='Value', hue='Data_Split', order=model_order, dodge=True, palette=color_dict)

    # Adjust legends
    g.add_legend(title="Data Split", fontsize=10, title_fontsize=10, loc = [0.1, 0.2])


    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(model_order))))
        ax.set_xticklabels(model_order, rotation=45, ha="right")
    
    # Add axis labels
    g.set_axis_labels("Model", "Value")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    save_fig('ml_hype_loop_eval_synth', dpi=800)

    plt.show()




if __name__ =='__main__':
    plot_model_eval()