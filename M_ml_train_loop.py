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



##################################################Purpose####################################################

##################################################Params####################################################

models = { 'ElasticNet': ElasticNet(), 'Lasso': Lasso(), 'LinearRegression': LinearRegression(), 'MLPRegressor': MLPRegressor(), 'Ridge': Ridge(), 'SVR': SVR(),   'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}
    
out_remove = True


##################################################Functions####################################################

def geography_similarity(df, make_plot = False):
    class ClusterSimilarity(BaseEstimator, TransformerMixin): 
        def __init__(self, n_clusters=10, gamma=1.0, random_state=None): 
            self.n_clusters = n_clusters 
            self.gamma = gamma 
            self.random_state = random_state 

        def fit(self, X, y=None, sample_weight=None): 
            self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state) 
            self.kmeans_.fit(X, sample_weight=sample_weight) 
            return self 
            
        def transform(self, X): 
            return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

        
        def get_feature_names_out(self, names=None): 
            return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

    
    # Create a new feature for the geographical similarity and cluster
    coords = pd.DataFrame([df.geometry.x, df.geometry.y]).T

    cluster_sim = ClusterSimilarity(n_clusters=10, gamma=1.0 / (2 * coords.var().min()))

    sim = cluster_sim.fit_transform(coords)
    
    
    
    df['max_similarity'] = sim.max(axis=1)

    if make_plot:
        #assert wb is not None, 'Please provide a world boundary shapefile to plot the results.'
        world_bounds_p = r'data\world_bound\world-administrative-boundaries.shp'

        wb = gpd.read_file(world_bounds_p)
        wb.to_crs(df.crs, inplace=True)

        f, ax = plt.subplots(1, 1, figsize=(14, 7))
        # plot cluster centers and max similarity
        sns.scatterplot(x=df.geometry.x, y=df.geometry.y, data=df, hue='max_similarity', ax=ax, palette='viridis')
        sns.scatterplot(x= cluster_sim.kmeans_.cluster_centers_[:, 0], y= cluster_sim.kmeans_.cluster_centers_[:, 1], c='red', marker='x',  s=100, ax = ax)
        wb.boundary.plot(ax=ax, color='black', linewidth=.5)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        save_fig('geography_similarity.png')

        plt.show()  

        

    return df

   
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


def hype_loop(df):


    df = geography_similarity(df, wb=None, make_plot=False)

    var_selection = {'Tailings_production': ['Tailings_production', 'Area_mine', 'Weight', 'Compactness', 'Primary_Copper', 'Primary_Nickel', 'Primary_Silver', 'Primary_Zinc', 'va', 'max_similarity'],
                     'Concentrate_production': ['Concentrate_production','Area_mine', 'Compactness', 'EPS_mean' , 'Primary_Copper', 'Primary_Silver',  'Primary_Zinc', 'pb', 'va', 'py', 'max_similarity'],
                     }
    

    # Feature for stratisfied sampling
    df['Copper'] = df['Primary_Copper'] + df['Byprod_Copper']
    df['Zinc'] = df['Primary_Zinc'] + df['Byprod_Zinc']
    df['Nickel'] = df['Primary_Nickel'] + df['Byprod_Nickel']
    production_features = ['Copper', 'Zinc', 'Nickel']
    
    df[production_features] = df[production_features].astype(int)
    df['Combination'] = df[production_features].astype(str).agg('-'.join, axis=1)
    
    def log_pipeline():
        # standard scale plus log
        return make_pipeline(FunctionTransformer(np.log1p), MinMaxScaler())
    
    def no_log_pipeline():
        # standard scale plus log
        return make_pipeline(MinMaxScaler())
    

    models = {  'ElasticNet': ElasticNet(), 'Lasso': Lasso(),  'MLPRegressor': MLPRegressor(), 'Ridge': Ridge(), 'SVR': SVR(),   'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}
    
    
    transfo = {'Tailings_production': ColumnTransformer([('Tailings_production', log_pipeline(), ['Tailings_production']), 
                                                         ('Area_mine', log_pipeline(), ['Area_mine'])]
                                                         , remainder=no_log_pipeline()), 
                'Concentrate_production': ColumnTransformer([('Concentrate_production', log_pipeline(), ['Concentrate_production']), 
                                                            ('Area_mine', log_pipeline(), ['Area_mine'])  ], remainder=no_log_pipeline()) 
                }
    
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
            'max_iter': [200, 500, 1000]
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
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [10, 50, 100, 200, 300, 400, 500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }

    res_df = pd.DataFrame(columns=['Model', 'Variable', 'R2_mean_train', 'R2_mean_test', 'R2_std_train', 'R2_std_test'])

    for target_var, all_vars in var_selection.items():

        df_trans = df[all_vars]
        df_trans = pd.DataFrame(transfo[target_var].fit_transform(df_trans), columns=all_vars)


        y = df_trans[target_var]
        X = df_trans.drop(target_var, axis=1)
        X = pd.DataFrame(PCA(n_components=0.95).fit_transform(X))
        

        strat_kflold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in tqdm(models.items()):
            
            # Instantiate RandomizedSearchCV
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist[model_name],
                n_iter=numer_of_samples[model_name],
                scoring='r2',
                n_jobs=-1,
                cv=strat_kflold.split(X, df['Combination']),  # Pass stratified groups here
                random_state=42,
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
            model_path = f'models/{model_name}_{target_var}.pkl'
            joblib.dump(random_search.best_estimator_, model_path)


            res_df = pd.concat([res_df, pd.DataFrame([{ 'Model': model_name, 'Variable': target_var, 'R2_mean_train': r2_mean_train, 'R2_mean_test': r2_mean_test, 'R2_std_train': r2_std_train, 'R2_std_test': r2_std_test}])], ignore_index=True)
        
    data_to_csv_int(res_df, 'ml_hype_opt_results')

    return res_df


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
    
    res_df = pd.DataFrame(columns=['Model', 'Variable', 'Fold',  'R2_train', 'R2_test', 'RMSE_train', 'RMSE_test', 'MAE_train', 'MAE_test', 'CV_train', 'CV_test'])

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        d = get_data_per_var(name, out_remove=out_remove)

        y = d['Cum_prod'].to_numpy()   
        X = d.drop('Cum_prod', axis=1)

        comb = get_comb(d)
        
        #Min Max Scale
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # pca
        X = PCA(n_components=0.95).fit_transform(X)

        idx = np.arange(X.shape[0])

        for train_idx, test_idx  in  StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(idx, comb):
            
            count = 0
            
            for model_name, model in tqdm(models.items()):
            
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

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

    melt = melt[(melt.Metric_Type.isin(['R2', 'RMSE']) & (melt.Model != 'MLPRegressor'))]

    for v in ['Concentrate_production', 'Tailings_production', 'Ore_processed_mass']:
        
        melt_v = melt[melt['Variable'] == v]
   
        # Create a facet plot for each metric, coloring by 'Data_Split'
        p = (ggplot(melt_v, aes(x='Model', y='Value', color='Data_Split')) 
             + geom_boxplot() 
             + facet_wrap('Metric_Type', scales='free') 
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
            save_fig_plotnine(p, f'ml_train_res_{v}_out_removed', dpi=800)
        else:
            # Save the plot
            save_fig_plotnine(p, f'ml_train_res_{v}', dpi=800)

    pass


def hype_results(df, repfig = True):
    # Melt the DataFrame

    model_order = [
        'ElasticNet', 'Lasso', 'LinearRegression', 'Ridge', 'MLPRegressor', 
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


    for v in ['Concentrate_production', 'Tailings_production']:
    
        
        # make a scatter plot with model on x

        p = (ggplot(melt_piv[melt_piv['Variable'] == v], aes(x='Model', y='R2_mean', color='Data_split')) 
             
        + geom_point(position=position_dodge(width=0.3), size=3)  # Shift points
        + geom_errorbar(aes(ymin='R2_mean - R2_std', ymax='R2_mean + R2_std'),
                             position=position_dodge(width=0.3), width=0.2)
        + labs(x='Model', y='R2') 
        + theme_minimal() 
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
        save_fig_plotnine(p, f'{v}_hype_opt_results', dpi=600)


def plot_hype_results(path = r'data\int\M_ml_train_loop\ml_hype_opt_results.csv'):
    df = pd.read_csv(path)
    hype_results(df, repfig=True)

if __name__ == '__main__':
    train_loop()
    plot_train_results()





