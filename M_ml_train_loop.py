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

from E_sample_explo import immpute_vars, unit_rename
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

from util import data_to_csv_int, save_fig, save_fig_plotnine


def geography_similarity(df, wb = None,  make_plot = False):
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
        assert wb is not None, 'Please provide a world boundary shapefile to plot the results.'
        wb.to_crs(df.crs, inplace=True)

        f, ax = plt.subplots(1, 1, figsize=(14, 7))
        # plot cluster centers and max similarity
        sns.scatterplot(x=df.geometry.x, y=df.geometry.y, data=df, hue='max_similarity', ax=ax, palette='viridis')
        sns.scatterplot(x= cluster_sim.kmeans_.cluster_centers_[:, 0], y= cluster_sim.kmeans_.cluster_centers_[:, 1], c='red', marker='x',  s=100, ax = ax)
        wb.boundary.plot(ax=ax, color='black', linewidth=.5)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.show()

        save_fig('geography_similarity.png')

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
    

    models = { 'RandomForestRegressor': RandomForestRegressor(), 'GradientBoostingRegressor': GradientBoostingRegressor()}
    
    
    transfo = {'Tailings_production': ColumnTransformer([('Tailings_production', log_pipeline(), ['Tailings_production']), 
                                                         ('Area_mine', log_pipeline(), ['Area_mine'])]
                                                         , remainder=no_log_pipeline()), 
                'Concentrate_production': ColumnTransformer([('Concentrate_production', log_pipeline(), ['Concentrate_production']), 
                                                            ('Area_mine', log_pipeline(), ['Area_mine'])  ], remainder=no_log_pipeline()) 
                }
    
    param_dist = {'RandomForestRegressor': {'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64], 'max_features': ['auto', 'sqrt', 'log2'], 'bootstrap': [True, False]},
                    'GradientBoostingRegressor': {'n_estimators': [10, 50, 100, 200, 300, 400, 500], 'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64], 'max_features': ['auto', 'sqrt', 'log2']}}

    res_df = pd.DataFrame(columns=['Model', 'Variable', 'Score'])

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
                n_iter=100,
                scoring='explained_variance',
                n_jobs=-1,
                cv=strat_kflold.split(X, df['Combination']),  # Pass stratified groups here
                random_state=42,
                return_train_score=True
            )

            # Fit RandomizedSearchCV
            random_search.fit(X, y)

            score = random_search.best_score_

            # save best estimater to path
            model_path = f'models/{model_name}_{target_var}.pkl'
            joblib.dump(random_search.best_estimator_, model_path)


            res_df = pd.concat([res_df, pd.DataFrame([{
                                    'Model': model_name, 
                                    'Variable': target_var, 
                                    'Score': score,
                                }])], ignore_index=True)
        
    data_to_csv_int(res_df, 'ml_hype_opt_results')

    return res_df

                
def plot_results(df):
    # Melt the DataFrame
    melt = df.melt(
        id_vars=['Model', 'Variable', 'Fold'], 
        value_vars=['R2_test', 'R2_train', 'RMSE_train', 'RMSE_test', 
                    'MAE_train', 'MAE_test', 'CV_train', 'CV_test'], 
        var_name='Metric', 
        value_name='Value'
    )

   
    # Split 'Metric' into 'Metric_Type' and 'Data_Split'
    melt[['Metric_Type', 'Data_Split']] = melt['Metric'].str.extract(r'(\w+)_(train|test)')
    
    for v in ['Concentrate_production', 'Tailings_production']:
        melt_v = melt[melt['Variable'] == v]
        
        # Create a facet plot for each metric, coloring by 'Data_Split'
        p = (ggplot(melt_v, aes(x='Model', y='Value', color='Data_Split')) 
             + geom_boxplot() 
             + facet_wrap('Metric_Type', scales='free') 
             + labs(x='Model', y='Value')
            + theme_minimal()
             + theme(
                 axis_text_x=element_text(rotation=45, hjust=1, vjust=1, size=8),
                 figure_size=(12, 8)
             )
        )

        # Save the plot
        save_fig_plotnine(p, f'model_performance_{v}.png')



def main():
    mines_p = r'data\int\D_train_tetst_df\features_all_mines.gpkg'
    tailings_p = r'data\int\D_train_tetst_df\tailings.gpkg'

    #world_bounds_p = r'data\world_bound\world-administrative-boundaries.shp'

    #wb = gpd.read_file(world_bounds_p)


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

    units = {'Active_years':'years', 'Concentrate_production':'t', 'Tailings_production': 't',
        'Polygon_count': 'count', 'Area_mine': 'km2', 'Area_mine_weighted':'km2',
        'Convex_hull_area': 'km2', 'Convex_hull_area_weighted': 'km2',
            'Convex_hull_perimeter': 'km', 'Convex_hull_perimeter_weighted': 'km',
            'Compactness': 'ratio', 'Compactness_weighted': 'ratio', 'Weight': 'ratio',
            'Waste_rock_production': 't', 'Ore_processed_mass': 't', 'Count': 'count', 'EPS_mean':'score', 'EPS_slope':'score' }
    
    tailings = gpd.read_file(tailings_p)
    tail_imp = immpute_vars(tailings, cat_vars, num_vars)

    res = hype_loop(tail_imp)

    plot_results(res)



    pass

if __name__ == '__main__':
    main()  



