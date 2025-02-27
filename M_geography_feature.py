
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import silhouette_score
from plotnine import *

from util import save_fig, save_fig_plotnine, get_world_bounds, df_to_csv_int
from E_ml_explo import get_data, clean_and_imput

############################################Purpose############################################


############################################Params#############################################

n_cluster_target ={'Ore_processed_mass': 6, 'Concentrate_production': 6, 'Tailings_production': 5}

crs = 'EPSG:6933'

############################################Functions#############################################


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
            return [f"Cluster_{i}_similarity" for i in range(self.n_clusters)]
        
        def fit_predict(self, X, y=None, sample_weight=None):
            self.fit(X, y, sample_weight)
            return self.kmeans_.labels_

def geo_sil(make_plot = False):
    
    df = get_data()

    res = []

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = df[df.Target_var == name]
        t = clean_and_imput(t)

        # Create a new feature for the geographical similarity and cluster
        coords = pd.DataFrame([t.Prop_id, t.Latitude, t.Longitude]).T

        coords.set_index('Prop_id', inplace=True)

        
        for n in range(2, 20):

            cluster_sim = ClusterSimilarity(n_clusters=n, gamma=1.0 / (2 * coords.var().min()))

            labels = cluster_sim.fit_predict(coords)
            
            score = silhouette_score(cluster_sim.transform(coords), labels)

            res.append({
                'Target_var': name,
                'n_clusters': n, 
                'Silhouette_score': score})             

    res = pd.DataFrame(res)
        
    if make_plot:
        p = (ggplot(res, aes(x='n_clusters', y='Silhouette_score')) 
        + geom_line() 
        + geom_point() 
        + facet_wrap('~Target_var', scales='free_y', ncol=1)
        + theme_minimal()
        + labs(y='Silhouette score', x='Number of clusters')
        )


        save_fig_plotnine(p, 'geo_sil.png')

    pass

def get_geom_feat(make_plot = False):
    
    d = get_data()

    res = []

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t= d[d.Target_var == name]
        t = clean_and_imput(t)

        coords = pd.DataFrame([t.Prop_id.astype(str), t.Longitude, t.Latitude]).T

        coords.set_index('Prop_id', inplace=True)

        cluster_sim = ClusterSimilarity(n_clusters=n_cluster_target[name], gamma=1.0 / (2 * coords.var().min()))

        similarity = cluster_sim.fit_transform(coords)

        similarity = pd.DataFrame(similarity, index=coords.index, columns=cluster_sim.get_feature_names_out())

        similarity['Target_var'] = name

        res.append(similarity)

        if make_plot == True:
            coords = coords.merge(similarity, left_index=True, right_index=True)

            coords['Max_similarity'] = coords.drop(columns=['Latitude', 'Longitude', 'Target_var']).max(axis=1)
            
            wb = get_world_bounds(crs = crs)
            f, ax = plt.subplots(1, 1, figsize=(14, 7))
            # plot cluster centers and max similarity
            sns.scatterplot(x=coords.Longitude, y=coords.Latitude, data=coords, hue='Max_similarity', ax=ax, palette='viridis')
            sns.scatterplot(x= cluster_sim.kmeans_.cluster_centers_[:, 0], y= cluster_sim.kmeans_.cluster_centers_[:, 1], c='#b30000', marker='P',  s=300, ax = ax, label='Cluster centers')
            wb.boundary.plot(ax=ax, color='black', linewidth=.5, label = 'World boundary')

            # include legends of all layers
            ax.legend(loc='center left', bbox_to_anchor=(0, 0.5), title='Max similarity')

            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            plt.tight_layout()

            save_fig(f'{name}_geo_sim.png')
                
    
    res = pd.concat(res, axis=0)

    res.reset_index(inplace=True)

    df_to_csv_int(res, 'geo_sim.csv')
        

def calc_geom_X_pred(p = r'data\int\D_ml_sample\X_pred_set.csv'):
    
    d = pd.read_csv(p)

    res = []

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        coords = pd.DataFrame([d.id_data_source.astype(str), d.Longitude, d.Latitude]).T

        coords.set_index('id_data_source', inplace=True)

        cluster_sim = ClusterSimilarity(n_clusters=n_cluster_target[name], gamma=1.0 / (2 * coords.var().min()))

        similarity = cluster_sim.fit_transform(coords)

        similarity = pd.DataFrame(similarity, index=coords.index, columns=cluster_sim.get_feature_names_out())

        similarity['Target_var'] = name

        res.append(similarity)

        
    res = pd.concat(res, axis=0)

    res.reset_index(inplace=True)

    df_to_csv_int(res, 'geo_sim_X_pred.csv')

############################################Main#############################################

if __name__ == '__main__':
    calc_geom_X_pred()