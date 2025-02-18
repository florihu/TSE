import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from plotnine import *

from E_ml_explo import get_data, clean_and_imput, log_vars, num_vars, cat_vars
from util import save_fig_plotnine, df_to_csv_int



##########################################Purpose############################################

# Implement DBSCAN Outlier detection to remove noise from the data.


##########################################Params#############################################

eps_target = {'Ore_processed_mass': 193779248, 
              'Concentrate_production': 25907499, 
              'Tailings_production': 78138775}

min_samples = 10


##########################################Functions###########################################


def add_geo_to_data():
    d = get_data()

    geo_p = 'data\int\M_geography_feature\geo_sim.csv.csv'

    geo = pd.read_csv(geo_p)
    geo.drop('Unnamed: 0', axis=1, inplace=True)

    m = d.merge(geo, on=['Prop_id', 'Target_var'], how='left')

    # get columns their string contains Cluster
    geo_cols = [col for col in m.columns if 'Cluster' in col]
    
    return m, geo_cols

def dbscan_anomaly_detect():

    d, geo_cols = add_geo_to_data()
     
    vars = num_vars + cat_vars + geo_cols

    res = []
    
    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        scan = DBSCAN(eps=eps_target[name], min_samples=min_samples)
        t = d[d.Target_var == name]
        t = clean_and_imput(t)

        t_var = t[vars]
        t_var.dropna(inplace=True, axis = 1)
      
        scan.fit(t_var)

        anom = pd.DataFrame(scan.labels_, columns=['Anomaly'], index=t.Prop_id)
        anom['Target_var'] = name

        res.append(anom)

        print(name)
        print(anom['Anomaly'].value_counts())

    res_df = pd.concat(res, axis=0)

    df_to_csv_int(res_df, 'anomalies')

    pass

def k_distance_graph():

    d, geo_vars = add_geo_to_data()

    vars = num_vars + cat_vars + geo_vars

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = d[d.Target_var == name]
        t = clean_and_imput(t)

        t = t[vars]

        t.dropna(inplace=True, axis = 1)

        neigh = NearestNeighbors(n_neighbors=min_samples)
        neigh.fit(t)
        distances, indices = neigh.kneighbors(t)

        sort = np.sort(distances, axis=0)

        p = (ggplot() + geom_line(aes(x=range(len(sort)), y=sort[:, 1])) + labs(x='Index', y='Distance') + theme_minimal())

        save_fig_plotnine(p, f'{name}_k_distance_graph')

        # print the point of maximum curvature
        print (name)
        print(sort[np.argmax(np.diff(sort[:, 1])), 1])

if __name__ == '__main__':
    k_distance_graph()