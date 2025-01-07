
import geopandas as gpd
import pandas as pd

from scipy.spatial import ConvexHull

from plotnine import *
import numpy as np
from util import save_fig_plotnine, data_to_csv_int, df_to_gpkg, save_fig
from tqdm import tqdm
import warnings
from multiprocessing import Pool
from functools import partial
import logging
import matplotlib.pyplot as plt
from rtree import index



log = logging.getLogger(__name__)

def calc_spat_par(group, df):
        un = group.union_all()
        
        if un.is_empty:
            return pd.Series({'Polygon_count': 0, 'Weight':0 , 'Area_mine': 0, 'Area_mine_weighted': 0, 'Convex_hull_area': 0,'Convex_hull_area_weighted': 0, 'Convex_hull_perimeter': 0, 'Convex_hull_perimeter_weighted': 0,'Compactness': 0, 'Compactness_weighted': 0, 'geometry': None})
        
        
        area = un.area / 10**6 # convert to km2
        
        # unique poly ids
        unique_ids = group['id'].unique().tolist()

        # how many uniqe mines are allocated ot polygons
        unique_mines = df[df['id'].isin(unique_ids)]['id_data_source'].unique()

        assert len(unique_mines) > 0, 'There are no unique ids in the group'
        # weight Assumption: the unary union union of the polygons is equally distributed among the unique mines
        # Overlaps are therefore equally not distributed among the mines


        w = 1 / len(unique_mines)
        
        # number of unique mines per unary union

        conv = un.convex_hull
        conv_area = conv.area / 10**6   
        conv_per = conv.length / 10**3
        
        w_area = area * w
        conv_area_w = conv_area * w
        conv_per_w = conv_per * w

        comp_w = w_area / conv_area_w
        comp = area / conv_area

        geom = un.centroid

        col = pd.Series({'Polygon_count': len(group), 'Weight':w , 'Area_mine': area, 'Area_mine_weighted': w_area, 'Convex_hull_area': conv_area,'Convex_hull_area_weighted': conv_area_w, 'Convex_hull_perimeter': conv_per,'Convex_hull_perimeter_weighted': conv_per_w ,'Compactness': comp, 'Compactness_weighted': comp_w, 'geometry': geom})
        
        return col

tqdm.pandas()

def alloc_cluster_to_mine(df):
    # Use tqdm for groupby apply
    area_per_mine = df.groupby('id_data_source').progress_apply(
        lambda group: calc_spat_par(group, df)
    ).reset_index()

    # Convert to GeoDataFrame
    area_per_mine = gpd.GeoDataFrame(area_per_mine, geometry='geometry', crs='EPSG:6933')
    
    # Save to GeoPackage
    df_to_gpkg(area_per_mine, 'allocated_area_union_geom', crs='EPSG:6933')
    
    return area_per_mine

def alloc_poly_to_sp():
    merge_n = read_and_merge()

    alloc = alloc_cluster_to_mine(merge_n)
    
    log.info('Data saved to allocated_area_coms.gpkg')

    return alloc


def check_overlap(gdf):
    # Create spatial index
    idx = index.Index()
    for i, poly in enumerate(gdf.geometry):
        idx.insert(i, poly.bounds)
    
    # Check for overlaps
    for i, poly in enumerate(gdf.geometry):
        potential_matches = list(idx.intersection(poly.bounds))
        potential_matches = [j for j in potential_matches if j > i]  # Avoid duplicate checks
        
        for j in potential_matches:
            if poly.intersects(gdf.geometry.iloc[j]):
                return True  # Stop at first overlap found
    
    return False  # No overlaps found

def read_and_merge(target_crs = 'EPSG:6933'):
    polys = gpd.read_file(r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg') 
    polys.to_crs(target_crs, inplace=True)

    #assert not check_overlap(polys), 'There are overlapping polygons' #-> There are overlaps in the polygons

    # check if the poly id are unique
    assert polys['id'].nunique() == len(polys), 'The polygon ids are not unique'


    polys_ids = pd.read_csv(r'data\dcrm_cluster_data\dcrm_cluster_data\cluster_points_concordance.csv')
    merge = polys.merge(polys_ids, on='id_cluster', how='left', suffixes=('', '_conc'))
    merge_n = merge[~merge['id_data_source'].isna()]

    # Dup check 
    cluster_sp_pairs = merge_n[['id_cluster', 'id_data_source']].drop_duplicates()
  
   # Check if there are duplicate data sources for the same cluster
    duplicate_sources_per_mine = cluster_sp_pairs.groupby('id_cluster')['id_data_source'].nunique() > 1
    
    if duplicate_sources_per_mine.any():
        count = duplicate_sources_per_mine.sum()
        warnings.warn(f'There are many mines per cluster for {count / len(duplicate_sources_per_mine)} of cluster', UserWarning)

    # Check if there are duplicate data sources for the same cluster
    duplicate_clusters_per_source = cluster_sp_pairs.groupby('id_data_source')['id_cluster'].nunique() > 1
    if duplicate_clusters_per_source.any():
        count = duplicate_clusters_per_source.sum()
        warnings.warn(f'There are many clusters per mine for {count / len(duplicate_clusters_per_source)} of mines', UserWarning) 


    return merge_n

def explo_land_use(land_use):
    '''
    Explore the land use data
    '''
    plot1 = (ggplot(land_use, aes(x= 'Area_mine')) 
            + geom_histogram(aes(y='..density..'), bins=50, alpha=0.6)
            +  geom_density(color='#66c2a5', size=1)
            + labs(x='Land use log(km2)', y='Density')
            + theme_minimal()
            + scale_x_log10()
    )

    save_fig_plotnine(plot1, 'land_use_hist.png')
    print(plot1)

    plot2 = (ggplot(land_use, aes(x= 'Area_mine', y='Count')) 
            + geom_point()
            + labs(x='Land use (log(km2))', y='Polygons per mine (log)')
            + theme_minimal()
            + geom_smooth(method='lm', color='red')
            + scale_x_log10()
            + scale_y_log10()
    )
    
    save_fig_plotnine(plot2, 'land_use_scatter.png')
    print(plot2)

    plot3 = (ggplot(land_use, aes( x= 'Count') )
            + geom_histogram(bins = 50, alpha= .6)
            + labs(x='Polygons per mine')
            + theme_minimal()
    )
    save_fig_plotnine(plot3, 'land_use_count_hist.png')
    print(plot3)





if __name__ == '__main__':
    allocated_polys = alloc_poly_to_sp()
