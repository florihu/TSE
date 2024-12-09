
import geopandas as gpd
import pandas as pd

from scipy.spatial import ConvexHull

from plotnine import *
import numpy as np
from util import save_fig_plotnine, data_to_csv_int
from tqdm import tqdm


def alloc_poly_to_sp():
    merge_n = read_and_merge()

    

    merge_g = merge_n.groupby('id_data_source')['area_mine'].agg(Area_mine='sum', Count='count').reset_index()
    merge_g['Area_mine'] = merge_g['Area_mine'] / 10**6 # convert to km2

    include_source_data = merge_g.merge(polys_ids[['id_data_source', 'data_source']], left_on='id_data_source', right_on='id_data_source', how='left')
    
    return include_source_data


def read_and_merge(target_crs = 'EPSG:6933'):
    polys = gpd.read_file(r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg') 
    polys.to_crs(target_crs, inplace=True)
    polys_ids = pd.read_csv(r'data\dcrm_cluster_data\dcrm_cluster_data\cluster_points_concordance.csv')
    merge = polys.merge(polys_ids, on='id_cluster', how='left', suffixes=('', '_conc'))
    merge_n = merge[~merge['id_data_source'].isna()]
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

    explo_land_use(allocated_polys)