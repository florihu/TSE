import geopandas as gpd
import pandas as pd

from scipy.spatial import ConvexHull

from plotnine import *
import numpy as np
from util import save_fig_plotnine, data_to_csv_int
from tqdm import tqdm

'''
This script is used to allocate land use data to the mining sites. The land use data is in the form of a shapefile containing polygons of land use types. The mining sites are in the form of points. The script will allocate the land use type to each mining site based on the land use polygon that contains the site.

Steps: 
1. spatial join of land use polygons and mining sites for multiple buffer distances
2. calculate the percentage of area that is allocated to only a single mine and the percentage of area that is allocated to multiple mines
3. Identify the pareto efficient buffer distance based on the above percentages

'''

def buffer_series(polys, mines, buffer = [i for i in range(0, 5001, 50)]):
    '''
    Single allocation percentage (SAP) = Percentage of allocated mines that are allocated to only one polygon per accuracy level
    Allocated mines (ALOC) = Percentage of mines that are allocated to a polygon per accuracy level
    '''

    buffers = []
    allocs = []
    allocs_m = []
    saps = []
    saps_m = []
    accuracy = []


    for b in tqdm(buffer, desc='Buffer Series'):

        subset = mines.copy()
        subset['geometry'] = subset.buffer(b)

        j = gpd.sjoin(polys, subset, predicate='intersects')
        
        for a in j.COORDINATE_ACCURACY.unique():
            j_sub = j.loc[j['COORDINATE_ACCURACY'] == a]
    
            sap = j_sub.groupby('PROP_ID').size().eq(1).sum() / len(j_sub['PROP_ID'].unique())
            alloc = j_sub['PROP_ID'].nunique() / len(mines[mines['COORDINATE_ACCURACY'] == a])

            alloc_m = j_sub['PROP_ID'].nunique() / len(mines)
            sap_m = j_sub.groupby('PROP_ID').size().eq(1).sum() / len(mines)

            accuracy.append(a)
            buffers.append(b)
            allocs.append(alloc)
            saps.append(sap)
            allocs_m.append(alloc_m)
            saps_m.append(sap_m)
        

    res = pd.DataFrame({'Buffer': buffers, 'Sap': saps,  'Alloc': allocs, 'Accuracy': accuracy, 'Alloc_mine_norm': allocs_m, 'Sap_mine_norm': saps_m})

    data_to_csv_int(res, 'buffer_series')

    return None

def pareto_optimization(res):
    '''
    Identify the sap, map, allo pair where an increase in allo leads to a drease in sap
    '''
    # Get the convex hull of the data
    hull = ConvexHull(res[['Sap', 'Allo']])
    # Get the indices of the hull
    hull_indices = hull.vertices
    # Get the pareto optimal points
    pareto_optimal = res.iloc[hull_indices]
    return pareto_optimal


def plot_decision_criteria(res):
    '''
    Plot decision criteria with buffer on the x-axis and dependent variables as lines.
    '''
    # Filter and preprocess data
    res = res[res['Buffer'] > 0]
    res['Alloc'] = res['Alloc'] * 100
    res['Sap'] = res['Sap'] * 100
    res['Alloc_mine_norm'] = res['Alloc_mine_norm'] * 100
    res['Sap_mine_norm'] = res['Sap_mine_norm'] * 100

    # Reshape the data for multiple lines
    res_melted = res.melt(id_vars=['Buffer', 'Accuracy'], 
                          value_vars=['Alloc', 'Sap'], 
                          var_name='Variable', 
                          value_name='Value'
                          )

    res_g = res.groupby('Buffer')[['Alloc_mine_norm', 'Sap_mine_norm']].sum().reset_index()

    res_g_melt = res_g.melt(id_vars=['Buffer'], 
                          value_vars=['Alloc_mine_norm', 'Sap_mine_norm'], 
                          var_name='Variable', 
                          value_name='Value'
                          )

    # Find where Sap curves hit the 75% mark
    sap_75 = (
        res_melted[(res_melted['Variable'] == 'Sap') & (res_melted['Value'] <= 75)]
        .groupby('Accuracy')
        .first()
        .reset_index()
    )

    plot = (
        ggplot(res_melted, aes(x='Buffer', y='Value', color='Variable', shape='Accuracy')) +
        geom_point(size=2) +
        geom_vline(data=sap_75, mapping=aes(xintercept='Buffer'), color='black', linetype='dashed') +
        geom_hline(yintercept=75, color='black', linetype='dotted') +
        labs(
            x='Buffer Distance (m)',
            y='Percentage (%)',
            color='Variable',
            shape='Accuracy'  # Legend title for line categories
        ) +
        theme_minimal()  # Use a clean theme, or choose another
    )

    # Save and display the plot
    save_fig_plotnine(plot, 'buffer_criteria_lines_per_acc.png')
    print(plot)
    

    plot2 = (
        ggplot(res_g_melt, aes(x='Buffer', y='Value', color= 'Variable')) +  # Map Variable to color
        geom_point(size = 2) +  # Add line plot with thicker lines
        labs(
            x='Buffer Distance (m)',
            y='Percentage (%)',
            color='Variable'
        ) +
        theme_minimal()  # Use a clean theme, or choose another
    
    )

    # Save and display the plot
    save_fig_plotnine(plot2, 'buffer_criteria_lines_per_mines.png')
    print(plot2)
    
    return None


def mines_to_gdf(mines):
    # drop first two rows
    mines = mines.iloc[2:]
    # drop rows with missing lat and long
    mines = mines.dropna(subset=['LATITUDE', 'LONGITUDE'])

    mines_gdf = gpd.GeoDataFrame(mines, geometry=gpd.points_from_xy(mines['LONGITUDE'], mines['LATITUDE']), crs='EPSG:4326')
    mines_gdf.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)

    # Only 4 have no accuracy level these are dropped
    mines_gdf = mines_gdf[~mines_gdf['COORDINATE_ACCURACY'].isna()]

    # check if mine ids are unique
    assert mines_gdf['PROP_ID'].nunique() == len(mines_gdf), 'Mine IDs are not unique'

    return mines_gdf

def land_use_alloc():
    '''
    Clean data and iterate over buffer distances to allocate land use data to mining sites.

    4. For Area Preservation: EPSG:6933 (World Equal Area)
Description:

    Type: Projected CRS.
    Purpose: Preserves area for global analyses.
    Advantages:
        Accurate area calculations.
        Minimizes distortions for spatial joins involving polygons.
    Limitations:
        Distorts distances and shapes.
    '''
    target_crs = 'EPSG:6933'
    
    mines = pd.read_excel(r'data\s&p\site_alloc\mines_alloc_sp.xls', skiprows=4)
    cmines = mines_to_gdf(mines)
    cmines.to_crs(target_crs, inplace=True)

    polys = gpd.read_file(r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg')
    polys.to_crs(target_crs, inplace=True)

    

    polys_ids = pd.read_csv
    buffer_series(polys, cmines)

    return None


def alloc_poly_to_sp(target_crs = 'EPSG:6933'):
    polys = gpd.read_file(r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg') 
    polys.to_crs(target_crs, inplace=True)
    polys_ids = pd.read_csv(r'data\dcrm_cluster_data\dcrm_cluster_data\cluster_points_concordance.csv')
    merge = polys.merge(polys_ids, on='id_cluster', how='left', suffixes=('', '_conc'))

    merge_n = merge[~merge['id_data_source'].isna()]

    merge_g = merge_n.groupby('id_data_source')['area_mine'].agg(Area_mine='sum', Count='count').reset_index()
    merge_g['Area_mine'] = merge_g['Area_mine'] / 10**6 # convert to km2

    include_source_data = merge_g.merge(polys_ids[['id_data_source', 'data_source']], left_on='id_data_source', right_on='id_data_source', how='left')
    
    return include_source_data

def explo_land_use(land_use):
    '''
    Explore the land use data
    '''
    plot1 = (ggplot(land_use, aes(x= 'Area_mine')) 
            + geom_histogram(bins=50)
            + labs(x='Area (km2)', y='Count')
            + theme_minimal()
            + scale_x_log10()
    )

    save_fig_plotnine(plot1, 'land_use_hist.png')
    print(plot1)

    plot2 = (ggplot(land_use, aes(x= 'Area_mine', y='Polygons per mine')) 
            + geom_point()
            + labs(x='Area (km2)', y='Count')
            + theme_minimal()
            + geom_smooth(method='lm')
    )
    
    save_fig_plotnine(plot2, 'land_use_scatter.png')
    print(plot2)

    plot3 = (ggplot(land_use, aes( x= 'Count') )
            + geom_histogram(bins = 50)
            + labs(x='Count')
            + theme_minimal()
    )
    save_fig_plotnine(plot3, 'land_use_count_hist.png')
    print(plot3)


def ident_buf_distance(res):
    '''
    find the buffer where the sap is 75%
    '''
    sap_75 = res[(res['Sap'] <= .75) & (res['Buffer'] > 0)].groupby('Accuracy').first().reset_index()

    return sap_75['Buffer']

def analysis_land_use_alloc(data_path  = r'data\int\M_land_use_alloc\buffer_series.csv'):
    data = pd.read_csv(data_path)
    #plot_decision_criteria(data)
    ident_buf_distance(data)
    return None

if __name__ == '__main__':
    allocated_polys = alloc_poly_to_sp()

    explo_land_use(allocated_polys)