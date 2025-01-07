
import geopandas as gpd
import pandas as pd
import pyogrio as pyo

from D_land_map import read_and_merge
import concurrent.futures
from tqdm import tqdm
from util import data_to_csv_int

def unary_union_for_group(group):

    """Helper function to perform unary_union for each group."""
    # Ensure 'group' is a GeoSeries
    if isinstance(group, gpd.GeoSeries):
        return group.unary_union
    else:
        return None  # Return None or handle as necessary if not a valid group

def unary_union_per_mine_parallel():
    
    # Read and merge the data
    merge_n = read_and_merge()

    # Group by 'id_data_source'
    grouped = merge_n.groupby('id_data_source')['geometry']

    # Create a ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Apply 'unary_union' for each group in parallel
        result = list(executor.map(unary_union_for_group, [group for _, group in grouped]))

    # Create a new GeoDataFrame to store the result
    result_gdf = gpd.GeoDataFrame({
        'id_data_source': grouped.groups.keys(),
        'geometry': result
    })
    
    return result_gdf

def spat_join(li_path, m_path,  target_crs = 'EPSG:6933'):
    li = pyo.read_dataframe(li_path)
    mines = gpd.read_file(m_path)

    mines = mines.to_crs(target_crs)
    li = li.to_crs(target_crs)

    joined = gpd.sjoin(li, mines, predicate='intersects')
    
    return joined

def li_class_to_columns(join):
    join.rename(columns={'xx': 'Li_class'}, inplace=True)
    
    # Generate dummy variables for 'Li_class'
    dummies = pd.get_dummies(join['Li_class']).astype(int)
        
    # Combine the dummy variables with the original data
    join = pd.concat([join['id_data_source'], dummies], axis=1)
    
    return join

def main():
    li_path = 'data\LiMW_GIS 2015.gdb\LiMW_GIS 2015.gdb'
    m_path =  r'data\int\D_land_map\allocated_area_union_geom.gpkg'
    
    join = spat_join(li_path, m_path)
    

    j_coltrans = li_class_to_columns(join)

    data_to_csv_int(j_coltrans, 'li_class_by_mine')

    return None


if __name__ == '__main__':
    main()
    