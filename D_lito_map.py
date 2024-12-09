
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


def spat_join(li_path, target_crs = 'EPSG:6933'):
    li = pyo.read_dataframe(li_path)
    mines = read_and_merge()
    mines = mines.to_crs(target_crs)
    li = li.to_crs(target_crs)
    joined = gpd.sjoin(li, mines, predicate='intersects')
    return joined

def li_class_to_columns(join):
    join.rename(columns={'xx': 'Li_class'}, inplace=True)
    # Area weighted class distribution
    by_mine = join.groupby(['id_data_source', 'Li_class'])['area_mine'].sum().unstack().fillna(0) 

    # Normalize by mine
    by_mine = by_mine.div(by_mine.sum(axis=1), axis=0)    
    return by_mine

def main():
    li_path = 'data\LiMW_GIS 2015.gdb\LiMW_GIS 2015.gdb'
    join = spat_join(li_path)

    by_mine = li_class_to_columns(join)

    data_to_csv_int(by_mine, 'li_class_by_mine')

    return None


if __name__ == '__main__':
    main()
    