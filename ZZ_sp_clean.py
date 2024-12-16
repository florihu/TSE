
from D_sp_data_clean import get_data, var_exp_path
import pandas as pd
import numpy as np
import seaborn as sns

from util import df_to_latex


def mine_ids_per_commodity(df, target_commodities):
    """
    This function identifies the IDs of mines that contain one or more of the target commodities, 
    and the unique commodities mined along with them.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'PROP_ID' and 'Commodities_list'.
        target_commodities (set): Set of target commodities to filter for.

    Returns:
        dict: Dictionary of sets containing the IDs of mines that contain each target commodity.
        list: Set of unique IDs of mines that contain at least one target commodity.
        list: Set of unique commodities mined along with the target commodities.
    """
    # Ensure target_commodities is a set for efficient lookups
    target_commodities = set(target_commodities)

    ids_per_target = {}
    unique_ids = set()
    unique_coms = set()

    for com in target_commodities:
        # not na commodities
        df = df[df['Commodities_list'].notna()]
        # Extract IDs for each target commodity, ensuring uniqueness
        ids_per_target[com] = set(df[df['Commodities_list'].str.contains(com)]['PROP_ID'])
        unique_ids.update(ids_per_target[com])

        # Extract commodities mined alongside the target commodity
        mined_commodities = df[df['Commodities_list'].str.contains(com)]['Commodities_list']
        for commodity_list in mined_commodities:
            unique_coms.update([commodity.strip() for commodity in commodity_list.split(',')])
    
    return ids_per_target, list(unique_ids), list(unique_coms)


def data_coverage(unique_ids, levels = ['site_temp', 'site_com', 'site_temp_com']):
    level_id_dict = {'site_temp': ['PROP_ID'], 'site_com': ['PROP_ID', 'Commodity'], 'site_temp_com': ['PROP_ID', 'Commodity']}
    for l in levels: 
        df = get_data(l)
        subset = df[df[level_id_dict[l]].isin(unique_ids)]
        # per id how much non nan years of ore production
        counts_per_id = subset.groupby(level_id_dict[l]).count()

        # write to latex
        df_to_latex(counts_per_id.describe(), f'{l}_coverage')

    return counts_per_id



# Define target commodities as a set
target_commodities = {'Copper', 'Zinc', 'Nickel'}

if __name__ == '__main__':

    site = get_data('site')  # Assuming get_data is a function that loads your DataFrame
    
    # Pass target_commodities as an argument to the function
    ids_per_target, unique_ids, unique_coms = mine_ids_per_commodity(site, target_commodities)
    
    
    data_coverage(unique_ids)
    
