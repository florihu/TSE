'''
Goal is the SP Werner merge plus the identification of the waste rock sample and the tailings sample.
'''
import pandas as pd
from D_sp_data_clean import get_data, var_exp_path
from util import df_to_latex, data_to_csv_int
import numpy as np

from M_prod_model import prep_data
from D_load_werner import merge_werner



def merge_sp(df1, df2, on='Prop_id'):
    df1.set_index(on, inplace=True)
    df2.set_index(on, inplace=True)
    return pd.merge(df1, df2, on=on).reset_index()

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
        ids_per_target[com] = set(df[df['Commodities_list'].str.contains(com)]['Prop_id'])
        unique_ids.update(ids_per_target[com])

        # Extract commodities mined alongside the target commodity
        mined_commodities = df[df['Commodities_list'].str.contains(com)]['Commodities_list']
        for commodity_list in mined_commodities:
            unique_coms.update([commodity.strip() for commodity in commodity_list.split(',')])

        unique_ids_int = [int(i) for i in unique_ids]
    
    return ids_per_target, unique_ids_int, list(unique_coms)


def merge_werner_conc(df_werner, conc):
    """
    Merge the Werner data with the concentrate table.

    Parameters:
        df_werner (pd.DataFrame): DataFrame containing the Werner data.
        conc (pd.DataFrame): DataFrame containing the concentrate data.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    conc = conc[conc['Flag_min_distance']]
    merge = df_werner.merge(conc, left_on='Prop_name', right_on='werner_key', how='left')
    merge = merge[(merge['Prop_id'].notna()) & (merge['Year'].notna())]
    merge['Prop_id'] = merge['Prop_id'].astype(int)

    starting_year_operation = merge.groupby('Prop_id')['Year'].min().reset_index()
    merge = merge.merge(starting_year_operation, on='Prop_id', suffixes=('', '_start'))
    merge.rename(columns={'Year_start': 'Start_up_year'}, inplace=True)
    
    return merge


def unique_werner_sp_keys(df_sp, df_werner, id_cols=['Prop_id', 'Year']):
    """
    Get unique ID-year pairs from both SP and Werner datasets.
    
    Parameters:
        df_sp (pd.DataFrame): Site data.
        df_werner (pd.DataFrame): Werner data.
        id_cols (list): Columns identifying unique records.

    Returns:
        pd.DataFrame: DataFrame containing all unique ID-year pairs.
    """
    # Combine unique keys from both datasets
    conc = pd.concat([
        df_sp[id_cols],
        df_werner[id_cols]
    ]).drop_duplicates().reset_index(drop=True)

    return conc

def calc_sp(df_sp, conc_calc_cols):
    """
    Calculate derived columns for the SP dataset.
    
    Parameters:
        df_sp (pd.DataFrame): SP dataset.
        conc_calc_cols (list): Columns contributing to concentrate production.

    Returns:
        pd.DataFrame: Updated SP DataFrame with calculated columns.
    """
    # Calculate derived columns
    df_sp = df_sp.copy()
    df_sp['Waste_rock_production'] = df_sp['Ore_processed_mass'] * df_sp['Stripping_ratio']
    df_sp['Concentrate_production'] = df_sp[conc_calc_cols].sum(axis=1) * 1e3  # Convert from kt to t
     # If concentrate productio = 0 replace wiht nan
    df_sp['Concentrate_production'] = df_sp['Concentrate_production'].replace(0, np.nan)

    # only if concentrate production >0 ore - concentrate
    df_sp['Tailings_production'] = np.where((df_sp['Concentrate_production'] > 0) & (df_sp['Ore_processed_mass'] > 0), df_sp['Ore_processed_mass'] - df_sp['Concentrate_production'], np.nan)

    # Drop inconsistent records - 45 records include nans
    df_sp[df_sp['Tailings_production'] < 0] = np.nan

    return df_sp


def calc_werner(df_werner, cc = ['Cu_concentrate_production', 'Mo_concentrate_production', 'Pb_concentrate_production', 'Zn_concentrate_production']):
    df_werner['Concentrate_production'] = df_werner[cc].sum(axis=1)

    # If concentrate productio = 0 replace wiht nan
    df_werner['Concentrate_production'] = df_werner['Concentrate_production'].replace(0, np.nan)

    return df_werner

def prio_source(df_sp, df_werner, target_vars):
    """
    Assign priority-based values and their sources for target variables.
    
    Parameters:
        df_sp (pd.DataFrame): SP dataset with calculated columns.
        df_werner (pd.DataFrame): Werner dataset.
        all_keys (pd.DataFrame): Unique ID-year pairs.
        target_vars (list): Variables to assign priority-based values for.

    Returns:
        pd.DataFrame: Updated `all_keys` with assigned values and sources.
    """

    df_sp = calc_sp(df_sp, conc_calc_cols)
    df_werner = merge_werner_conc(df_werner, conc)
    df_werner = calc_werner(df_werner)
    all_keys = unique_werner_sp_keys(df_sp, df_werner)

    df_werner.drop_duplicates(subset=['Prop_id', 'Year'], inplace=True)
    df_sp.drop_duplicates(subset=['Prop_id', 'Year'], inplace=True)

    # Merge SP and Werner data with all_keys
    sp_merged = all_keys.merge(df_sp, on=['Prop_id', 'Year'], how='left', suffixes=('', '_sp'))
    werner_merged = all_keys.merge(df_werner, on=['Prop_id', 'Year'], how='left', suffixes=('', '_werner'))

    # assert the target cols are in the dfs
    assert all(col in sp_merged.columns for col in target_vars), 'Target vars not in SP df'
    assert all(col in werner_merged.columns for col in target_vars), 'Target vars not in Werner df'

    # Initialize result DataFrame
    result = all_keys.copy()

    # Assign values and sources
    for var in target_vars:
        # Assign SP values when available
        result[var] = sp_merged[var].where(sp_merged[var].notna(), werner_merged[var])
        result[f'{var}_source'] = np.where(
                sp_merged[var].notna() & (sp_merged[var] != 0), 'SP',
                np.where(
                    werner_merged[var].notna() & (werner_merged[var] != 0), 'WERNER',
                    np.nan
                )
        )               
    
    # drop columns that have ore or waste na 
    result = result[((~result['Ore_processed_mass'].isna() & ~result['Waste_rock_production'].isna()) | 
                    (~result['Concentrate_production'].isna() & ~result['Tailings_production'].isna()) &
                    (~result['Start_up_year'].isna()))]

    data_to_csv_int(result, 'target_vars_prio_source')

    return result




lookup = pd.read_excel(var_exp_path, sheet_name='sp_lookup')
waste_vars = lookup[lookup['Calc_waste_rock']].Var_trans
tail_vars = lookup[lookup['Calc_tailings']].Var_trans




# Script params
conc = pd.read_excel(var_exp_path, sheet_name='fuzzy_werner_sp')
site_temp = get_data('site_temp')
site = get_data('site')

unique_sp_ids = mine_ids_per_commodity(site, {'Copper', 'Zinc', 'Nickel'})[1]


df_werner = merge_werner()
df_sp = merge_sp(site_temp, site, on='Prop_id')



target_vars=['Ore_processed_mass', 'Waste_rock_production', 'Concentrate_production', 'Tailings_production', 'Start_up_year']

conc_calc_cols = ['Copper_concentrate_production', 'Nickel_concentrate_production',
       'Zinc_concentrate_production',  'Nico_powder_production',
       'Mo_concentrate_production', 'Lead_concentrate_production',
       'Ferronickel_concentrate_production', 'Pgm_concentrate_production',
       'Co_concentrate_production', 'Co_powder_production',
       'Bulk_copper_concentrate_production',
       'Bulk_zinc_concentrate_production', 'Matte_nickel_production']

if __name__ == '__main__':
    prio_source(df_sp, df_werner, target_vars)

    
    

