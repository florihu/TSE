'''
Size of training test set for the waste rock and the tailing model


'''
import pandas as pd
from D_sp_data_clean import get_data, var_exp_path
from util import df_to_latex
import numpy as np


def merge_df(df1, df2, on='Prop_id'):
    df1.set_index(on, inplace=True)
    df2.set_index(on, inplace=True)
    return pd.merge(df1, df2, on=on).reset_index()

def waste_rock_sample_sp(df, ids):
    '''
    Assess the sample size for the calculation of waste rock 

    Parameters:
        df (pd.DataFrame): DataFrame containing the site_temp and site data.
        ids (list): List of unique mine IDs.
    
    Returns:
        None
    '''
    ofint = df[df.Prop_id.isin(ids)][waste_vars]

    # drop all where Ore_processed_mass and Production_capacity_t ore strip ratio is nan
    subset = ofint[((ofint['Ore_processed_mass'].notna()) | (ofint['Mill_capacity_tonnes_per_year'].notna())) & (ofint['Stripping_ratio'].notna())]


    def lable_assign(row):
        if  pd.notna(row['Ore_processed_mass']) and row['Ore_processed_mass'] > 0:
            return 'site_temp'
        else:
            return 'site'
        
    subset['Unc_lable'] = subset.apply(lable_assign, axis=1)

    # calculate sample size per lable and total export latex table
    sample_size = subset.groupby('Unc_lable')['Prop_id'].unique().apply(lambda x: len(x))
    sample_size['Total'] = sample_size.sum()

    unwsid = subset.Prop_id.unique()
    
    return sample_size, unwsid


def tailings_sample_sp(df, ids):
    '''
    Assess the sample size for the calculation of waste rock 

    Parameters:
        df (pd.DataFrame): DataFrame containing the site_temp and site data.
        ids (list): List of unique mine IDs.
    
    Returns:
        None
    '''

    # tail vars included in columns of df
    included_vars = [i for i in tail_vars if i in df.columns]

    ofint = df[df.Prop_id.isin(ids)][included_vars]

    ofint['Concentrate_production'] = ofint[conc_calc_cols].sum(axis=1)


    subset = ofint[((ofint['Ore_processed_mass'].notna()) | (ofint['Mill_capacity_tonnes_per_year'].notna())) & (ofint['Concentrate_production'].notna())]

    sub_nonzero = subset[subset['Concentrate_production'] > 0]

    def lable_assign(row):
        if  pd.notna(row['Ore_processed_mass']) and row['Ore_processed_mass'] > 0:
            return 'site_temp'
        else:
            return 'site'
        
    sub_nonzero['Unc_lable'] = sub_nonzero.apply(lable_assign, axis=1)

    # calculate sample size per lable and total export latex table
    sample_size = sub_nonzero.groupby('Unc_lable')['Prop_id'].unique().apply(lambda x: len(x))
    sample_size['Total'] = sample_size.sum()

    unwsid = sub_nonzero.Prop_id.unique()
    
    return sample_size, unwsid

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


lookup = pd.read_excel(var_exp_path, sheet_name='sp_lookup')
waste_vars = lookup[lookup['Calc_waste_rock']].Var_trans
tail_vars = lookup[lookup['Calc_tailings']].Var_trans

conc_calc_cols = ['Copper_concentrate_production', 'Nickel_concentrate_production',
       'Zinc_concentrate_production',  'Nico_powder_production',
       'Mo_concentrate_production', 'Lead_concentrate_production',
       'Ferronickel_concentrate_production', 'Pgm_concentrate_production',
       'Co_concentrate_production', 'Co_powder_production',
       'Bulk_copper_concentrate_production',
       'Bulk_zinc_concentrate_production', 'Matte_nickel_production']

if __name__ == '__main__':
    
    site_temp = get_data('site_temp')
    site = get_data('site')

    unique_ids = mine_ids_per_commodity(site, {'Copper', 'Zinc', 'Nickel'})[1]
    merge = merge_df(site_temp, site, on='Prop_id')

    #swr = waste_rock_sample_sp(merge, unique_ids,)
    tail = tailings_sample_sp(merge, unique_ids)