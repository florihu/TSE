import os
import pandas as pd
import numpy as np
from util import get_path, append_to_excel
from D_sp_data_clean import dtype_conversion, var_exp_path
from tqdm import tqdm
from fuzzywuzzy import process

def load_prod(path):
    '''
    Load production data from an Excel file with unknown sheet names.
    
    Parameters:
        path (str): The path to the Excel file.
        
    Returns:
        DataFrame: A dictionary of DataFrames, each corresponding to a sheet.
    '''
    # Load the sheet names
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    
    area = pd.read_excel(path, sheet_name=sheet_names[0], skiprows=7, usecols='A:AA', header = None)
    prod = pd.read_excel(path, sheet_name=sheet_names[1], skiprows=1, usecols='A:AX')
    
    return area, prod

def clean_area(area_df, commodities):
    """
    Clean the area DataFrame by dropping rows if the commodity in the first column
    matches any in the specified commodities list for the current row and the next three rows.

    Parameters:
        area_df (DataFrame): The DataFrame containing area data.
        commodities (list): A list of commodities to check against.

    Returns:
        DataFrame: The cleaned DataFrame with specified rows dropped.
    """
    # Create a mask to keep rows that do not match the condition
    mask = pd.Series([True] * len(area_df))

    column_names = area_df.iloc[1, :].values  # Get the column names

    area_df = pd.DataFrame(area_df.values, columns=column_names)  # Rename the columns

    # Iterate through the DataFrame
    for index in range(len(area_df) - 3):  # Ensure we don't go out of bounds
        # Check the current and the next three rows
        value = area_df.iloc[index, 0]  # First column values
        
        # If any of the current_rows match the commodities list
        if value in commodities:
            # Mark these rows for dropping
            mask.iloc[index:index + 4] = False  # Drop current and next three rows

    masked_area = area_df[mask]  # Rename the columns

    assert masked_area.shape[0] == area_df.shape[0] - 4 * len(commodities), "Rows not dropped correctly"

    # Drop columns with no name
    area_dropped = masked_area.dropna(axis=1, how='all')
    area_dropped['Complete'] = area_dropped['Complete'].map(lambda x: x == 'Y')


    # Columns conversion
    lookup_site = werner_lookup[werner_lookup['Dependency'] == 'site']
    site_temp_cols = lookup_site['Var_original'].values
    rename_vars = dict(zip(lookup_site['Var_original'], lookup_site['Var_trans']))
    dtype_dict = dict(zip(lookup_site['Var_trans'], lookup_site['Dtype']))

    site_temp = area_dropped[area_dropped.columns.intersection(site_temp_cols)]
    renamed = site_temp.rename(columns=rename_vars)
    converted = dtype_conversion(renamed, type_dict=dtype_dict)

    return converted

def clean_production(prod_data):
    '''
    Clean production data by dropping rows with specific column values and renaming columns.

    Parameters:
        prod_data (DataFrame): The production data.

    Returns:
        DataFrame: The cleaned production data.
        #TODO add also the site_temp_com df
    '''
    # Reset index if necessary to ensure correct row access
    prod_data = prod_data.reset_index(drop=True)

    # Drop the second row (index 1)
    dropped = prod_data.drop(index=0, axis=0).dropna(axis=1, how='all').drop(['Other.1', 'Other.2', 'Other.3'], axis=1)

    # Drop all rows with columns with 'Cum. Prod.', 'Std Dev', 'Count' in the second column
    excluded = dropped[~dropped.iloc[:, 1].isin(prod_keys_exclude)]

    # Get lookup tables
    site_temp_cols = werner_lookup[werner_lookup['Dependency'] == 'site_temp']['Var_original'].values
    rename_vars = dict(zip(werner_lookup['Var_original'], werner_lookup['Var_trans']))
    dtype_dict = dict(zip(werner_lookup['Var_trans'], werner_lookup['Dtype']))
    
    site_temp = excluded[excluded.columns.intersection(site_temp_cols)]
    renamed = site_temp.rename(columns=rename_vars)
    converted = dtype_conversion(renamed, type_dict=dtype_dict)

    drop_nan = converted.dropna(axis=0, how='all')
    
    return drop_nan

def save_intermediate_data_to_csv(data, out_name, file_extension='.csv'):
    inter_path = 'data/int'

    out_path = os.path.join(inter_path, out_name + file_extension)

    data.to_csv(out_path)

def merge_werner(on='Prop_name'):
    '''
    Merge the area and production DataFrames on the specified column.

    '''
    area, prod = load_prod(prod_path)
    c_area = clean_area(area, com_study)
    c_prod = clean_production(prod)
    c_area.set_index(on, inplace=True)
    c_prod.set_index(on, inplace=True)
    merged = werner_to_werner_merge(c_prod, c_area)
    return merged


def fuzzy_mapping(area_keys, prod_keys):
    '''
    This function performs a fuzzy matching between two lists of keys.
    
    Parameters:
        wkeys (list): A list of keys from the Werner data.
        spkeys (list): A list of keys from the SP data.
        
    Returns:
        dict: A dictionary containing the Werner keys as keys and the closest SP key as values.
    '''
    
    
    area_col = []
    prod_col = []
    scores = []
    
    for akey in tqdm(area_keys, 'matching keys'):
        match, score = process.extractOne(akey, prod_keys)
        area_col.append(akey)
        prod_col.append(match)
        scores.append(score)

    df = pd.DataFrame({'w_area_key': area_col, 'w_prod_key': prod_col, 'score': scores})

    append_to_excel(r'data\variable_description.xlsx', df, 'fuzzy_werner_a_prod')

    print(np.mean(scores))

    return df

def werner_to_werner_merge(c_prod, c_area):

    match_wtow = pd.read_excel(var_exp_path, sheet_name='fuzzy_werner_a_prod')
    match_sub = match_wtow[match_wtow['score'] >= 90]
    area_m = c_area.merge(match_sub, left_on='Prop_name', right_on='w_area_key', how='left')

    prod_m = c_prod.merge(area_m, left_on='Prop_name', right_on='w_prod_key', how='left')

    prod_m.rename(columns={'w_area_key': 'Prop_name'}, inplace=True)
    
    # 18 Mines could not be matched because the fuzzy matching score was below 90
    return prod_m

'script parameters'
file_name = 'Supplementary Part V - Site Data(1).xlsx'
com_study = ['Cu', 'PbZn', 'PbZnCu', 'Ni', 'PGEs', 'Au', 'Diamonds', 'Uranium']
prod_keys_exclude = ['Cum. Prod.', 'Std Dev', 'Count']
prod_path = get_path(file_name) 

werner_lookup = pd.read_excel(var_exp_path, sheet_name='werner_lookup')
match_wtow = pd.read_excel(var_exp_path, sheet_name='fuzzy_werner_a_prod')

targets = ['Tailings_production', 'Waste_rock_production', 'Concentrate_production', 'Ore_processed_mass']


if __name__ == '__main__':
    merge_werner()


    

    
