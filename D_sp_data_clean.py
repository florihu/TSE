
from util import get_path

import pandas as pd 
import os
import numpy as np
from functools import reduce
import re


def get_data(folder_name):
    """
    Get site data and merges additional entries.

    Parameters:
        folder_name (str): The name of the site folder.
        var_exp_path (str): The path to the variable explanation file.

    Returns:
        DataFrame: The merged or singular site data.
    """
    
    folder_path = get_path(folder_name)

    var_types_df = pd.read_excel(var_exp_path, sheet_name='sp_lookup', )

    var_types = dict(zip(var_types_df['Var_original'], var_types_df['Dtype']))

    rename_vars = dict(zip(var_types_df['Var_original'], var_types_df['Var_trans']))

    collect = []
    # site
    if folder_name == 'site':
        for i, f in enumerate(os.listdir(folder_path)):
            site_data = pd.read_excel(os.path.join(folder_path, f), skiprows=4).dropna(how='all')
            id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]
            id_cols = ['Prop_id']
            site_data_id = site_data.iloc[id_col_index:]
            collect.append(init_conv_nan_removal(site_data_id, var_types, id_cols, rename_vars, first = i))            

        merge = merging_of_dfs(collect, id_cols)
        return merge

    elif folder_name == 'site_com':
        for f in os.listdir(folder_path):
            site_data = pd.read_excel(os.path.join(folder_path, f), skiprows=4).dropna(how='all')

            id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]
            id_cols = ['Prop_id']
            com = f.split('_')[-1].split('.')[0]
            
            site_data = site_data.iloc[1:,id_col_index:]
            site_data['COMMODITY'] = com

            renamed = init_conv_nan_removal(site_data, var_types, id_cols, rename_vars)
            

            collect.append(renamed)

        return concat_if_columns_same(collect)

    elif folder_name == 'site_temp':
        for f in os.listdir(folder_path):
            id_cols = ['Prop_id', 'Year']
            commodity = f.split('.')[0].split('_')[-1]

            site_data = pd.read_excel(os.path.join(folder_path, f), skiprows=4).dropna(how='all')
            id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]
            site_data = col_trans(site_data, id_col_index, commodity)
            if site_data.empty:
                continue
            renamed = init_conv_nan_removal(site_data, var_types, id_cols, rename_vars)
            collect.append(renamed)

        return merging_of_dfs(collect, id_cols)

    elif folder_name == 'site_temp_com':
            conc = []
            for f in os.listdir(folder_path):
                collect = []
                commodity = f.split('/')[-1]
                id_cols = ['Prop_id', 'Year']
                com_path = os.path.join(folder_path, f)
                for f_ in os.listdir(com_path):
                    site_data = pd.read_excel(os.path.join(com_path, f_), skiprows=4).dropna(how='all')
                    id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]
                    site_data = col_trans(site_data, id_col_index)

                    collect.append(init_conv_nan_removal(site_data, var_types, id_cols, rename_vars))

                merged = merging_of_dfs(collect, id_cols)
                merged['Commodity'] = commodity
                conc.append(merged)
            
            return concat_if_columns_same(conc)


   
def init_conv_nan_removal(site_data, var_types, id_cols, rename_vars, first=None): 
    
    if first is not 0:
        site_data = site_data.drop('PROP_NAME', axis=1, errors='ignore')

    dtype_converted = dtype_conversion(site_data, var_types)
    non_nan = nan_remove(dtype_converted)

    for c in non_nan.columns:
        if c in rename_vars.keys():
            non_nan.rename(columns={c: rename_vars[c]}, inplace=True)
        else:
            raise KeyError(f'The column could not be renamed_{c}')
        
    
    non_nan.set_index(id_cols, inplace=True)

    return non_nan

def merging_of_dfs(df_list, id_cols):
    '''
    Merging of DataFrames

    Parameters:
        df_list (list): A list of DataFrames to merge
        id_cols (list): A list of column names to merge on

    Returns:
        DataFrame: The merged DataFrame
    '''
    merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=id_cols), df_list).reset_index()

    non_nan = nan_remove(merged)

    return non_nan.reset_index()

def concat_if_columns_same(df_list):
    '''
    Concatenate DataFrames if columns are the same

    Parameters:
        df_list (list): A list of DataFrames to concatenate

    Returns:
        DataFrame: The concatenated DataFrame
    '''
    if len(df_list) == 1:
        return df_list[0]
    else:
        all_columns = sorted(set().union(*(df.columns for df in df_list)))
        # Reindex each DataFrame to ensure consistent columns and sort them
        conc = [df.reindex(columns=all_columns).sort_index(axis=1) for df in df_list]

        conc_conc = pd.concat(conc, axis=0)
        return conc_conc.reset_index()

def col_trans(site_data, id_col_index, commodity=None):
    # Assume the third column as the variable name
    var_name = site_data.columns[id_col_index + 1]

    if commodity and any([i in var_name.split('_') for i in vars_spec_handle]):
        var_name = f'{var_name}_{com_dict[commodity]}'
    # Set headers based on the first row and adjust columns
    site_data.columns = site_data.iloc[0]
    site_data = site_data.iloc[2:, id_col_index:]
    site_data.rename(columns={site_data.columns[0]: 'PROP_ID'}, inplace=True)
    site_data = site_data.dropna(subset=['PROP_ID'])

    # Stack data with 'PROP_ID' and 'YEAR' as indexes, renaming the variable column
    stacked = site_data.set_index('PROP_ID').stack()

    stacked.index.set_names(['PROP_ID', 'YEAR'], inplace=True)

    stacked_df = stacked.reset_index().rename({0: var_name}, axis=1)

    # Clean 'YEAR' column by removing 'Y' prefix
    stacked_df['YEAR'] = stacked_df['YEAR'].str.replace('Y', '', regex=False).astype(int)

    return stacked_df

def nan_remove(df):
    '''
    Drop rows and columns with all NaN values

    Parameters:
        df (DataFrame): The DataFrame to clean

    Returns:
        DataFrame: The cleaned DataFrame
    '''
    return df.dropna(how='all').dropna(axis=1, how='all')

def dtype_conversion(df, type_dict):
    '''
    Convert the columns of a DataFrame to the specified data types

    Parameters:
        df (DataFrame): The DataFrame to convert
        type_dict (dict): A dictionary of column names and their corresponding data types
    
    Returns:
        DataFrame: The converted DataFrame
    '''
    for c in df.columns:

        dtype = type_dict.get(c, None)

        assert dtype is not None, f'The data type could not be found for column: {c}'

        if dtype == 'date':
            to_str = df[c].astype('str')
            df[c] = pd.to_datetime(to_str, errors='coerce')

        elif dtype == 'int':
            df[c]= pd.to_numeric(df[c], errors='coerce').astype('Int64')
        elif dtype == 'float':
            df[c] = pd.to_numeric(df[c], errors='coerce')
        elif dtype == 'str':
            df[c] = df[c].astype('str')
        elif dtype == 'bool':
            df[c] = df[c].astype('bool')

    return df


vars_spec_handle = ['MATTE', 'BULK']

com_dict = {'cu': 'COPPER','ni': 'NICKEL', 'zn': 'ZINC'}
# script parameters
var_exp_path = get_path('variable_description.xlsx')

if __name__ == '__main__':
    site_folder_path = get_data('site_temp')
