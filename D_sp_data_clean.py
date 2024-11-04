
from util import get_path

import pandas as pd 
import os

def get_site_data(site_name, var_exp_path):
    '''
    Get the site data, merges further additions

    Parameters:
        site_name (str): The name of the site folder
    
    Returns:
        DataFrame: The site data
    '''
    collect = []

    var_types_df = pd.read_excel(var_exp_path, sheet_name='site_keys', usecols='A:B')
    # to dict
    var_types = dict(zip(var_types_df['Var_name'], var_types_df['Dtype']))

    for f in os.listdir(site_name):
        site_data = pd.read_excel(os.path.join(site_name, f), skiprows=4)  

        site_data.dropna(axis=0, how='all', inplace=True)

        dup_removed = duplicate_columns_drop(site_data)

        dtype_converted = dtype_conversion(dup_removed, var_types)

        dtype_converted.set_index('PROP_ID', inplace=True)

        collect.append(dtype_converted)
    
    if len(collect) == 1:
        singular = duplicate_columns_drop(collect[0])
        return singular
    else:
        merged_data = pd.merge(collect, how='outer', on='PROP_ID', axis=1)
        merged_dropped = duplicate_columns_drop(merged_data)
        return merged_dropped

def duplicate_columns_drop(df, suffixes = ['.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']):
    '''
    Identify duplicate columns in a DataFrame

    Parameters:
        df (DataFrame): The DataFrame to check for duplicates
    
    Returns:
        list: A list of duplicate columns
    '''
    suf_cols = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes)]

    dup_cols = df.columns[df.columns.duplicated(keep='first')].tolist()
    
    # Drop identified columns
    df_dropped = df.drop(columns=suf_cols+dup_cols)
    
    return df_dropped
    

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

        if dtype == 'date':
            df[c] = pd.to_datetime(df[c], errors='coerce')
        elif dtype == 'int':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        elif dtype == 'float':
            df[c] = pd.to_numeric(df[c], errors='coerce')
        elif dtype == 'str':
            df[c] = df[c].astype('str')

    return df


    


def merge_site_temp_data():
    '''
    Unit of analysis are temporal dependent site variables
    keys are 'Mine', 'Year'
    '''
    return None

def merge_site_temp_com_data():
    '''
    Unit of analysis are tmporal dependent and commodity dependent site variables
    keys are 'Mine', 'Commodity', 'Year'
    '''
    return None



def get_site_com_files():
    '''
    Get the site commodity files
    '''
    return None

if __name__ == '__main__':

    var_exp_path = get_path('sp_variable_description.xlsx')

    site_folder_path = get_path('site')

    temp_folder_path = get_path('site_temp')
    
    com_folder_path = get_path('site_temp_com')

    site = get_site_data(site_folder_path, var_exp_path)
