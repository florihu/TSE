
from util import get_path

import pandas as pd 
import os
import numpy as np
from functools import reduce

def get_data(folder_name, var_exp_path):
    """
    Get site data and merges additional entries.

    Parameters:
        folder_name (str): The name of the site folder.
        var_exp_path (str): The path to the variable explanation file.

    Returns:
        DataFrame: The merged or singular site data.
    """
    
    folder_path = get_path(folder_name)

    if folder_name in ['site', 'site_temp', 'site_com']:
        collect = []
        for f in os.listdir(folder_path):
            site_data = pd.read_excel(os.path.join(folder_path, f), skiprows=4).dropna(how='all')

            id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]

            if folder_name == 'site':
                id_cols = ['PROP_ID']
                dtypes = site_data.iloc[0].astype(str)
                site_data = site_data.iloc[id_col_index:]
                var_types = dict(zip(site_data.columns, dtypes))

            elif folder_name == 'site_com':
                id_cols = ['PROP_ID']
                com = f.split('_')[-1].split('.')[0]
                dtypes = site_data.iloc[0].astype(str)
                var_types = dict(zip(site_data.columns, dtypes))

                site_data = site_data.iloc[1:,id_col_index:]
                site_data['COMMODITY'] = com
                
                var_types['COMMODITY'] = 'str'

            elif folder_name == 'site_temp':
                id_cols = ['PROP_ID', 'YEAR']
                site_data, var_types = col_trans(site_data, id_col_index)
        

            dtype_converted = dtype_conversion(site_data, var_types)
            dtype_converted.set_index(id_cols, inplace=True)
            collect.append(dtype_converted)
        
    if folder_name in  ['site', 'site_temp']:
        merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=id_cols), collect).reset_index()
        return merged
    elif folder_name == 'site_com':
        if len(collect) == 1:
            return collect[0]
        else:
            # sort al dfs in conc by columns
            conc = [df.sort_index(axis=1) for df in collect]
            # assert columns are the same in all dfs that are concated
            assert all([collect[0].columns == c.columns for c in collect[1:]])
            return pd.concat(collect, axis=0)

    elif folder_name == 'site_temp_com':
            conc = []
            for f in os.listdir(folder_path):
                collect = []
                commodity = f.split('/')[-1]
                id_cols = ['PROP_ID', 'YEAR']
                com_path = os.path.join(folder_path, f)
                for f_ in os.listdir(com_path):
                    site_data = pd.read_excel(os.path.join(com_path, f_), skiprows=4).dropna(how='all')
                    
                    id_col_index = np.where(site_data.columns.str.contains('PROP_ID', case=False))[0][0]
                    
                    site_data, var_types = col_trans(site_data, id_col_index)

                    dtype_converted = dtype_conversion(site_data, var_types)
                    dtype_converted.set_index(id_cols, inplace=True)
                    collect.append(dtype_converted)

                merged = reduce(lambda left, right: pd.merge(left, right, how='outer', on=id_cols), collect).reset_index()
                
                merged_reset = merged.reset_index() 

                merged_reset['COMMODITY'] = commodity

                conc.append(merged_reset)
            
            if len(conc) == 1:
                return conc[0]
            else: 
                # sort al dfs in conc by columns
                conc = [df.sort_index(axis=1) for df in conc]
                # assert columns are the same in all dfs that are concated
                assert all([conc[0].columns == c.columns for c in conc[1:]])
                return pd.concat(conc, axis=0)

    



def col_trans(site_data, id_col_index):
    # Assume the third column as the variable name
    var_name = site_data.columns[id_col_index + 1]

    # Set headers based on the first row and adjust columns
    site_data.columns = site_data.iloc[0]
    site_data = site_data.iloc[1:, id_col_index:]
    site_data.rename(columns={site_data.columns[id_col_index]: 'PROP_ID'}, inplace=True)
    site_data = site_data.dropna(subset=['PROP_ID'])

    # Stack data with 'PROP_ID' and 'YEAR' as indexes, renaming the variable column
    stacked = site_data.set_index('PROP_ID').stack()

    stacked.index.set_names(['PROP_ID', 'YEAR'], inplace=True)

    stacked_df = stacked.reset_index().rename({0: var_name}, axis=1)

    # Clean 'YEAR' column by removing 'Y' prefix
    stacked_df['YEAR'] = stacked_df['YEAR'].str.replace('Y', '', regex=False).astype(int)

    # Define variable types based on exceptions or defaults
    var_types = {'YEAR': 'date', 'PROP_ID': 'int', var_name: exept_vars.get(var_name, default_dtype)}

    return stacked_df, var_types





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

            df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df[(df[c] >= 1677) & (df[c] <= 2262)]  # Filter rows with valid years

            # Convert to datetime, handling potential issues with mixed formats
            df[c] = pd.to_datetime(df[c].astype(str), format='%Y', errors='coerce')
            
        elif dtype == 'int':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
        elif dtype == 'float':
            df[c] = pd.to_numeric(df[c], errors='coerce')
        elif dtype == 'str':
            df[c] = df[c].astype('str')

        else:
            AssertionError('The data type could not be assigned')

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


exept_vars = {'MODEL_EST_DATE': 'date'}
default_dtype = 'float'

if __name__ == '__main__':

    var_exp_path = get_path('sp_variable_description.xlsx')

    site_folder_path = get_data('site', var_exp_path)
