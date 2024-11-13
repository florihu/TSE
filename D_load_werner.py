import os
import pandas as pd
import numpy as np
from util import get_path
from D_sp_data_clean import dtype_conversion, var_exp_path

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
    
    area = pd.read_excel(path, sheet_name=sheet_names[0], skiprows=7, usecols='A:S', header = None)
    prod = pd.read_excel(path, sheet_name=sheet_names[1], skiprows=1, usecols='A:AX')
    
    return area, prod

def clean_area(area_df, commodities, out_name):
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

    # Columns conversion

    string_columns = ['Mine','Country', 'Primary']
    date = 'Google'

    area_dropped[string_columns] = area_dropped[string_columns].astype(str)
    # Remove text from date strings
    area_dropped[date] = area_dropped[date].astype(str).str.extract(r'(\d{4})')[0]

    # Convert year to datetime format, setting month and day to a default (e.g., January 1)
    area_dropped[date] = pd.to_datetime(area_dropped[date], format='%Y', errors='coerce')
    
    numeric_columns = numeric_columns = area_dropped.columns.difference(string_columns + [date])


    # Replace empty strings with NaN
    area_dropped.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Proceed to convert numeric columns to float
    area_dropped[numeric_columns] = area_dropped[numeric_columns].astype(float)   

    renamed = area_dropped.rename(columns={'Google': 'Year', 'Primary': 'Commodity'}).set_index('Mine')

    save_intermediate_data_to_csv(renamed, out_name=out_name)

    return renamed


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
    werner_lookup = pd.read_excel(var_exp_path, sheet_name='werner_lookup')
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


'script parameters'
file_name = 'Supplementary Part V - Site Data(1).xlsx'
com_study = ['Cu', 'PbZn', 'PbZnCu', 'Ni', 'PGEs', 'Au', 'Diamonds', 'Uranium']
prod_keys_exclude = ['Cum. Prod.', 'Std Dev', 'Count']
prod_path = get_path(file_name) 
area, prod = load_prod(prod_path)
c_prod = clean_production(prod)


