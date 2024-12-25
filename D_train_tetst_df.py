'''

This script is used to combine the

* target vars
* commodity vars
* lithography vars
* environmental regulation variable
* geography

and split into a dataset for prediction of waste rock and a dataset for prediction of tailing mass.


'''

import pandas as pd

from M_prod_model import hubbert_model, femp
from R_prod_analysis import identify_cum_model
from D_land_map import alloc_poly_to_sp
import geopandas as gpd
from tqdm import tqdm
from util import append_to_excel, data_to_csv_int, df_to_gpkg

def return_integrated_values(row, df_res):
    '''
    This function is used to return the integrated values for the production model.
    The function is used in the apply method of the dataframe.
    '''
    # specify the model type

    period = 2019 - df_res[df_res.Prop_id == row['Prop_id']]['Start_up_year'].values[0]
    
    if row['Class'] == 'H':
        p1, p2, p3 = df_res[(df_res.Prop_id == row['Prop_id']) & 
                            (df_res.Target_var == row['Target_var']) &
                            (df_res.Model == 'hubbert')][['P1_value', 'P2_value', 'P3_value']].values.flatten()
        return period, hubbert_model(period, p1, p2, p3)
        
    elif row['Class'] == 'F':
        # unpack the values
        p1, p2 = df_res[(df_res.Prop_id == row['Prop_id']) & 
                            (df_res.Target_var == row['Target_var']) &
                            (df_res.Model == 'femp')][['P1_value', 'P2_value']].values.flatten()
        return period, femp(period, p1, p2)

    else:
        return None
    

def get_cumsum(df):
    cum_ident = identify_cum_model(df)
    cum_ident[['Year_delt','Cumsum_2019']] = cum_ident.apply(lambda row: return_integrated_values(row, df), axis=1)

    cum_pivot = cum_ident.pivot_table(index='Prop_id', columns='Target_var', values='Cumsum_2019').reset_index()

    return cum_pivot

def com_col_trans(df, threshold=0, com_path = r'data\variable_description.xlsx'):


    rel_com = pd.read_excel(com_path, sheet_name='Byproduct_conc')['Com_names_sp'].tolist()

    # One-hot encode materials with a minimum frequency threshold
    material_dummies = pd.get_dummies(df['Material_list'].str.split(',', expand=True).stack())
    material_dummies = material_dummies.groupby(level=0).sum()
    material_dummies = material_dummies.loc[:, material_dummies.sum() > threshold]
    
    # Create a set of primary commodities for each mine
    primary_commodities = df['Primary_commodities'].str.split(',').apply(lambda x: set(map(str.strip, x)))
    
    # Initialize the result dataframe with the mine_id
    result = pd.DataFrame({'id_data_source': df['id_data_source']})
    
    # assert all rel materials are in material dummies
    assert all([com in material_dummies.columns for com in rel_com]), 'Not all relevant commodities are in the material dummies'


    # Create primary and secondary columns for each material
    for material in tqdm(rel_com):
        result[f'Primary_{material}'] = ((material_dummies[material] == 1) & primary_commodities.apply(lambda x: material in x)).astype(int)
        result[f'Byprod_{material}'] = ((material_dummies[material] == 1) & ~primary_commodities.apply(lambda x: material in x)).astype(int)
    

    # merge with the original data
    result = df.merge(result, on='id_data_source', how='left')
    result.drop(['Primary_commodities', 'Material_list'], axis=1, inplace=True)
    return result

def byproduct_map(conc_path=r'data\es4c05293_si_001(1).xlsx', output_path= r'data\variable_description.xlsx'):
    # Read only the relevant columns
    rel_com = ['Cu', 'Zn', 'Ni']
    usecols = ['Host'] + rel_com
    
    # Read the concordance file efficiently
    conc = pd.read_excel(conc_path, sheet_name='ResC', usecols=usecols)
    
    # Replace NaN with 0 and convert to boolean in one step
    conc[rel_com] = conc[rel_com].fillna(0).astype(bool)
    
    # Sum the relevant byproducts
    conc['Relevant'] = conc[rel_com].sum(axis=1)
    
    # Filter rows where at least one byproduct is present
    conc = conc[conc['Relevant'] > 0]
    
    # If you need to save the result
    # conc.to_csv(output_path, index=False)

    cg = conc.groupby('Host')[rel_com].apply(lambda x: x.any()).reset_index()
    
    append_to_excel(output_path, cg, 'Byproduct_conc')

    return conc

def divide_ws_tail(df, crs = 'EPSG:6933'):
    # assert that the prop ids are unique
    assert df['Prop_id'].nunique() == df.shape[0], 'The prop ids are not unique'

    # subset non na ws and tailings
    ws = df.dropna(subset=['Waste_rock_production', 'Ore_processed_mass'])
    ws.drop(['Tailings_production', 'Concentrate_production'], axis=1, inplace=True)
    t = df.dropna(subset=['Tailings_production', 'Concentrate_production'])
    t.drop(['Waste_rock_production', 'Ore_processed_mass'], axis=1, inplace=True)

    df_to_gpkg(ws, 'waste_rock', crs = crs)
    df_to_gpkg(t, 'tailings', crs = crs)

    return None

def main():
     # load the modelres
    mod_res_p = r'data\int\production_model_fits.json'
    targets_fit_p = r'data\int\D_target_prio_prep\target_vars_prio_source.csv'
    li_path = r'data\int\D_lito_map\li_class_by_mine.csv'
    land_com_path = r'data\int\D_land_map\land_com.csv'
    land_com_path = r'data\int\D_land_map\allocated_area_coms.gpkg'

    mod_res = pd.read_json(mod_res_p)
    targets_fit = pd.read_csv(targets_fit_p)

    merged = pd.merge(mod_res, targets_fit[['Prop_id', 'Start_up_year']].drop_duplicates(), on=['Prop_id'], how='left')
    
    cum = get_cumsum(merged)

    li = pd.read_csv(li_path)

    area_com = gpd.read_file(land_com_path)

    area_com_trans = com_col_trans(area_com)

    merge_li = area_com_trans.merge(li, on='id_data_source', how='left')

    # harmonize id type 
    cum['Prop_id'] = cum['Prop_id'].astype(str)
    merge_li['id_data_source'] = merge_li['id_data_source'].astype(str)

    merge_cum = cum.merge(merge_li, left_on='Prop_id', right_on='id_data_source', how='left')

    divide_ws_tail(merge_cum)

    return None


if __name__ == '__main__':
   
    main()