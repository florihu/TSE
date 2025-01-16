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
from util import append_to_excel, data_to_csv_int, df_to_gpkg, save_fig_plotnine
from plotnine import *

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
        return period, None   

def get_cumsum(df):
    # Identify cumulative model data
    cum_ident = identify_cum_model(df)

    # Apply the integration function and create new columns
    cum_ident[['Active_years', 'Cumsum_2019']] = cum_ident.apply(
        lambda row: return_integrated_values(row, df), axis=1, result_type="expand"
    )

    # Pivot table to restructure the data
    cum_pivot = cum_ident.pivot_table(
        index=['Prop_id', 'Active_years'], columns='Target_var', values='Cumsum_2019'
    ).reset_index()

    # Return the final pivot table
    return cum_pivot

def com_col_trans(df, threshold=0, do_all_coms = False,  com_path = r'data\variable_description.xlsx'):


    rel_com = pd.read_excel(com_path, sheet_name='Byproduct_conc')['Com_names_sp'].tolist()

    # One-hot encode materials with a minimum frequency threshold
    material_dummies = pd.get_dummies(df['materials_list'].str.split(',', expand=True).stack())
    material_dummies = material_dummies.groupby(level=0).sum()
    material_dummies = material_dummies.loc[:, material_dummies.sum() > threshold]
    
    # Create a set of primary commodities for each mine
    primary_commodities = df['primary_materials_list'].apply(
            lambda x: set(map(str.strip, x.split(','))) if isinstance(x, str) else set()
        )
    
    # Initialize the result dataframe with the mine_id
    result = pd.DataFrame({'id_data_source': df['id_data_source']})
    
    # assert all rel materials are in material dummies
    assert all([com in material_dummies.columns for com in rel_com]), 'Not all relevant commodities are in the material dummies'

    if do_all_coms:
        rel_com = material_dummies.columns.to_list()

        rel_com.remove('Not relevant')

    # Create primary and secondary columns for each material
    for material in tqdm(rel_com):
        result[f'Primary_{material}'] = ((material_dummies[material] == 1) & primary_commodities.apply(lambda x: material in x)).astype(int)
        result[f'Byprod_{material}'] = ((material_dummies[material] == 1) & ~primary_commodities.apply(lambda x: material in x)).astype(int)
    

    # merge with the original data
    result = df.merge(result, on='id_data_source', how='left')
    result.drop(['primary_materials_list', 'materials_list'], axis=1, inplace=True)
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
    

    #keep only first instance of prop id
    df.drop_duplicates(subset='Prop_id', inplace=True)
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

def get_coms_to_area(poly_df_p, cluster_df_p,  area_df_p):
    '''
    This function is used to merge the commodities to the area data.
    One duplicate is dropped in the process, 
    '''
    cluster_df = pd.read_csv(cluster_df_p)
    poly_df = gpd.read_file(poly_df_p)
    area_df = gpd.read_file(area_df_p)

    poly_to_cluster = poly_df.merge(cluster_df, on='id_cluster', how='left')
    subset = poly_to_cluster[['id_data_source', 'primary_materials_list', 'materials_list']].drop_duplicates()

    subset['id_data_source'] = subset['id_data_source'].astype(str)

    # drop duplicates - there
    subset.drop_duplicates(subset='id_data_source', inplace=True)

    # merge with the area df
    area_com = area_df.merge(subset, on='id_data_source', how='left')
    
    #assert that every id is unique
    assert area_com['id_data_source'].nunique() == area_com.shape[0], 'The id_data_source is not unique'

    return area_com

def eps_explo_plot(eps):
    
    # filter important mining countries
    eps = eps[eps['COU'].isin(['AUS', 'CAN', 'CHL', 'RUS', 'USA', 'ZAF', 'BRA', 'PER', 'IDN', 'MEX'])]

    plot = (ggplot(eps, aes(x='TIME_PERIOD', y= 'OBS_VALUE', color='COU')) 
    + geom_point()
    + geom_line()
    + theme_minimal()
    + labs(x='Year', y='EPS')
    + geom_smooth(method='lm', color='black')
    )

    save_fig_plotnine(plot, 'eps_explo.png')
    print(plot)


def get_eps_per_mine():
    
    '''
    Merges environmental performance index (EPI) data with allocated area data based on geographical boundaries.
    This function reads in three datasets: world administrative boundaries, EPI results, and allocated area data.
    It then performs a spatial join to merge the area data with the world boundaries and subsequently merges the 
    resulting dataset with the EPI data based on ISO country codes. The final output is a DataFrame containing 
    the data source IDs and their corresponding EPI values.
    Returns:
        pd.DataFrame: A DataFrame with columns 'id_data_source' and 'EPI', where 'EPI' represents the environmental 
                      performance index for each data source. 
    
    
    '''
    world_bound_p = r'data\world_bound\world-administrative-boundaries.shp'
    eps_p = r'data\eps\OECD,DF_EPS,+all(1).csv'
    area_p = r'data\int\D_land_map\allocated_area.gpkg'

    eps = pd.read_csv(eps_p)
    eps_f = eps[eps['VAR'] == 'EPS']
    #eps_explo_plot(eps_f)
    
    # calculate mean and slope of linear regression per country
    eps_f['OBS_VALUE'] = eps_f['OBS_VALUE'].astype(float)
    eps_f['TIME_PERIOD'] = eps_f['TIME_PERIOD'].astype(int)

    eps_stat = eps_f.groupby('COU').apply(
                lambda x: pd.DataFrame({
                    'COU': [x['COU'].values[0]],
                    'EPS_mean': [x['OBS_VALUE'].mean()],
                    'EPS_slope': [x.sort_values('TIME_PERIOD')['OBS_VALUE'].diff().mean()]
                })).reset_index(drop=True)

    area = gpd.read_file(area_p)
    world_bound = gpd.read_file(world_bound_p)

    world_bound = gpd.read_file(world_bound_p)
    world_bound.to_crs(area.crs, inplace=True)

    j = gpd.sjoin(area, world_bound, how='left', predicate='within')

    m = j[['id_data_source','continent', 'iso3']].merge(eps_stat, left_on='iso3', right_on='COU', how='left')

    # Group by continent and calculate the mean for EPS_mean and EPS_slope
    continent_means = m.groupby('continent')[['EPS_mean', 'EPS_slope']].transform('mean')

    # Fill NaN values in EPS_mean and EPS_slope with continent-level means
    m[['EPS_mean', 'EPS_slope']] = m[['EPS_mean', 'EPS_slope']].fillna(continent_means)

    # assert that the id_data_source is unique
    assert m['id_data_source'].nunique() == m.shape[0], 'The id_data_source is not unique'
    return m


def main():
     # load the modelres
    mod_res_p = r'data\int\production_model_fits.json'
    targets_fit_p = r'data\int\D_target_prio_prep\target_vars_prio_source.csv'
    li_path = r'data\int\D_lito_map\li_class_by_mine.csv'
    poly_path = r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg'
    area_path = r'data\int\D_land_map\allocated_area_union_geom.gpkg'
    cluster_path = r'data\dcrm_cluster_data\dcrm_cluster_data\cluster_points_concordance.csv'
    
    all_coms = True
    area_com = get_coms_to_area(poly_path, cluster_path, area_path)


    li = pd.read_csv(li_path)

    area_com_trans = com_col_trans(area_com, do_all_coms=all_coms)

    merge_li = area_com_trans.merge(li, on='id_data_source', how='left')

    eps = get_eps_per_mine()

    merge_eps = merge_li.merge(eps, on='id_data_source', how='left')

    if all_coms:
        df_to_gpkg(merge_eps, 'features_all_mines_all_coms', crs = 'EPSG:6933')
        
    mod_res = pd.read_json(mod_res_p)





    targets_fit = pd.read_csv(targets_fit_p)

    merged = pd.merge(mod_res, targets_fit[['Prop_id', 'Start_up_year']].drop_duplicates(), on=['Prop_id'], how='left')
    
    # harmonize id type 

    cum = get_cumsum(merged)
    cum['Prop_id'] = cum['Prop_id'].astype(str)
    merge_li['id_data_source'] = merge_li['id_data_source'].astype(str)

    merge_cum = cum.merge(merge_li, left_on='Prop_id', right_on='id_data_source', how='left')

    

    merge_epi = merge_cum.merge(eps, on='id_data_source', how='left')
    
    divide_ws_tail(merge_epi)

    return None


if __name__ == '__main__':
    
    main()