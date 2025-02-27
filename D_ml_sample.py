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
from scipy.stats import norm, lognorm, uniform
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from plotnine import *



from M_prod_model import hubbert_model, femp
from R_cumprod_per_mine_analysis import add_mine_context
from D_land_map import alloc_poly_to_sp

from util import append_to_excel, df_to_csv_int, df_to_gpkg, save_fig_plotnine


############################################Purpose############################################



###########################################Parameter############################################
mod_res_p = r'data\int\production_model_fits_trans.json'
unc_res_p = 'data\int\M_cumprod_mc_confidence\cumprod_mc_confidence.csv'

li_path = r'data\int\D_lito_map\li_class_by_mine.csv'
poly_path = r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg'
area_path = r'data\int\D_land_map\allocated_area_union_geom.gpkg'
cluster_path = r'data\dcrm_cluster_data\dcrm_cluster_data\cluster_points_concordance.csv'

world_bound_p = r'data\world_bound\world-administrative-boundaries.shp'
eps_p = r'data\eps\OECD,DF_EPS,+all(1).csv'
area_p = r'data\int\D_land_map\allocated_area.gpkg'
all_coms = False

sig = .05

get_predicton_set = True

target_commodities = ['Copper', 'Zinc', 'Nickel']

def get_cumsum():
    # Identify cumulative model data
    modelres = pd.read_json(mod_res_p)
    ua = pd.read_csv(unc_res_p)

    ua['Year'] = ua['Time_period'] + ua['Start_up_year']
    
    # filter year 2019
    ua = ua[ua['Year'] == 2019]

    # merge with the model results
    df_res = ua.merge(modelres, on=['Prop_id', 'Target_var', 'Model'], how='left')

    # filter hubbert and p1_pval, p2_pval, p3_pval are significant
    df_res = df_res[(df_res['Model'] == 'hubbert') & df_res[['P1_pval', 'P2_pval', 'P3_pval']].apply(lambda x: x < sig).all(axis=1)]

    column_select = ['Prop_id', 'Target_var', 'F_p_star', 'F_lower_ci', 'F_upper_ci', 'R2', 'NRMSE', 'Start_up_year_x']

    df_res = df_res[column_select]

    #rename
    df_res.rename(columns={'Start_up_year_x': 'Start_up_year', 'F_p_star': 'Cum_prod', 'F_upper_ci': 'Cum_prod_upper', 'F_lower_ci': 'Cum_prod_lower'}, inplace=True)

    df_cont = add_mine_context(df_res)

    # Transform the Prop_id col to string
    df_cont['Prop_id'] = df_cont['Prop_id'].astype(str)

    return df_cont

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

def get_coms_to_area():
    '''
    This function is used to merge the commodities to the area data.
    One duplicate is dropped in the process, 
    '''
    cluster_df = pd.read_csv(cluster_path)
    poly_df = gpd.read_file(poly_path)
    area_df = gpd.read_file(area_path)

    poly_to_cluster = poly_df.merge(cluster_df, on='id_cluster', how='left')
    subset = poly_to_cluster[['id_data_source', 'primary_materials_list', 'materials_list']].drop_duplicates()

    subset['id_data_source'] = subset['id_data_source'].astype(str)

    # drop duplicates - there
    subset.drop_duplicates(subset='id_data_source', inplace=True)

    # merge with the area df
    area_com = area_df.merge(subset, on='id_data_source', how='left')

    area_com['Coalloc_mines_count'] = 1 / area_com['Weight']

    area_com.rename(columns={'Area_mine': 'Unary_area', 'Area_mine_weighted': 'Unary_area_weighted'}, inplace=True)
    
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


#################################################Main############################################
def main():

    cumsum = get_cumsum()
    
    # merge the area with the mines
    area_com = get_coms_to_area()

    li = pd.read_csv(li_path)

    # transform the material col
    area_com_trans = com_col_trans(area_com, do_all_coms=all_coms)

    merge_li = area_com_trans.merge(li, on='id_data_source', how='left')

    eps = get_eps_per_mine()

    merge_eps = merge_li.merge(eps, on='id_data_source', how='left')

    # transform the geography column in latitude longitude
    merge_eps['Latitude'], merge_eps['Longitude'] = merge_eps.geometry.y, merge_eps.geometry.x


    cols_to_drop = ['Unnamed: 0', 'geometry','id_data_source', 'continent', 'iso3', 'COU']
    if get_predicton_set:
        # take only instances where Latitude Longitude is not null
        pred_feat = merge_eps[(~merge_eps.Latitude.isna()) & (~merge_eps.Longitude.isna())]

        pred_feat = pred_feat.drop(columns=cols_to_drop)

        # filter out all columns that contain target commodities
        target_cols = [col for col in pred_feat.columns if any([com in col for com in target_commodities])]

        # filter only instances that contain at least one target commodity
        pred_feat = pred_feat[pred_feat[target_cols].sum(axis=1) > 0]


        # filter out zero instances
        pred_feat.dropna(inplace=True)

        assert pred_feat.isna().sum().sum() == 0, 'There are still missing values in the prediction features'


        df_to_csv_int(pred_feat, 'X_pred_set')
    
    else:

        cumsum_final = cumsum.merge(merge_eps, left_on = 'Prop_id', right_on = 'id_data_source', how = 'left')


        cumsum_final.drop(columns= cols_to_drop, inplace = True)

        cumsum_final = cumsum_final[~cumsum_final.Latitude.isna()]

        # Filter out zeros only columns
        cumsum_final = cumsum_final[~cumsum_final.isnull()]

        df_to_csv_int(cumsum_final, 'ml_sample')


if __name__ == '__main__':
    
    main()