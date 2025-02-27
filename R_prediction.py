
import pandas as pd

############################################Purpose############################################



##########################################Params#############################################
vars = ['id_data_source', 'Polygon_count', 'Weight',
       'Unary_area', 'Unary_area_weighted', 'Convex_hull_area',
       'Convex_hull_area_weighted', 'Convex_hull_perimeter',
       'Convex_hull_perimeter_weighted', 'Compactness', 'Compactness_weighted',
       'Coalloc_mines_count', 'Primary_Chromium', 'Byprod_Chromium',
       'Primary_Cobalt', 'Byprod_Cobalt', 'Primary_Copper', 'Byprod_Copper',
       'Primary_Crude Oil', 'Byprod_Crude Oil', 'Primary_Gold', 'Byprod_Gold',
       'Primary_Indium', 'Byprod_Indium', 'Primary_Iron', 'Byprod_Iron',
       'Primary_Lead', 'Byprod_Lead', 'Primary_Manganese', 'Byprod_Manganese',
       'Primary_Molybdenum', 'Byprod_Molybdenum', 'Primary_Nickel',
       'Byprod_Nickel', 'Primary_Palladium', 'Byprod_Palladium',
       'Primary_Platinum', 'Byprod_Platinum', 'Primary_Rhenium',
       'Byprod_Rhenium', 'Primary_Silver', 'Byprod_Silver', 'Primary_Tin',
       'Byprod_Tin', 'Primary_Titanium', 'Byprod_Titanium', 'Primary_Tungsten',
       'Byprod_Tungsten', 'Primary_Uranium', 'Byprod_Uranium',
       'Primary_Vanadium', 'Byprod_Vanadium', 'Primary_Zinc', 'Byprod_Zinc',
       'ev', 'mt', 'nd', 'pa', 'pb', 'pi', 'py', 'sc', 'sm', 'ss', 'su', 'va',
       'vb', 'vi', 'wb',  'Latitude', 'Longitude']

log_vars = [
    'Unary_area', 'Unary_area_weighted', 'Convex_hull_area',
    'Convex_hull_area_weighted', 'Convex_hull_perimeter',
    'Convex_hull_perimeter_weighted']
    

p_data = r'data\int\D_ml_sample\X_pred_set.csv'
p_geo = r'data\int\M_geography_feature\geo_sim_X_pred.csv.csv'

def get_X_per_var(var_name):

    d = pd.read_csv(p_data)
    geo = pd.read_csv(p_geo)

    geo = geo[geo.Target_var == var_name]

    merge = d.merge(geo, on='id_data_source', how='left')

    merge.set_index('id_data_source', inplace=True)

    pass

def run_prediction(var_name):
    X = get_X_per_var(var_name)
    y = X[var_name]
    X.drop(var_name, axis=1, inplace=True)

    pass

if __name__ == '__main__':
   pass