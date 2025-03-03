
import pandas as pd
import joblib
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt 
import seaborn as sns

from util import df_to_csv_int, save_fig

import cartopy.crs as ccrs
import cartopy.feature as cfeature



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

model_paths = {'Ore_processed_mass': r'models\SVR_synth_Ore_processed_mass.pkl',
               'Tailings_production': r'models\SVR_synth_Tailings_production.pkl',
               'Concentrate_production': r'models\SVR_synth_Concentrate_production.pkl'}

def get_X_per_var(var_name):

    d = pd.read_csv(p_data)
    d.drop(['Unnamed: 0'], axis=1, inplace=True)
    geo = pd.read_csv(p_geo)
    geo.drop(['Unnamed: 0'], axis=1, inplace=True)

    geo = geo[geo.Target_var == var_name]

    merge = d.merge(geo, on='id_data_source', how='left')

    merge.set_index('id_data_source', inplace=True)
    merge.drop(['Target_var'], axis=1, inplace=True)

    return merge

def run_prediction():
    p = pd.read_csv(p_data)

    res = []

    for var_name in ['Ore_processed_mass', 'Tailings_production', 'Concentrate_production']:
        
        X = get_X_per_var(var_name)
        X[log_vars] = np.log(X[log_vars])

        index = X.index
        X.reset_index(inplace=True, drop=True)

        model = joblib.load(model_paths[var_name])

        model_feat = model.feature_names_in_
        X = X[model_feat]

        y_pred = model.predict(X)

        res.append(pd.DataFrame({'id_data_source': index, 'Pred' : y_pred, 'Target_var': [var_name]*len(y_pred)}))

    res = pd.concat(res, axis=0)

    df_to_csv_int(res, 'best_model_prediction')

    pass


def merge2geo():
    p='data\\int\\R_prediction\\best_model_prediction.csv.csv'
    f=r'data\\int\\D_ml_sample\\X_pred_set.csv'

    # Read the CSV files
    df_pred = pd.read_csv(p)
    df_feat = pd.read_csv(f)

    # Clean and merge data
    df_pred.drop(['Unnamed: 0'], axis=1, inplace=True)

    q_99 = df_pred.groupby('Target_var')['Pred'].quantile(0.99).reset_index()
    df = df_pred.merge(q_99, on='Target_var', how='left')
    df = df[df['Pred_x'] < df['Pred_y']]
    df.drop(['Pred_y'], axis=1, inplace=True)
    df.rename(columns={'Pred_x': 'Pred'}, inplace=True)

    df['Pred'] = np.log(df['Pred'])


    df_merged = df.merge(df_feat[['id_data_source', 'Unary_area_weighted', 'Latitude', 'Longitude']], 
                              on='id_data_source', how='left')

    # Create a GeoDataFrame with lat/lon (EPSG:4326)
    gdf = gpd.GeoDataFrame(df_merged, 
                           geometry=gpd.points_from_xy(df_merged.Longitude, df_merged.Latitude),
                           crs='EPSG:6933')
    
    # Transform the GeoDataFrame to EPSG:4326 (geographic coordinates)
    gdf = gdf.to_crs(epsg=4326)

    return gdf


def geoplot_predictions():
    gdf = merge2geo()

    # Define the Cartopy projection (Interrupted Goode Homolosine)
    proj = ccrs.InterruptedGoodeHomolosine()

    # Loop over each unique target variable
    for target in gdf['Target_var'].unique():
        
        subset = gdf[gdf['Target_var'] == target]

        q_99 = subset['Pred'].quantile(0.99)
        subset = subset[subset['Pred'] < q_99]

        subset['Pred'] = np.log10(subset['Pred'])

        # Create a figure and axes with the Cartopy projection
        fig = plt.figure(figsize=(14, 6))
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot the points using scatter; data is in lat/lon so use PlateCarree as transform.
        sc = ax.scatter(subset.geometry.x, subset.geometry.y, 
                        c=subset['Pred'], cmap='viridis', s= subset['Unary_area_weighted'], transform=ccrs.PlateCarree())

        # Add a colorbar
        plt.colorbar(sc, ax=ax, orientation='vertical', label='Prediction log10(t)')	

        handles, labels = sc.legend_elements(prop="sizes", num=5, 
                                              func=lambda s: s)
        size_legend = ax.legend(handles, labels, loc=[0.1,0.05], title="Area (km2)", labelspacing=1, fontsize=8, title_fontsize=8)
        ax.add_artist(size_legend)


        # Save figure using your own save_fig() function
        save_fig(f'spat_explicit_{target}')
        plt.show()


def distributions(p='data\\int\\R_prediction\\best_model_prediction.csv.csv'):
    df = pd.read_csv(p)

    df['Pred'] = np.log(df['Pred'])

    # filter the 99% quantile per variable
    q_99 = df.groupby('Target_var')['Pred'].quantile(0.99).reset_index()
    df = df.merge(q_99, on='Target_var', how='left')
    df = df[df['Pred_x'] < df['Pred_y']]
    df.drop(['Pred_y'], axis=1, inplace=True)
    df.rename(columns={'Pred_x': 'Pred'}, inplace=True)

    g = sns.FacetGrid(df, col='Target_var', col_wrap=3, height=4, sharex=True, sharey=True)
    g.map(sns.boxplot, 'Pred')

    g.set_axis_labels('Prediction log10(t)', 'Count')


    # add a vertical line at the 3*iqr for ever target var
    for ax, target in zip(g.axes, df['Target_var'].unique()):
        q_99 = df[df['Target_var'] == target]['Pred'].quantile(0.99)
        ax.axvline(x=q_99, color='red', linestyle='--')


    save_fig('distributions_pred_per_target')
    plt.show()

def world_regions_agg(crs = '6933'):

    w_regions = gpd.read_file(r'data\world_bound\world-administrative-boundaries.shp')

    gdf = merge2geo()

    w_regions = w_regions.to_crs(epsg=crs)
    gdf = gdf.to_crs(epsg=crs)

    # Spatial join
    w_regions = gpd.sjoin(w_regions, gdf, how='inner', predicate='intersects')

    # sum per region
    w_agg = w_regions.groupby(['iso3', 'Target_var']).agg({'Pred': 'sum'}).reset_index()

    w_agg.rename(columns={'Pred': 'Pred_agg'}, inplace=True)

    w_merge = w_regions.merge(w_agg, on=['iso3', 'Target_var'], how='left')

    w_merge.to_crs(epsg=4326, inplace=True)

    g = sns.FacetGrid(w_merge, col='Target_var', col_wrap=1, height=3, sharex=True, sharey=True)

    # plot the polygons in the facet grid
    for ax, target in zip(g.axes, w_merge['Target_var'].unique()):
        subset = w_merge[w_merge['Target_var'] == target]
        subset['Pred_agg'] = np.exp(subset['Pred_agg'])
        subset['Pred_agg'] = subset['Pred_agg'] / 10**9
        subset.plot(column='Pred_agg', ax=ax, legend=True, cmap='crest', legend_kwds={'label': 'Cumprod (Gt)'})
        ax.set_title(target)

    save_fig('world_regions_agg')
    plt.show()


    pass
    
if __name__ == '__main__':
   world_regions_agg()