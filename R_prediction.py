
import pandas as pd
import joblib
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt 
import seaborn as sns



import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.preprocessing import StandardScaler

import contextily as ctx


from util import df_to_csv_int, save_fig


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


replace_dict = {'Ore_processed_mass': 'COP',
                'Tailings_production': 'CTP',
                'Concentrate_production': 'CCP'}

palette_dict = {'Copper': 'rocket_r',
                'Nickel':'BuGn',
                'Zinc':'PuBuGn'}



def get_alloc_pred():
    alloc_p = r'data\int\M_alloc\alloc_com.csv.csv'
    alloc = pd.read_csv(alloc_p)
    pred = merge2geo()

    merge = pred.merge(alloc, on='id_data_source', how='left')

    melt = merge.melt(id_vars=['id_data_source', 'geometry', 'Pred', 'Unary_area_weighted','Target_var', 'Commodity'],
                        value_vars=['Occ_weight', 'Prim_weight'],
                        var_name='Alloc_type', value_name='Weight')
    
    melt['Alloc_type'] = melt['Alloc_type'].apply(lambda x: x.split('_')[0])
    split = melt['Commodity'].str.split('_')

    melt['Com_type'], melt['Commodity'] = split.str[0], split.str[1]
    return melt

def get_countries_pred(crs = 6933, valid_switch = False):

    w_regions = gpd.read_file(r'data\world_bound\world-administrative-boundaries.shp')

    w_regions.to_crs(epsg=crs, inplace=True)

    pred = get_alloc_pred()
    pred.to_crs(epsg=crs, inplace=True)

    merge = pred.sjoin(w_regions[['name', 'iso3', 'geometry']], how='left')

    merge['Weight_pred'] = merge['Weight'] * merge['Pred']

    agg = merge.groupby(['iso3', 'name', 'Target_var', 'Commodity', 'Alloc_type'])['Weight_pred'].sum().reset_index()

    agg.rename(columns={'Weight_pred': 'Cumprod_weight'}, inplace=True)
    return agg

def bar_country_alloc_facet():
    data = get_countries_pred()
    data['Target_var'] = data['Target_var'].apply(lambda x: replace_dict[x])

    # Filter for target variable "COP" and selected commodities
    data = data[(data['Target_var'] == 'COP') & data['Commodity'].isin(['Copper', 'Nickel', 'Zinc'])]

    # Find top 10 producing countries for each commodity
    top_countries = (
        data[data['Alloc_type'] == 'Occ']
        .groupby(['Commodity', 'iso3'])['Cumprod_weight']
        .sum()
        .groupby(level=0, group_keys=False)
        .nlargest(10)
        .index.get_level_values(1)
        .unique()
    )

    # Keep only data for the top countries
    data = data[data['iso3'].isin(top_countries)]

    # Create FacetGrid
    g = sns.FacetGrid(data, col='Commodity', col_wrap=3, sharex=False, sharey=True)
    
    g.map_dataframe(sns.barplot, x='iso3', y='Cumprod_weight', hue='Alloc_type', errorbar=None)
    g.add_legend(loc=[0.8, 0.7], title='Commodity Type')

    g.set_axis_labels('Country Code', 'COP log10(t)')

    g.set_xticklabels(rotation=45, size = 8)

    g.set_titles(col_template="{col_name}")

    # make axis log 10 scale
    g.set(yscale='log')
    

    plt.tight_layout()
    save_fig("COP_top10_countries_facet_by_alloc")
    plt.show()

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

def return_quantile(data, name):
    if name == 'Tailings_production':
        return data.quantile(0.97)
    else:
        return data.max()+1

def merge2geo():
    p='data\\int\\R_prediction\\best_model_prediction.csv.csv'
    f=r'data\\int\\D_ml_sample\\X_pred_set.csv'

    # Read the CSV files
    df_pred = pd.read_csv(p)
    df_feat = pd.read_csv(f)

    # Clean and merge data
    df_pred.drop(['Unnamed: 0'], axis=1, inplace=True)

    # calculate iqrs per target var
    q_99  = df_pred.groupby('Target_var')['Pred'].apply(lambda x: return_quantile(x, x.name)).reset_index()
    df_pred = df_pred.merge(q_99, on='Target_var', how='left')
    df_pred = df_pred[df_pred['Pred_x'] < df_pred['Pred_y']]
    df_pred.drop(['Pred_y'], axis=1, inplace=True)
    df_pred.rename(columns={'Pred_x': 'Pred'}, inplace=True)

    
    df_merged = df_pred.merge(df_feat[['id_data_source', 'Unary_area_weighted', 'Latitude', 'Longitude']], 
                              on='id_data_source', how='left')

    # Create a GeoDataFrame with lat/lon (EPSG:4326)
    gdf = gpd.GeoDataFrame(df_merged, 
                           geometry=gpd.points_from_xy(df_merged.Longitude, df_merged.Latitude),
                           crs='EPSG:6933')
    
    # Transform the GeoDataFrame to EPSG:4326 (geographic coordinates)
    gdf = gdf.to_crs(epsg=4326)

    return gdf


def return_iqr(data):
    return (data.quantile(0.75) - data.quantile(0.25))*3

def geoplot_predictions(com = None, alloc = 'occ'):
    
    # Define the Cartopy projection (Interrupted Goode Homolosine)
    proj = ccrs.InterruptedGoodeHomolosine()

    if com != None:
        gdf = get_alloc_pred()
            
        # get all commodities that contain the keyword
        gdf = gdf[gdf['Commodity'].str.contains(com)]  

        pal = palette_dict[com]

        if alloc == 'occ':
            gdf['Pred'] = gdf['Pred'] * gdf['Occ_weight']
        elif alloc == 'prim':
            gdf['Pred'] = gdf['Pred'] * gdf['Prim_weight']

        else:
            ValueError('Invalid allocation method')
    else:
        gdf = merge2geo()
        pal ='viridis'

    # Loop over each unique target variable
    for target in gdf['Target_var'].unique():

        subset = gdf[gdf['Target_var'] == target]

        subset['Pred'] = np.log10(subset['Pred'])

        # Create a figure and axes with the Cartopy projection
        fig = plt.figure(figsize=(14, 6))
        ax = plt.axes(projection=proj)
        ax.set_global()
        ax.coastlines(color='grey', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, color='grey', linewidth=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray')


        # Plot the points using scatter; data is in lat/lon so use PlateCarree as transform.
        sc = ax.scatter(subset.geometry.x, subset.geometry.y, 
                        c=subset['Pred'], cmap=pal, s= subset['Unary_area_weighted'], transform=ccrs.PlateCarree())

        # Add a colorbar
        plt.colorbar(sc, ax=ax, orientation='vertical', label='Prediction log10(t)')	


        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
        
        gl.right_labels = True  # Show longitude labels on the right edge
        gl.left_labels = False  # Hide longitude labels on the left (inner plots)

        gl.top_labels = False   # Hide latitude labels on the top (inner plots)
        gl.bottom_labels = True # Show latitude labels on the bottom edge

        gl.xlabel_style = {'size': 8, 'color': 'grey'}
        gl.ylabel_style = {'size': 8, 'color': 'grey'}

        percentiles = np.quantile(subset['Unary_area_weighted'], [0.10, 0.25, 0.50, 0.75, 0.90, 0.99])

        # Create legend handles based on percentile sizes
        handles = [plt.scatter([], [], s=p, color='gray', alpha=0.8) for p in percentiles]
        labels = [f"{p:.1f} km²" for p in percentiles]

        # Add size legend
        size_legend = ax.legend(handles, labels, loc="lower left", title="Area (km²)", 
                        labelspacing=1, fontsize=8, title_fontsize=8)
        
        ax.add_artist(size_legend)

        plt.tight_layout()

        if com != None:
            # Save figure using your own save_fig() function
            save_fig(f'spat_explicit_{target}_{com}_{alloc}')
        else:
            # Save figure using your own save_fig() function
            save_fig(f'spat_explicit_{target}')
        plt.show()


def distributions(p='data\\int\\R_prediction\\best_model_prediction.csv.csv'):
    df = pd.read_csv(p)

    # filter the 99% quantile per variable
    thres  = df.groupby('Target_var')['Pred'].apply(lambda x: return_quantile(x, x.name)).reset_index()
   

    df['Target_var'] = df['Target_var'].apply(lambda x: replace_dict[x])#

    df['Pred'] = df['Pred'] / 10**6


    thres['Target_var'] = thres['Target_var'].apply(lambda x: replace_dict[x])
    g = sns.FacetGrid(df, col='Target_var', col_wrap=3, height=4, sharex=False, sharey=True)
    g.map(sns.boxplot, 'Pred')

    g.set_axis_labels('Prediction (Mt)', 'Count')


    # add a vertical line at the 3*iqr for ever  var
    for ax, target in zip(g.axes, df['Target_var'].unique()):
        thres_t = thres[thres.Target_var == target]['Pred'] / 10**6
        ax.axvline(x=thres_t.values, color='red', linestyle='--')


    save_fig('distributions_pred_per_target')
    plt.show()


def world_regions_agg(crs = '6933',alloc = 'occ'):

    w_regions = gpd.read_file(r'data\world_bound\world-administrative-boundaries.shp')

    gdf = get_alloc_pred()
    

    if alloc == 'occ':
        gdf['Pred'] = gdf['Pred'] * gdf['Occ_weight']
    elif alloc == 'prim':
        gdf['Pred'] = gdf['Pred'] * gdf['Prim_weight']

    else:
        ValueError('Invalid allocation method')
  

   

    gdf['Com_abs'] = gdf['Commodity'].apply(lambda x: x.split('_')[1])

    gdf = gdf[gdf.Com_abs.isin(['Copper', 'Nickel', 'Zinc'])]

    # Spatial join
    w_regions = gpd.sjoin(w_regions, gdf, how='inner', predicate='intersects')

    # sum per region
    w_agg = w_regions.groupby(['iso3', 'Target_var', 'Com_abs']).agg({'Pred': 'sum'}).reset_index()

    w_agg.rename(columns={'Pred': 'Pred_agg'}, inplace=True)

    w_agg['Pred_agg'] = w_agg['Pred_agg'] / 10**6

    w_merge = w_agg.merge(w_regions[['iso3', 'name','geometry']].drop_duplicates(), on=['iso3'], how='left')

    w_merge = gpd.GeoDataFrame(w_merge, geometry='geometry')
    w_merge.to_crs(epsg=4326, inplace=True)



    for t in w_merge['Target_var'].unique():

        w_t = w_merge[w_merge['Target_var'] == t]

        g = sns.FacetGrid(w_t, col='Com_abs', col_wrap=1,  height=2, aspect=5,  sharex=True, sharey=True)

        plt.subplots_adjust(hspace=0.01)

        g.set_xlabels('Longitude')
        g.set_ylabels('Latitude')


        for ax , com in zip(g.axes, w_t['Com_abs'].unique()):

            w_t[w_t.Com_abs == com].plot(column='Pred_agg', cmap='viridis', legend=True, ax=ax, legend_kwds={'label': 'Prediction (Mt)', 'orientation': 'vertical', 'shrink': 0.9})

            #increase the space to the legend
            
        plt.tight_layout()

        save_fig(f'{t}_world_regions_agg_{alloc}')
        plt.show()
      
    
def consistency_check():
    df = merge2geo()
    df['Target_var'] = df['Target_var'].apply(lambda x: replace_dict[x])

    area_per_id = df[['id_data_source', 'Unary_area_weighted']].drop_duplicates()

    piv = df.pivot_table(index='id_data_source', columns='Target_var', values='Pred')

    piv = piv.merge(area_per_id, on='id_data_source', how='left')

    piv['MB_consistent'] = piv['CTP'] < piv['COP']   

    piv['Unary_area_weighted'] = np.log(piv['Unary_area_weighted'])

    piv['COP'] = np.log(piv['COP'])
    piv['CTP'] = np.log(piv['CTP'])

    piv['COP'] = StandardScaler().fit_transform(piv['COP'].values.reshape(-1, 1)).flatten()   
    piv['CTP'] = StandardScaler().fit_transform(piv['CTP'].values.reshape(-1, 1)).flatten()

    sns.scatterplot(data=piv, x='COP', y='CTP', hue='MB_consistent')

    # add 0 line
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)

    
   
    plt.xlabel('COP standardized log(t)')
    plt.ylabel('CTP standardized log(t)')

    save_fig('ctp_consistency_check')
    plt.show()

    
if __name__ == '__main__':
    bar_country_alloc_facet()