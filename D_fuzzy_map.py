
from D_sp_data_clean import get_data, var_exp_path 
from plotnine import ggplot, geom_histogram, labs, theme_minimal, aes, theme_538
import pandas as pd
from fuzzywuzzy import process
from tqdm import tqdm
import numpy as np 
from openpyxl import load_workbook
from shapely.geometry import Point
import geopandas as gpd

from util import save_fig_plotnine

def append_to_excel(filename, df, sheet_name):
    with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        # Save the new sheet while keeping the existing ones
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def fuzzy_mapping(wkeys, spkeys):
    '''
    This function performs a fuzzy matching between two lists of keys.
    
    Parameters:
        wkeys (list): A list of keys from the Werner data.
        spkeys (list): A list of keys from the SP data.
        
    Returns:
        dict: A dictionary containing the Werner keys as keys and the closest SP key as values.
    '''
    
    
    wkeylist = []
    spkeylist = []
    scores = []
    
    for wkey in tqdm(wkeys, 'matching keys'):
        match, score = process.extractOne(wkey, spkeys)
        wkeylist.append(wkey)
        spkeylist.append(match)
        scores.append(score)

    df = pd.DataFrame({'werner_key': wkeylist, 'sp_key': spkeylist, 'score': scores})

    append_to_excel(r'data\variable_description.xlsx', df, 'fuzzy_werner_sp')

    print(np.mean(scores))

    return df


def score_hist(scores):
    """
    This function plots a histogram of the fuzzy matching scores using plotnine.
    
    Parameters:
        scores (dict): A dictionary containing the fuzzy matching scores.
        
    Returns:
        None
    """
    # Convert scores to a DataFrame
    score_df = pd.DataFrame({'scores': scores})
    
    # Create the plot
    plot = (
        ggplot(score_df, aes(x='scores')) +
        geom_histogram(binwidth = 5, alpha=0.7) +  
        labs(
            x='Fuzzy Matching Score',
            y='Frequency'
        ) +
        theme_538()   # Use a different theme if desired
    )
    
    save_fig_plotnine(plot, 'fuzzy_score_hist.png')
    # Display the plot
    print(plot)
    


sp = get_data('site')
werner = pd.read_csv(r'data\int\area_cleaned.csv')

def main_fuzzy():
    
    
    sp_keys = sp['Prop_name'].unique()
    werner_keys = werner['Mine'].unique()

    mapping = fuzzy_mapping(werner_keys, sp_keys)
    return mapping

def d_calc(row, sp, werner):
        '''
        Calculate the distance between two keys.
        '''
        point_werner = werner[werner['Mine'] == row['werner_key']].geometry.values[0]
        point_sp = sp[sp['Prop_id'] == row['Prop_id']].geometry.values[0]

        return point_werner.distance(point_sp)

def main_add_prop_id():
    '''
    Add the Prop_id to the fuzzy mapping DataFrame.
    '''
    fuzzy_conc = pd.read_excel(var_exp_path, sheet_name='fuzzy_werner_sp')
    #score_hist(fuzzy_conc['score'])
    fuzzy_conc = fuzzy_conc.merge(sp[['Prop_name', 'Prop_id']], left_on='sp_key', right_on='Prop_name', how='left')

    sp_geo = gpd.GeoDataFrame(sp, geometry=gpd.points_from_xy(sp['Longitude'], sp['Latitude']), crs='EPSG:4326')
    werner_geo = gpd.GeoDataFrame(werner, geometry=gpd.points_from_xy(werner['Long'], werner['Lat']), crs='EPSG:4326')

    sp_geo.to_crs('EPSG:3395', inplace=True)
    werner_geo.to_crs('EPSG:3395', inplace=True)

    # compute distance between werner and sp keys
    fuzzy_conc['Distance'] = fuzzy_conc.apply(lambda row: d_calc(row, sp =  sp_geo, werner= werner_geo), axis=1)

    # flag per werner_key and prop_id pair the one with the smallest distance get True / False
    fuzzy_conc['Flag_min_distance'] = fuzzy_conc.groupby('werner_key')['Distance'].transform('min') == fuzzy_conc['Distance']

    append_to_excel(var_exp_path, fuzzy_conc, 'fuzzy_werner_sp')

    return None

if __name__ == '__main__':
    main_add_prop_id()