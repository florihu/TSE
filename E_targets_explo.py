
import pandas as pd
from plotnine import *
from util import df_to_latex, save_fig_plotnine, data_to_csv_int
import numpy as np


def reindex_df(df, targets):
    ''' 
    Here we define a reindexed df that starts for every mine wiht the first year of production
    
    '''
    # Reindex the data with the start year of a mine

    df = df.copy()

    df['Year'] = pd.to_datetime(df['Year']).dt.year.astype(int)
    df['Start_up_year'] = pd.to_datetime(df['Start_up_year']).dt.year
    df = df[df['Start_up_year'].isna() == False]
    df['Start_up_year'] = df['Start_up_year'].astype(int)

    
    all_years_df = (
    df[['Prop_id', 'Start_up_year']]
    .drop_duplicates()
    .assign(Year=lambda x: x.apply(lambda row: list(range(row['Start_up_year'], 2023)), axis=1))
    .explode('Year')
    .drop(columns='Start_up_year')
    )
    
    columns_to_select = targets + ['Prop_id', 'Year']  # Create a new list
    res1 = pd.merge(all_years_df, df[columns_to_select], on=['Prop_id', 'Year'], how='left')

    res2 = pd.merge(res1, df[['Prop_id', 'Start_up_year']].drop_duplicates(), on='Prop_id', how='left')

    # assert that there are no duplicate prop_id-year pairs
    assert len(res2) == len(res2.drop_duplicates(subset=['Prop_id', 'Year'])), 'Duplicated id-year pairs in res2'
        
    return res2

def missing_values_per_mine(df, targets):
    # Compute the proportion of missing values for each target column, grouped by 'Prop_id'
    prop = df.groupby('Prop_id')[targets].apply(lambda group: group.isna().mean())

    summary = prop.describe().T

    df_to_latex(summary, 'missing_values_per_mine')

    return prop

def histplot_with_v_line(df):
    
    df_melt = df.melt(value_vars=['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production'], var_name='Target', value_name='Value')

    # histplot with .1, .25 and .3 lines
    p = (ggplot(df_melt, aes(x='Value', fill='Target'))
         + geom_histogram(bins=30, alpha = .6)
         + facet_wrap('~Target', scales='free')
         + geom_vline(xintercept=[.1, .25, .4], color='black')
         + theme_minimal()
         + scale_fill_brewer(type='qual', palette='Set2')
         )
    
    # add a text to the lines
    #p = p + geom_text(aes(x=[.1, .25, .4], y=0, label=['.1', '.25', '.4']), color='black', size=6, nudge_y=0.01)


    save_fig_plotnine(p, 'histplot_missing_values_per_mine')
    p.draw()

    return None
    
def heatmap_missing_values_per_mine(df, targets):
    '''
    
    Assumptions
    * only for mines with at least ten values
    * only for years from 1950 onwards
    '''
    df_indexed = df.set_index(['Prop_id', 'Year'])
    df_missing = df_indexed.isna().astype(int).reset_index()

    for t in targets:
        df_t = df_missing[['Year', 'Prop_id', t]] 

        # take only mines with at least ten values
        df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)

        # prop id categorical
        df_t['Prop_id'] = pd.Categorical(df_t['Prop_id'], categories=df_t['Prop_id'].unique(), ordered=True)

        
        df_t = df_t[df_t.Year >= 1950]
        df_t = df_t.pivot(index='Prop_id', columns='Year', values=t)
        df_t = df_t.fillna(-1).applymap(lambda x: "Not_applicable" if x == -1 else ("Missing" if x == 1 else "Value_present"))

        p = (
            ggplot(df_t.reset_index().melt(id_vars='Prop_id', var_name='Year', value_name='Category'))
            + geom_tile(aes(x='Year', y='Prop_id', fill='Category'), color='white')
            + scale_fill_manual(values={"Not_applicable": "white", "Missing": "#c51b7d", "Value_present": "#4d9221"})
            + theme_minimal()
            + theme(axis_text_x=element_text(angle=90, hjust=1))
        )
        # axes annotation off
        p = p + theme(axis_text_x=element_blank(), axis_text_y=element_blank(), axis_ticks_major_x=element_blank(), axis_ticks_major_y=element_blank())

        save_fig_plotnine(p, f'heatmap_missing_values_{t}')
    return None
    
def time_series_per_target(df, targets):

    '''
    Assumptions:
    1. Only for the 49 mines with the most non nan values
    2. Only for the years from 1950
    
    '''
    # Time series plot for each target variable
    for t in targets:
        df_t = df[['Year', 'Prop_id', t]]
        
        # 49 mines wiht most non nan values
        top_49 = df[~df[t].isna()]['Prop_id'].value_counts().nlargest(49).index
        df_t = df_t[df_t['Prop_id'].isin(top_49)]

        df_t['Year'] = pd.to_datetime(df_t['Year']).dt.year.astype(int)
        df_t = df_t[df_t['Year'] >= 1950]

        df_t[t] = df_t[t] / 10**6 # convert to Mt
        p = (
            ggplot(df_t, aes(x='Year', y=t))
            + geom_point()
            + geom_smooth(method='loess', se=True, color = '#b2182b')
            + theme_minimal()
            + labs(x='Year', y=f'{t} (Mt)')
            + facet_wrap('~Prop_id', scales='free')
        )

        save_fig_plotnine(p, f'time_series_{t}', w= 18, h= 12)

    return None

def hist_per_mine(df, targets):
    # Time series plot for each target variable
    for t in targets:
        df_t = df[['Year', 'Prop_id', t]]
        
        # 49 mines wiht most non nan values
        top_49 = df[~df[t].isna()]['Prop_id'].value_counts().nlargest(49).index
        df_t = df_t[df_t['Prop_id'].isin(top_49)]

        df_t['Year'] = pd.to_datetime(df_t['Year']).dt.year.astype(int)
        df_t = df_t[df_t['Year'] >= 1950]

        df_t[t] = df_t[t] / 10**6 # convert to Mt

        p = (
            ggplot(df_t, aes(x=t))
            + geom_histogram(bins=30, alpha = .6)
            + theme_minimal()
            + labs(x=f'{t} (Mt)')
            + facet_wrap('~Prop_id', scales='free')
        )
       

        save_fig_plotnine(p, f'hist_per_mine_{t}', w= 18, h= 12)

def characteristics_per_mine(df, targets):
    """
    Calculate descriptive statistics for specified targets per 'Prop_id',
    including the median, for properties with at least 10 values.

    Args:
    - df (DataFrame): Input data containing 'Year', 'Prop_id', and target columns.
    - targets (list): List of target column names to analyze.

    Returns:
    - None: Outputs results to LaTeX files.
    """
    for t in targets:
        df_t = df[['Year', 'Prop_id', t]].copy()
        
        # Filter groups with more than 10 values
        df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)
        
        # Compute describe statistics
        stats = df_t.groupby('Prop_id')[t].describe()
        
        # Add median explicitly
        median = df_t.groupby('Prop_id')[t].median()
        stats['median'] = median

        # Save results to LaTeX
        df_to_latex(stats, f'sample_characteristics_{t}')

def samples_per_target_variable(df, targets):
    """
    Calculate the number of samples per target variable.

    Args:
    - df (DataFrame): Input data containing 'Year', 'Prop_id', and target columns.
    - targets (list): List of target column names to analyze.

    Returns:
    - None: Outputs results to LaTeX files.
    """
    res = []
    for t in targets:
        df_t = df[['Year', 'Prop_id', t]].copy()
        
        # Filter groups with more than 10 values
        df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)
        
        # Count samples per target variable
        samples = df_t.groupby('Prop_id')[t].count().describe()

        res.append(samples)

    res = pd.concat(res, axis=1)

    # Save results to LaTeX
    df_to_latex(res, 'samples_per_target_variable')

    return None

def rolling_outlier_detection(data, window_size, threshold):
    """
    Detect and handle outliers in time series using rolling statistics.

    Args:
    - data (pd.Series): Time series data.
    - window_size (int): Size of the rolling window.
    - threshold (float): Number of standard deviations for outlier detection.

    Returns:
    - cleaned_data (pd.Series): Time series with outliers handled.
    - outliers (pd.Series): Boolean mask indicating outlier positions.
    """
    if data.empty or data.isnull().all():
        # Handle cases with empty or NaN-only data
        return data, pd.Series(False, index=data.index)

    # Compute rolling mean and standard deviation
    rolling_mean = data.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = data.rolling(window=window_size, center=True, min_periods=1).std()

    # Identify outliers
    outliers = (np.abs(data - rolling_mean) > threshold * rolling_std)

    # Handle outliers (replace with rolling mean)
    cleaned_data = data.copy()
    cleaned_data[outliers] = rolling_mean[outliers]

    return cleaned_data, outliers


def rolling_mean_outlier_loop(df_t, t, targets, window_size, threshold):
    """
    Calculate the rolling mean and identify outliers. 

    Take only mines with at least 10 values. Sample random for respresentation

    Args:
    - df (pd.DataFrame): Input data containing 'Year', 'Prop_id', and target columns.
    - targets (list): List of target column names to analyze.

    Returns:
    - pd.DataFrame: DataFrame containing outlier flags and cleaned data for each target.
    """
    # sorting important for rolling mean compute
    df_t = df_t.sort_values(['Prop_id', 'Year'])

    # Loop over groups and apply rolling outlier detection
    df_t[f'{t}_outlier'] = df_t.groupby('Prop_id')[t].transform(
        lambda x: rolling_outlier_detection(x, window_size=window_size, threshold=threshold)[1]
    )

    return df_t

def plot_time_series_target_outliers(df, targets, threshold=2):

    for t in targets:
        df_t = df[['Year', 'Prop_id', t]].copy()
        df_t = df_t[(df_t[t].isna() == False) & (df_t[t] >0)]
        df_t['Year'] = pd.to_datetime(df_t['Year']).dt.year.astype(int)

        # choose randomly 49 mines
        rand_49 = df_t['Prop_id'].sample(49).unique()
        df_t = df_t[df_t['Prop_id'].isin(rand_49)]


        for w in [5, 7, 10, 12]: 
            
            # Filter groups with more than 10 values
            df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)
            
            df_t = rolling_mean_outlier_loop(df_t, t, targets, window_size=w, threshold=threshold)
            # scale to Mt
            df_t[t] = df_t[t] / 10**3
            # Plot time series with outliers
            p = (
                ggplot(df_t, aes(x='Year', y=t, color=f'{t}_outlier'))
                + geom_point()
                + theme_minimal()
                + labs(x='Year', y=f'{t} (kt)')
                + facet_wrap('~Prop_id', scales='free')
            )

            # add color map red outlier non black
            p = p + scale_color_manual(values={True: '#b2182b', False: '#4d4d4d'})

            save_fig_plotnine(p, f'time_series_{t}_window_{w}_outliers', w=18, h=12)

    return None

def outlier_detection_and_ouput(df, targets, 
                                window_dict = {'Tailings_production': 10, 'Waste_rock_production': 7,
                                                'Ore_processed_mass': 10, 'Concentrate_production': 10}, 
                                threshold=2):
    collected_dfs = []

    df = df.groupby('Prop_id').filter(lambda x: any(x[i].count() >= 10 for i in targets))

    for t in targets:
        df_t = df[['Year', 'Prop_id', t]].copy()
        df_t = df_t[(df_t[t].isna() == False) & (df_t[t] >0)]
        df_t = rolling_mean_outlier_loop(df_t, t, targets, window_size=window_dict[t], threshold=threshold)
        df_t = df_t[[f'{t}_outlier', 'Prop_id', 'Year']]
        collected_dfs.append(df_t)

    df = df.copy()
    for df_outlier in collected_dfs:
        df = df.merge(df_outlier, on=['Prop_id', 'Year'], how='left')

    df.drop(columns='Unnamed: 0', inplace=True)

    # Average number of outliers per mine and target
    outlier_stat = df.groupby('Prop_id')[[f'{t}_outlier' for t in targets]].mean().astype(float).describe()

    # Save results to LaTeX
    df_to_latex(outlier_stat, 'outlier_counts')

    data_to_csv_int(df, 'target_sample_with_outlier_detection')
    return df
    



def main():
    
    path = r'data\int\D_target_prio_prep\target_vars_prio_source.csv'

    data = pd.read_csv(path)
    targets = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    outlier_detection_and_ouput(data, targets)
    
    return None



if __name__ == '__main__':
    main()