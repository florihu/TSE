
import pandas as pd
from plotnine import *

import numpy as np



from util import df_to_latex, save_fig_plotnine, df_to_csv_int

from R_cumprod_per_mine_analysis import add_mine_context


############################################ Purpose ############################################
'''
1. Generate a histogram of the proportion of missing values per mine for each target variable.
2. Generate a heatmap of missing values per mine for each target variable.
3. Generate time series plots for each target variable.

'''

############################################ Params ############################################

rename_dict = {'Tailings_production': 'TP', 'Waste_rock_production': 'WRP', 'Ore_processed_mass': 'OP', 'Concentrate_production': 'CP'}
targets = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']

windows_to_eval = [10, 12]

randome_state = 42

############################################ Functions ############################################

def reindex_df(df, targets):
    ''' 
    Here we define a reindexed df that starts for every mine wiht the first year of production
    
    '''
    # Reindex the data with the start year of a mine

    df = df.copy()
    df.drop(columns='Unnamed: 0', inplace=True)
    df['Year'] = df.Year.astype(int)
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
    # Ensure proper reindexing
    df = reindex_df(df, targets)
    
    # Calculate the proportion of non-missing values for each `Prop_id`
    group_counts = df.groupby('Prop_id')[targets].count()
    group_missing = df.groupby('Prop_id')[targets].apply(lambda x: x.isna().sum())

    prop = group_counts.div(group_counts + group_missing)
    
    # Generate summary statistics
    summary = prop.describe().round(2)

    # rename
    summary = summary.rename(columns = rename_dict)
    # Save the summary to LaTeX
    df_to_latex(summary, 'missing_values_per_mine')
    
    return prop

def histplot_with_v_line(df):

    df = missing_values_per_mine(df, ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production'])

    # rename columns
    
    
    df_melt = df.melt(value_vars=['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production'], var_name='Target', value_name='Value')
    df_melt['Target'] = df_melt['Target'].replace(rename_dict)

    # histplot with .1, .25 and .3 lines
    p = (ggplot(df_melt, aes(x='Value', fill='Target'))
         + geom_histogram(bins=30, alpha = .6)
         + facet_wrap('~Target', scales='free')
         + geom_vline(xintercept=[.75, .5, .25], color='black')
         + theme_minimal()
         + scale_fill_brewer(type='qual', palette='Dark2')
        + theme(
             legend_position='bottom',  # Move legend to the bottom
            text = element_text(size= 14)  # Adjust axis tick label size
         )
         )
    

    save_fig_plotnine(p, 'histplot_missing_values_per_mine', w=12, h=8)
    p.draw()

    return None
    
def heatmap_missing_values_per_mine(df, targets):
    '''
    Assumptions:
    * Only for mines with at least ten values.
    * Only for years from 1950 onwards.
    '''
    df = reindex_df(df, targets)
    df_indexed = df.set_index(['Prop_id', 'Year'])
    df_missing = df_indexed.isna().astype(int).reset_index()

    for t in targets:
        df_t = df_missing[['Year', 'Prop_id', t]]

        # Take only mines with at least ten values
        df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)

        # Ensure 'Prop_id' is categorical
        df_t['Prop_id'] = pd.Categorical(df_t['Prop_id'], categories=df_t['Prop_id'].unique(), ordered=True)

        # Filter for years from 1950 onwards and ensure 'Year' is numeric
        df_t = df_t[df_t.Year >= 1950]
        df_t['Year'] = df_t['Year'].astype(int)  # Ensure numeric type for Year

        # Pivot and map values
        df_t = df_t.pivot(index='Prop_id', columns='Year', values=t)
        df_t = df_t.fillna(-1).applymap(
            lambda x: "Not_applicable" if x == -1 else ("Missing" if x == 1 else "Value_present")
        )

        # Melt the data and ensure 'Year' is numeric
        melted = df_t.reset_index().melt(id_vars='Prop_id', var_name='Year', value_name='Category')
        melted['Year'] = melted['Year'].astype(int)  # Ensure numeric type for Year again

        # Create the heatmap
        p = (
            ggplot(melted)
            + geom_tile(aes(x='Year', y='Prop_id', fill='Category'), color='white')
            + scale_fill_manual(
                values={"Not_applicable": "white", "Missing": "#c51b7d", "Value_present": "#4d9221"}
            )
            + scale_x_continuous(breaks=range(1950, 2023, 10), limits=(1950, 2023), expand = (0, -1, 0, -1))         
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1, size=14),
                axis_title_x= element_text(size=14),
                axis_title_y= element_text(size=14),
                axis_ticks_major_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                legend_position='bottom',
                legend_text=element_text(size=14),
            )
        )

        # Save the plot
        save_fig_plotnine(p, f'heatmap_missing_values_{t}', w=14, h=10)

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

    res = res.astype(float).round(2)

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
        df_t['Year'] = df_t['Year'].astype(int)

         # Filter groups with more than 10 values
        df_t = df_t.groupby('Prop_id').filter(lambda x: x[t].count() > 10)

        # Sample random 40 mines
        df_t = df_t[df_t['Prop_id'].isin(df_t['Prop_id'].sample(40, random_state=randome_state))]
    
        for w in windows_to_eval: 
            df_roll = rolling_mean_outlier_loop(df_t, t, targets, window_size=w, threshold=threshold)

            df_cont = add_mine_context(df_roll)
            # scale to kt
            df_cont[t] = df_cont[t] / 10**3
            # Plot time series with outliers

            # years to int
            df_cont['Year'] = df_cont['Year'].astype(int, errors='ignore')


            p = (
                ggplot(df_cont, aes(x='Year', y=t, color=f'{t}_outlier'))
                + geom_point(size = 2.5)
                + geom_smooth(method='loess', se=True, color = '#1b7837')
                + theme_minimal()
                + labs(x='Year', y=f'{rename_dict[t]} (kt)', color=f'{rename_dict[t]} Outlier')
                + facet_wrap('~Prop_name', scales='free', ncol=5)
                + theme(legend_position='bottom')

            )

            # add color map red outlier non black
            p = p + scale_color_manual(values={True: '#c51b7d', False: '#4d4d4d'})


            save_fig_plotnine(p, f'time_series_{t}_window_{w}_outliers', w=12, h=16)

    return None

def outlier_detection_and_ouput(df, targets, 
                                window_dict = {'Tailings_production': 10, 'Waste_rock_production': 10,
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

    df_to_csv_int(df, 'target_sample_with_outlier_detection_trans')
    return df
    



def main():


    
    path = r'data\int\D_target_prio_prep\target_vars_prio_source_trans.csv'

    prod = pd.read_csv(path)
    
    plot_time_series_target_outliers(prod, targets, threshold=2)

    
    return None



if __name__ == '__main__':
    main()