'''

This script is used to explore the sample for ws and t.

Explored:

1. Multicollinearity
2. Distribution of the target variable
3. Distribution of the features

'''
import numpy as np
import pandas as pd
import geopandas as gpd
from plotnine import *
import seaborn as sns


from util import save_fig_plotnine, data_to_csv_int, df_to_gpkg, save_fig

def multi_cor(df, vars = None):

    if vars is not None:
        df = df[vars]

    df[['Tailings_production', 'Concentrate_production', 'Area_mine']]  = df[['Tailings_production', 'Concentrate_production', 'Area_mine']].apply(np.log10)
    
    df.drop(['geometry', 'data_source', 'id_data_source'], axis=1, inplace=True)
    df.set_index('Prop_id', inplace=True)

    # drop columns where the sum of the columns is 0 
    df = df.loc[:, (df.sum(axis=0) != 0)]

    corr = df.corr().round(2)
    
    # Convert index and columns to strings
    corr.index = corr.index.astype(str)
    corr.columns = corr.columns.astype(str)
    
    # Melt the correlation matrix
    corr_melted = corr.reset_index().melt(id_vars='index', var_name='variable', value_name='correlation')
    
    # Ensure 'index' and 'variable' are treated as categories
    corr_melted['index'] = pd.Categorical(corr_melted['index'])
    corr_melted['variable'] = pd.Categorical(corr_melted['variable'])
    
    # Create the heatmap
    plot = (
        ggplot(corr_melted, aes(x='index', y='variable', fill='correlation'))
        + geom_tile(color="white", size=0.1)
        + scale_fill_gradient2(low="#276419", mid="white", high='#8e0152', midpoint=0)
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            axis_title=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position="right"
        )
        + coord_fixed()
        + geom_text(aes(label='correlation'), size=4, color="black")
    )
        
    # Save the plot
    save_fig_plotnine(plot, 'correlation_matrix_heatmap.png', w=14, h=14)
    return None

def hist_per_variable(df, vars = None):
    if vars is not None:
        df = df[vars]

    df[['Tailings_production', 'Concentrate_production', 'Area_mine', 'Count']]  = df[['Tailings_production', 'Concentrate_production', 'Area_mine', 'Count']].apply(np.log10)

    df.drop(['geometry', 'data_source', 'id_data_source'], axis=1, inplace=True)
    df.set_index('Prop_id', inplace=True)

    # drop columns where the sum of the columns is 0 
    df = df.loc[:, (df.sum(axis=0) != 0)]

    # Melt the DataFrame
    df_melted = df.reset_index().melt(id_vars='Prop_id', var_name='variable', value_name='value')
    
    # Ensure 'Prop_id' and 'variable' are treated as categories
    df_melted['Prop_id'] = pd.Categorical(df_melted['Prop_id'])
    df_melted['variable'] = pd.Categorical(df_melted['variable'])
    
    # Create the histogram
    plot = (
        ggplot(df_melted, aes(x='value', fill='variable'))
        + geom_histogram(bins=30, color='black', alpha=0.5)
        + facet_wrap('~variable', scales='free')
        + theme_minimal()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            axis_title=element_blank(),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
            legend_position="right"
        )
    )

    # put legend off
    plot += theme(legend_position='none')
    
    # Save the plot
    save_fig_plotnine(plot, 'histogram_per_variable.png', w=14, h=14)
    return None


def main():
    tpath = r'data\int\D_train_tetst_df\tailings.gpkg'
    t = gpd.read_file(tpath)
    hist_per_variable(t)

    return None



if __name__ == '__main__':
    main()