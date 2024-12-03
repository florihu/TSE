import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util import get_path, save_fig, save_fig_plotnine, df_to_latex
from sklearn.preprocessing import StandardScaler
from plotnine import *
from M_prod_model import hubbert_model, hubbert_L_restrict, power_law, femp, prep_data
from D_load_werner import merge_werner
from matplotlib.ticker import FuncFormatter


modelres = pd.read_json('data\int\production_model_fits.json')

def model_analytics_facet_plotnine(data, v, scale=False, unit = 't'):
    '''
    Function to create a facet grid of histograms for each target variable and model
    for a given variable of interest using plotnine, with optional log scale and automatic y-axis scaling.
    '''
    
    # Define a formatter function for automatic scaling
    def auto_scale_formatter(x, _):
        if np.max(x) > 1e9:
            return [f'{i / 1e9:.1f}G' for i in x]
        elif np.max(x) > 1e6:  
            return [f'{i / 1e6:.1f}M' for i in x]

    # Base plot setup
    plot = (
        ggplot(data, aes(x=v, fill='Model'))
        + geom_histogram(bins=20, alpha=0.8, position="identity")
        + facet_wrap('~Target_var', nrow=2, scales='free_x')
        + labs(x=v, y='Frequency')
        + theme_minimal()
        + theme(subplots_adjust={'wspace': 0.25, 'hspace': 0.25})
        )
    
    if scale:
        plot += scale_x_log10()

    if v in ['RMSE_train', 'RMSE_test']:
        plot += labs(x=v + f' ({unit})')
    

    # Save and draw the plot
    save_fig_plotnine(plot, f'{v}_prodmod_facet.png')
    
    return None

def summarize_results(data):
    '''
    Function to summarize the results of the production model fits.
    '''
    # Group the data by target variable and model, and calculate the mean and standard deviation of the R^2 and RMSE values
    summary = data.groupby(['Target_var', 'Model'])[['R2_train', 'RMSE_train', 'R2_test', 'RMSE_test']].agg(['mean', 'std']).reset_index()

    new_columns = ['_'.join(col).strip() for col in summary.columns[2:].values]

    # Reassign the columns after index 2 with the new names
    summary.columns = ['Target_var', 'Model'] + new_columns

    p_values = data.groupby(['Target_var', 'Model'])[['P1_pval', 'P2_pval', 'P3_pval']].agg(lambda x: (x < sig).sum() / len(x))

    p_values = p_values.reset_index()
    # Merge the summary statistics with the p-values
    summary = pd.merge(summary, p_values, on=['Target_var', 'Model']).set_index('Target_var')

    summary[summary.Model.isin(['femp', 'power_law'])]['P3_pval'] = np.nan

    sample_sizes = data.groupby(['Target_var', 'Model']).size().reset_index(name='Sample_size')
    
    summary = pd.merge(summary, sample_sizes, on=['Target_var', 'Model']).set_index('Target_var')

    df_to_latex(summary, 'model_summary')
    summary.to_csv(r'data\int\production_model_summary.csv')

    return summary

def plot_p_vals(modelres, sig=0.05):
    # Step 1: Calculate percentage of significant values
    significance_summary = (
        modelres.groupby(['Target_var', 'Model'])[['P1_pval', 'P2_pval', 'P3_pval']]
        .apply(lambda df: (df < sig).sum() / len(df))
        .reset_index()
        .melt(id_vars=['Target_var', 'Model'], var_name='Parameter', value_name='Significant_Percentage')
    )
    
    # Step 2: Add non-significant percentage
    significance_summary['Non_Significant_Percentage'] = 1 - significance_summary['Significant_Percentage']
    
    # Step 3: Reshape for plotting
    plot_data = significance_summary.melt(
        id_vars=['Target_var', 'Model', 'Parameter'],
        value_vars=['Significant_Percentage', 'Non_Significant_Percentage'],
        var_name='Significance',
        value_name='Percentage'
    )
    plot_data['Significance'] = plot_data['Significance'].replace({
        'Significant_Percentage': 'Significant',
        'Non_Significant_Percentage': 'Non-Significant'
    })

    plot_data = plot_data[(plot_data['Model'] == 'hubbert')]

    plot_data['Parameter'] = plot_data['Parameter'].replace({'P1_pval': 'L', 'P2_pval': 'k', 'P3_pval': 't0'})

    # Step 4: Create the plot
    plot = (
        ggplot(plot_data, aes(x='Parameter', y='Percentage', fill='Significance')) +
        geom_bar(stat='identity', position='stack') +
        facet_wrap('~Target_var', scales='free') +
        labs(
            x='Model',
            y='Percentage',
            fill='Significance'
        ) +
        theme_minimal() 
        )

    save_fig_plotnine(plot, 'p_val_significance.png', w=8, h=6)
    plot.draw()
    
    return plot

def error_analysis_pred(modelres):
    werner = merge_werner()
    werner_prep = prep_data(werner)
    

    return None


sig = .05

if __name__ == '__main__':
    plot_p_vals(modelres, sig)
    