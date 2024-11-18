import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util import get_path, save_fig, save_fig_plotnine, df_to_latex
from sklearn.preprocessing import StandardScaler
from plotnine import *

from matplotlib.ticker import FuncFormatter


modelres = pd.read_json('data\int\production_model_fits.json')

def model_analtics_facet(data, v):
    '''
    Function to create a facet grid of histograms for each target variable and model
    for a given variable of interest.
    
    '''
    # Set up the subplot grid
    f, ax = plt.subplots(2, 2, figsize=(8, 6))

    # Loop through each unique target variable to create a subplot
    for i, t in enumerate(data['Target_var'].unique()):
        # Filter data for the current target variable
        subset_data = data[data['Target_var'] == t]

        if i == 0 : legbool = True 
        else : legbool = False
        
        # Plot histogram for each target variable and model
        plot = sns.histplot(data=subset_data, x=v, hue='Model', bins=20, kde=True, ax=ax[i//2, i%2], legend=legbool, palette='viridis_r', alpha=0.6) 
        
        # Set titles and labels
        ax[i//2, i%2].set_title(f'{t}')
        ax[i//2, i%2].set_xlabel(v)
        ax[i//2, i%2].set_ylabel('Frequency')


    plt.tight_layout()
    save_fig(f'{v}_prodmod_anal_facet.png')
    plt.show()
    plt.close()

    return None


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
        + geom_histogram(bins=20, alpha=0.6, position="identity")
        + facet_wrap('~Target_var', nrow=2, scales='free_x')
        + labs(title=f'{v} Distribution by Target Variable and Model',
               x=v, y='Frequency')
        + theme_bw()
        + theme(subplots_adjust={'wspace': 0.25, 'hspace': 0.25})
        )
    
    if scale:
        plot += scale_x_continuous(labels=FuncFormatter(auto_scale_formatter))

    if v =='RMSE':
        plot += labs(x='RMSE (t)')
    

    # Save and draw the plot
    save_fig_plotnine(plot, f'{v}_prodmod_anal_facet.png')
    plot.draw()
    
    return None

def summarize_results(data):
    '''
    Function to summarize the results of the production model fits.
    '''
    # Group the data by target variable and model, and calculate the mean and standard deviation of the R^2 and RMSE values
    summary = data.groupby(['Target_var', 'Model'])[['R2', 'RMSE']].agg(['mean', 'std']).reset_index()

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

sig = .05

if __name__ == '__main__':
    summarize_results(modelres)