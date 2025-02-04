import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from plotnine import *

from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis



from M_prod_model import hubbert_model, hubbert_L_restrict, power_law, femp, prep_data
from D_load_werner import merge_werner
from util import get_path, save_fig, save_fig_plotnine, df_to_latex

############################################ Purpose ############################################
'''
This script contains functions to analyze the results of the production model fits.

'''

############################################ Params ############################################

rename_dict = {'Tailings_production': 'TP', 'Waste_rock_production': 'WRP', 'Ore_processed_mass': 'OP', 'Concentrate_production': 'CP'}

randome_state = 42
sig = 0.05

############################################ Functions ############################################


def model_analytics_hist(data, v):
    '''
    Function to create a facet grid of histograms for each target variable and model
    for a given variable of interest using plotnine, with optional log scale and automatic y-axis scaling.
    '''

    if v == 'RMSE':
        data[v] = data[v] / 10**6 # convert to Mt

    color_dict = {'femp': '#e08214', 'hubbert': '#542788'}


    # Get sample size per target var
    sample_size = data.groupby(['Target_var', 'Model']).size().reset_index(name='Sample_size')
    sample_size['Target_var'] = sample_size['Target_var'].replace(rename_dict)


    data['Target_var'] = data['Target_var'].replace(rename_dict)

    # add sample size to the Target var string
    data = pd.merge(data, sample_size, on=['Target_var', 'Model'])

    data['Target_var'] = data['Target_var'] + ' (n=' + data['Sample_size'].astype(str) + ')'

    

    # Base plot setup
    plot = (
        ggplot(data, aes(x=v, fill='Model'))
        + geom_histogram(bins=20, alpha=0.7, position="identity")
        + facet_wrap('~Target_var', nrow=2, scales='free')
        + labs(x=v, y='Frequency')
        + theme_minimal()
        + theme(subplots_adjust={'wspace': 0.3, 'hspace': 0.3}, text=element_text(size=14), legend_position='bottom')
        + scale_fill_manual(values=color_dict)
        )

    if v == 'RMSE':
        plot += labs(x=v + ' (Mt)')
        
    

    # Save and draw the plot
    save_fig_plotnine(plot, f'{v}_prodmod_facet_by_model_trans.png', w= 14, h=10)
    
    return None

def summarize_results(data):
    '''
    Function to summarize the results of the production model fits.
    '''
    # Group the data by target variable and model, and calculate the mean and standard deviation of the R^2 and RMSE values
    summary = (
    data.groupby(['Target_var', 'Model'])[['R2', 'RMSE', 'NRMSE']]
    .agg(['mean', 'std', 'median', skew, kurtosis])
    .reset_index()
    )

    new_columns = ['_'.join(col).strip() for col in summary.columns[2:].values]

    # Reassign the columns after index 2 with the new names
    summary.columns = ['Target_var', 'Model'] + new_columns

    p_values = data.groupby(['Target_var', 'Model'])[['P1_pval', 'P2_pval', 'P3_pval']].agg(lambda x: (x < sig).sum() / len(x))

    p_values = p_values.reset_index()
    # Merge the summary statistics with the p-values
    summary = pd.merge(summary, p_values, on=['Target_var', 'Model']).set_index('Target_var')

    summary[summary.Model.isin(['femp'])]['P3_pval'] = np.nan

    sample_sizes = data.groupby(['Target_var', 'Model']).size().reset_index(name='Sample_size')
    
    summary = pd.merge(summary, sample_sizes, on=['Target_var', 'Model']).set_index('Target_var')

    # reset index, invert and make multi column target ar and model
    summary = summary.reset_index().set_index(['Target_var', 'Model']).T
    # get two digit decimals
    summary = summary.round(2)

    df_to_latex(summary, 'model_summary', multicolumn=True)
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
        + scale_fill_brewer(type='qual', palette='Set2')
        )

    save_fig_plotnine(plot, 'p_val_significance.png', w=8, h=6)
    plot.draw()
    
    return plot


def plot_errors(data_records):
    data_records['Prop_id'] = data_records['Prop_id'].astype(int)

    for t in rec.Target_var.unique():
        plot = (
            ggplot(data_records[data_records['Target_var'] == t], aes(x='Observed', y='Predicted', color = 'Class'))
            + geom_point(alpha= 0.5)
            + labs(x='Observed log(t)', y='Predicted log(t)')
            + facet_wrap('~Model', scales='free')
            + theme_minimal()
            + scale_color_brewer(type='qual', palette='Set2')
            )
        # add the 1:1 line
        plot += geom_abline(intercept=0, slope=1, linetype='dashed', color='black')

        # x y log
        plot += scale_x_log10()
        plot += scale_y_log10()


        #log transformed 10% deviation lines
        plot += geom_abline(intercept=0, slope=.9, linetype='dashed', color='black')
        plot += geom_abline(intercept=0, slope=1.1, linetype='dotted', color='black')


        save_fig_plotnine(plot, f'{t}_error_plot.png', w=10, h=8)
        plot.draw()

    return None


def error_time_series(data_records):
    data_records['Prop_id'] = data_records['Prop_id'].astype(str)

    data_records = data_records[data_records['Year'] > 1950]

    for t in rec.Target_var.unique():
        data = data_records[data_records['Target_var'] == t]
        data['Error'] = data['Predicted'] - data['Observed']
        data['Error'] = StandardScaler().fit_transform(data['Error'].values.reshape(-1, 1))


        plot = (
            ggplot(data, aes(x='Year', y='Error', color = 'Class'))
            + geom_point(alpha= 0.5)
            + labs(x='Year', y='Error standardized')
            + facet_wrap('~Model', scales='free')
            + theme_minimal()
            + scale_fill_brewer(type='qual', palette='Set2')
            )

        save_fig_plotnine(plot, f'{t}_error_time_series_plot.png')
        plot.draw()

    return None


def identify_significant_model(modelres, sig=0.05):
    """
    Classify model results based on parameter significance.
    
    Class:
    - "FH" = Both femp and hubbert parameters are significant
    - "F" = Only femp parameters are significant
    - "H" = Only hubbert parameters are significant
    - "N" = None of the parameters are significant
    """

    # Create boolean masks for significance
    modelres['femp_significant'] = (
        (modelres['Model'] == 'femp') &
        (modelres[['P1_pval', 'P2_pval']] < sig).all(axis=1)
    )
    modelres['hubbert_significant'] = (
        (modelres['Model'] == 'hubbert') &
        (modelres[['P1_pval', 'P2_pval', 'P3_pval']] < sig).all(axis=1)
    )

    # Aggregate significance per group
    agg = modelres.groupby(['Prop_id', 'Target_var']).agg({
        'femp_significant': 'any',
        'hubbert_significant': 'any'
    }).reset_index()

    # Determine classification based on significance
    conditions = [
        agg['femp_significant'] & agg['hubbert_significant'],
        agg['femp_significant'],
        agg['hubbert_significant']
    ]
    choices = ['FH', 'F', 'H']
    agg['Class'] = np.select(conditions, choices, default='N')

    # Merge classifications back to the original DataFrame
    modelres = modelres.merge(agg[['Prop_id', 'Target_var', 'Class']], on=['Prop_id', 'Target_var'])

    # rmse & r2 comparison
    comp = modelres.groupby(['Prop_id', 'Target_var']).apply(lambda x: x[x.Model =='hubbert']['RMSE'].values[0]>x[x.Model =='femp']['RMSE'].values[0])

    # Drop intermediate columns
    modelres = modelres.drop(columns=['femp_significant', 'hubbert_significant'])

    return modelres


def identify_cum_model(modelres):
    """
    Classify model results based on parameter significance, RMSE, and R2.
    Prefer Hubbert model in case of a trade-off between RMSE and R2.
    """
    # Create boolean masks for significance
    modelres['femp_significant'] = (
        (modelres['Model'] == 'femp') &
        (modelres[['P1_pval', 'P2_pval']] < sig).all(axis=1)
    )
    modelres['hubbert_significant'] = (
        (modelres['Model'] == 'hubbert') &
        (modelres[['P1_pval', 'P2_pval', 'P3_pval']] < sig).all(axis=1)
    )

    # Aggregate significance per group
    agg = modelres.groupby(['Prop_id', 'Target_var']).agg({
        'femp_significant': 'any',
        'hubbert_significant': 'any',
        'RMSE': lambda x: x.iloc[0] if len(x) == 1 else x.iloc[1] - x.iloc[0],
        'R2': lambda x: x.iloc[0] if len(x) == 1 else x.iloc[1] - x.iloc[0]
    }).reset_index()

    # Determine classification based on significance, RMSE, and R2
    def classify(row):
        if row['femp_significant'] and row['hubbert_significant']:
            if row['RMSE'] < 0 and row['R2'] > 0:
                return 'F'  # femp has lower RMSE and higher R2
            elif row['RMSE'] > 0 and row['R2'] < 0:
                return 'H'  # Hubber has lower RMSE and higher R2
            else:
                return 'H'  # Prefer Hubbert in case of trade-off
        elif row['femp_significant']:
            return 'F'
        elif row['hubbert_significant']:
            return 'H'
        else:
            return 'N'

    agg['Class'] = agg.apply(classify, axis=1)

    return agg


def class_bar_chart(modelres):
    '''
    Create a bar chart showing the class shares differentiated by Target_var.

    Class Descriptions:
    - "F": Only femp parameters are significant
    - "H": Only hubbert parameters are significant
    - "N": None of the parameters are significant
    '''
    
    # Group by Target_var and Class, and calculate the share of each class
    class_share = (
        modelres.groupby(['Target_var', 'Class']).size()
        .reset_index(name='Count')
    )
    class_total = class_share.groupby('Target_var')['Count'].transform('sum')
    class_share['Share'] = class_share['Count'] / class_total
    class_share['Percentage'] = (class_share['Share'] * 100).round(1).astype(str) + '%'  # Convert to percentage


    # rename target var
    class_share['Target_var'] = class_share['Target_var'].replace(rename_dict)


    class_share['Share'] = class_share['Share'] * 100

    # Create the bar chart using plotnine
    plot = (
        ggplot(class_share, aes(x='Target_var', y='Share', fill='Class'))
        + geom_bar(stat='identity', position='stack', alpha = 0.7)
        + theme_minimal()
        + scale_fill_brewer(type='qual', palette='Dark2', name='Class')
        + theme(legend_position='bottom')
        + labs(x='Target Variable', y='Percentage', fill='Class')
        + theme(axis_text_x=element_text(size = 8),
                axis_text_y=element_text(size = 8),
                axis_title_x=element_text(size = 8),
                axis_title_y=element_text(size = 8),
                legend_title=element_text(size = 8),
                legend_text=element_text(size = 8))
    )

    # Include value per category
    plot += geom_text(aes(label='Percentage'), position=position_stack(vjust=0.5), size=8)

    # Save the plot (optional)
    save_fig_plotnine(plot, 'class_bar_chart.png')
    plot.draw()

    return plot

def box_plot(data, v):
    '''
    Create a box plot showing the distribution of R^2 values for each Target_var and Model.
    '''
    
    # Create the box plot using plotnine
    plot = (
        ggplot(data, aes(x='Target_var', y=v, fill='Class'))
        + geom_boxplot()
        + theme_minimal()
        + scale_fill_brewer(type='qual', palette='Dark2', name='Class')
    )
    if v in ['RMSE']:
        plot += labs(x=v + 'log(t)')
        plot += scale_x_log10()



    # Save the plot (optional)
    save_fig_plotnine(plot, f'{v}_box_plot.png')
    plot.draw()

    return plot


def sample_size_box(modelres):

    subset = modelres[['Target_var',  'Sample_size', 'Sample_']]

    subset.rename(columns={'Sample_size_train': 'Train', 'Sample_size_test': 'Test'}, inplace=True)

    # one sample size column and another with type
    subset = subset.melt(id_vars='Target_var', var_name='Sample_type', value_name='Sample_size', value_vars = ['Train', 'Test'])


    plot = (
        ggplot(subset, aes(x='Target_var', y='Sample_size', fill='Sample_type'))
        + geom_boxplot()
        + theme_minimal()
        + scale_fill_brewer(type='qual', palette='Set2', name='Class')
    )

    save_fig_plotnine(plot, 'sample_size_box_plot.png')
    plot.draw()


    return None


#################################################### Main ########################################################

def main_class_bar_chart(modelres):
    classified = identify_cum_model(modelres)
    class_bar_chart(classified)

def main_summarize_results(modelres):
    summarize_results(modelres)

if __name__ == '__main__':
    sig = .05    
    modelres = pd.read_json(r'data\int\production_model_fits_trans.json')
    rec = pd.read_csv(r'data\int\data_records.csv')
    main_summarize_results(modelres)


    
   