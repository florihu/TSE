import pandas as pd
from plotnine import *

from util import save_fig_plotnine , save_fig, df_to_latex
import scipy.stats as stats

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import zscore


def hist_per_target(data, targets):

    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']]
    stacked = subset.stack().reset_index()
    stacked.rename(columns={'level_1': 'Variable', 0: 'Value'}, inplace=True)
    stacked['Value'] = stacked['Value'] / 10**6 # Convert to Mt

    plot = (ggplot(stacked, aes(x='Value', fill='Variable'))
            + geom_histogram(bins=30, alpha=0.6)
            + facet_wrap('~Variable', scales='free')
            + scale_fill_brewer(type='qual', palette='Set2')
            + theme_minimal()
            + labs(x='Value (Mt)', y='Count')
    )

    save_fig_plotnine(plot, 'hist_per_target.png', w=12, h=10)
    plot.draw()
    return None

def qqplot_plotnine(data, targets):
    # Log-transform the data for the target variables
    subset = data[["Prop_id"] + targets].copy()
    melted_data = (
        subset.set_index("Prop_id")
        .apply(np.log)
        .reset_index()
        .melt(id_vars="Prop_id", var_name="Variable", value_name="Log_Value")
    )

    # Create the QQ plot
    plot = (
        ggplot(melted_data, aes(sample="Log_Value"))
        + stat_qq()  # Add QQ points
        + stat_qq_line(color = 'red')  # Add QQ line
        + facet_wrap("~Variable", scales="free")  # Facet by variable
        + labs(
            title="QQ Plots of Log-Transformed Variables",
            x="Theoretical Quantiles",
            y="Sample Quantiles",
        )
        + theme(subplots_adjust={"wspace": 0.3, "hspace": 0.3})
        + theme_minimal()
    )
    save_fig_plotnine(plot, "qqplot_plotnine.png", w=12, h=10)
    plot.draw()


def qqplot(data, targets):
    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, targets]
    subset = np.log(subset)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for i, target in enumerate(targets):
        row = i // 2
        col = i % 2
        # Generate QQ plot data
        qq_data = stats.probplot(subset[target], dist="norm")
        
        # Scatter plot for QQ points
        ax[row, col].scatter(qq_data[0][0], qq_data[0][1], c='blue', label="Data")
        
        # Regression line
        slope, intercept = qq_data[1][0], qq_data[1][1]
        ax[row, col].plot(qq_data[0][0], slope * qq_data[0][0] + intercept, color="red", label="Fit Line")
        
        # Add title and legend
        ax[row, col].set_title(f'{target}')
        ax[row, col].legend()

    # Save and display the plot
    plt.tight_layout()
    plt.savefig('qq_plots.png')
    plt.show()

def test_log_normality(data, targets):
    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, targets]
    subset = np.log(subset)

    var = []
    ps = []

    for target in targets:

        dropna = subset[target].dropna()
        stat, p = stats.shapiro(dropna)
        var.append(target)
        ps.append(p)

    df = pd.DataFrame({'Variable': var, 'p-value': ps})
    df_to_latex(df, 'shapiro_test')

def outlier_ident_zscore(data, targets):
    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, targets]
    subset = np.log(subset)

    var = []
    outliers = []

    for target in targets:

        dropna = subset[target].dropna()
        z = np.abs(zscore(dropna))
        outliers.append(len(z[z > 3]))
        var.append(target)

    df = pd.DataFrame({'Variable': var, 'Outliers': outliers})
    

    return None

def outlier_flag_per_target(data, targets):
    # Log transform the data subset
    data_indexed = data.set_index(['Prop_id','Year'])
    subset = np.log(data_indexed[targets])

    # Initialize outlier flags as False for all entries
    outlier_flags = pd.DataFrame(False, index=subset.index, columns=[f"{target}_outlier" for target in targets])

    # Calculate z-scores and flag outliers
    z_scores = np.abs(zscore(subset, nan_policy='omit'))
    filter = z_scores > 3
    outlier_flags.iloc[:, :] = filter

    merge = pd.merge(data_indexed, outlier_flags, left_index=True, right_index=True)

    # Generate outlier statistics summary
    outlier_stats = pd.DataFrame({
        'Variable': targets,
        'Outliers': outlier_flags.sum()
    })

    # Output the outlier stats as needed
    df_to_latex(outlier_stats, 'outliers_zscore')

    return merge


def time_series_plus_outlier_mask (data, targets):

    data_out = outlier_flag_per_target(data, targets)

    
    subset = data_out.loc[:, ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production' ]]
    stacked = subset.stack().reset_index()
    stacked.rename(columns={'level_2': 'Variable', 0: 'Value'}, inplace=True)
    stacked['Value'] = stacked['Value'] / 10**6 # Convert to Mt

    # add outlier flag
    outlier_flags = data_out.loc[:, ['Tailings_production_outlier', 'Waste_rock_production_outlier', 'Ore_processed_mass_outlier', 'Concentrate_production_outlier']]
    outlier_flags = outlier_flags.stack().reset_index()
    outlier_flags.rename(columns={'level_2': 'Variable', 0: 'Outlier'}, inplace=True)

    # remove outlier grom name 
    outlier_flags['Variable'] = outlier_flags['Variable'].str.replace('_outlier', '')
    
    merge = pd.merge(stacked, outlier_flags, on=['Prop_id', 'Year', 'Variable'])
    # prop id to categorical
    merge['Prop_id'] = merge['Prop_id'].astype('category') 
    
    plot = (ggplot(merge, aes(x='Year', y='Value', color='Prop_id'))
            + geom_point()
            + facet_wrap('~Variable', scales='free')
            + theme_minimal()
            + labs(x='Year', y='Value (Mt)')
            + scale_y_log10()
    )
    # dont show legend
    plot = plot + theme(legend_position='none')


    save_fig_plotnine(plot, 'time_series.png', w=12, h=10)
    plot.draw()

    return None


def main():
    
    path = r'data\int\D_build_sample_sets\target_vars_prio_source.csv'
    data = pd.read_csv(path)
    targets = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    time_series_plus_outlier_mask(data, targets)
    
    return None



if __name__ == '__main__':
    main()