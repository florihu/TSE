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

    stacked['Prop_id'] = stacked['Prop_id'].astype('category')

    plot = (ggplot(stacked, aes(x='Value', fill='Prop_id'))
            + geom_density(alpha=0.5)
            + facet_wrap('~Variable', scales='free')
            + theme_minimal()
            + labs(x='Value (Mt)', y='Count')
            + theme(legend_position='none')  # Exclude the legend
    )   

    save_fig_plotnine(plot, 'hist_per_target.png', w=12, h=10)
    plot.draw()
    return None


def boxplot_per_target(data, targets):
    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']]
    stacked = subset.stack().reset_index()
    stacked.rename(columns={'level_1': 'Variable', 0: 'Value'}, inplace=True)

    plot = (ggplot(stacked, aes(x='Variable', y='Value', fill='Variable'))
            + geom_boxplot()
            + theme_minimal()
            + labs(x='Variable', y='Value log(t)')
            + scale_y_log10()
            + scale_fill_brewer(type='qual', palette='Set2')
    )

    save_fig_plotnine(plot, 'boxplot_per_target.png', w=12, h=10)
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

def test_normality(data, targets, sig = .05):
    data_indexed = data.set_index('Prop_id')
    subset = data_indexed.loc[:, targets]
    

    var = []
    ps = []
    tests = []
    ids = []
    sigs = []


    subset_g = subset.groupby('Prop_id')

    for id, group in subset_g:
        for target in targets:
            for t in [stats.shapiro, stats.kstest]:
                dropna = group[target].dropna()
                
                if len(dropna) < 10:
                    continue

                if t == stats.kstest:
                    # Perform the Kolmogorov-Smirnov test assuming a normal distribution
                    stat, p = t(dropna, 'norm')  # Specify the CDF (e.g., 'norm' for normal distribution)
                else:
                    # Perform the Shapiro-Wilk test
                    stat, p = t(dropna)
                
                ids.append(id)
                var.append(target)
                tests.append(t.__name__)
                ps.append(p)
                sigs.append(p > sig)
      
    df = pd.DataFrame({'Prop_id': ids, 'Variable': var, 'p-value': ps, 'Test': tests, 'Normal_distrib': sigs })
     # Format the p-value column in scientific notation
    df['p-value'] = df['p-value'].apply(lambda x: f"{x:.2e}")

    res = df.groupby(['Variable', 'Test'])['Normal_distrib'].agg(['sum', 'count']).reset_index()
    res.rename({'sum': 'Normal_distrib', 'count': 'Total'}, inplace=True)
    # Export to LaTeX
    df_to_latex(res, 'normal_test_per_mine')


def outlier_ident_iqr(data, targets):
    """
    Identify outliers using the IQR method for mines with more than 10 values.
    Store the results in separate columns for each target variable.
    """
    # Create a copy of the data to avoid modifying the original dataframe
    data = data.copy()
    
    # Group data by 'Prop_id' and apply outlier detection per group
    def compute_outliers(group):
        result = {}
        for target in targets:
            if group[target].notna().sum() < 8:
                result[f'{target}_outlier'] = [False] * len(group)
                continue

            q1 = group[target].quantile(0.25)
            q3 = group[target].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            result[f'{target}_outlier'] = np.where(
                (group[target] < lower_bound) | (group[target] > upper_bound), True, False
            )
        return pd.DataFrame(result, index=group.index)

    # Apply outlier detection and merge results back
    outlier_flags = data.groupby('Prop_id').apply(compute_outliers).reset_index(level=0, drop=True)
    data = pd.concat([data, outlier_flags], axis=1)

    return data


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
    data = data.reset_index()
    data['Year'] = pd.to_datetime(data['Year']).dt.year
    data.set_index(['Prop_id', 'Year'], inplace=True)

    subset = data.loc[:, ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production' ]]
    stacked = subset.stack().reset_index()
    stacked.rename(columns={'level_2': 'Variable', 0: 'Value'}, inplace=True)
    stacked['Value'] = stacked['Value'] / 10**6 # Convert to Mt

    # add outlier flag
    outlier_flags = data.loc[:, ['Tailings_production_outlier', 'Waste_rock_production_outlier', 'Ore_processed_mass_outlier', 'Concentrate_production_outlier']]
    outlier_flags = outlier_flags.stack().reset_index()
    outlier_flags.rename(columns={'level_2': 'Variable', 0: 'Outlier'}, inplace=True)

    # remove outlier grom name 
    outlier_flags['Variable'] = outlier_flags['Variable'].str.replace('_outlier', '')
    
    merge = pd.merge(stacked, outlier_flags, on=['Prop_id', 'Year', 'Variable'])
    # prop id to categorical
    merge['Prop_id'] = merge['Prop_id'].astype('category') 

    # select the 40 mines with most counts
    top_49 = merge['Prop_id'].value_counts().nlargest(49).index

    merge = merge[merge['Prop_id'].isin(top_49)]

    merge = merge[merge['Year']> 1950]

    
    for t in targets:
        plot = (ggplot(merge[merge['Variable'] == t], aes(x='Year', y='Value'))
                + geom_point()
                + facet_wrap('~Prop_id', scales='free')
                + theme_minimal()
                + labs(x='Year', y='Value (Mt)')
    
        )
        # I want to add now red outlier points
        plot = plot + geom_point(merge[(merge['Outlier'] == True) & (merge['Variable'] == t)], aes(x='Year', y='Value'), color='red', size=1)

        # dont show legend
        plot = plot + theme(legend_position='none')

        save_fig_plotnine(plot, f'{t}_time_series.png', w=18, h=12)
        plot.draw()

    return None


def error_plots_time_series(data, targets):

    sample_49 = np.random.choice(data['Prop_id'].unique(), size=49, replace=False)
    data = data[data['Prop_id'].isin(sample_49)]
    data['Residual'] = data['Residual'] / 10**6 # Convert to Mt
    
    for t in targets:
        for m in data.Model.unique():
            plot = (ggplot(data[(data['Target_var'] == t) & (data['Model'] == m)], aes(x='Year', y='Residual'))
                        + geom_point()
                        + facet_wrap('~Prop_id', scales='free')
                        + theme_minimal()
                        + labs(x='Year', y='Value (Mt)')
                )
            
            # dont show legend
            plot = plot + theme(legend_position='none')
            
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 10:
                plot = plot + geom_smooth()

            save_fig_plotnine(plot, f'{t}_{m}_error_time_series.png', w=18, h=12)

    return None


def pred_obs_ts(data, targets):

    sample_49 = np.random.choice(data['Prop_id'].unique(), size=49, replace=False)
    data = data[data['Prop_id'].isin(sample_49)]
    

    df_melt = data.melt(id_vars=['Prop_id', 'Year', 'Target_var', 'Model'], value_vars=['Observed', 'Predicted'], var_name='Type', value_name='Value')

    df_melt['Value'] = df_melt['Value'] / 10**6 # Convert to Mt
    for t in targets:
        for m in data.Model.unique():
            plot = (ggplot(df_melt[(df_melt['Target_var'] == t) & (df_melt['Model'] == m)], aes(x='Year', y='Value', color = 'Type'))
                        + geom_point()
                        + facet_wrap('~Prop_id', scales='free')
                        + theme_minimal()
                        + labs(x='Year', y='Value (Mt)')
                        
                )
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 10:
                plot = plot + geom_smooth()
    
            
            save_fig_plotnine(plot, f'{t}_{m}_obs_pred_time_series.png', w=18, h=12)

    return None



def error_vs_obs(data, targets):
    sample_49 = np.random.choice(data['Prop_id'].unique(), size=49, replace=False)
    data = data[data['Prop_id'].isin(sample_49)]


    data['Error'] = (data['Observed'] - data['Predicted'] )/ 10**6 # Convert to Mt

    data['Observed'] = data['Observed'] / 10**6 # Convert to Mt

    data['Prop_id'] = data['Prop_id'].astype('category')

    for t in targets:
        for m in data.Model.unique():
            plot = (ggplot(data[(data['Target_var'] == t) & (data['Model'] == m)], aes(x='Observed', y='Error'))
                        + geom_point()
                        + facet_wrap('~Prop_id', scales='free')
                        + theme_minimal()
                        + labs(x='Observed (Mt)', y='Error (Mt)')
                        
                )
            
            # do geom smooth only if sample size is sufficient
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 10:
                plot = plot + geom_smooth()
            

            # dont show legend
            plot = plot + theme(legend_position='none')
            
            save_fig_plotnine(plot, f'{t}_{m}_error_vs_obs.png', w=18, h=12)

    return None


def obs_vs_pred(data, targets):
    # choose random 49 indexes
    sample_49 = np.random.choice(data['Prop_id'].unique(), size=49, replace=False)
    data = data[data['Prop_id'].isin(sample_49)]

    data['Observed'] = data['Observed'] / 10**6 # Convert to Mt
    data['Predicted'] = data['Predicted'] / 10**6 # Convert to Mt

    data['Prop_id'] = data['Prop_id'].astype('category')

    for t in targets:
        for m in data.Model.unique():
            plot = (ggplot(data[(data['Target_var'] == t) & (data['Model'] == m)], aes(x='Observed', y='Predicted'))
                        + geom_point()
                        + facet_wrap('~Prop_id', scales='free')
                        + theme_minimal()
                        + labs(x='Observed (Mt)', y='Predicted (Mt)')
                        
                )
            
            # do geom smooth only if sample size is sufficient
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 10:
                plot = plot + geom_smooth()
            
            # include 1: 1 line
            plot = plot + geom_abline(intercept=0, slope=1, color='red')
            
            # dont show legend
            plot = plot + theme(legend_position='none')
            
            save_fig_plotnine(plot, f'{t}_{m}_obs_vs_pred.png', w=18, h=12)

    return None


def main():
    
    path = r'data\int\D_build_sample_sets\target_vars_prio_source.csv'

    model_res_path = r'data\int\data_records.csv'
    data = pd.read_csv(path)
    targets = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    res = pd.read_csv(model_res_path)
    pred_obs_ts(res, targets)
    obs_vs_pred(res, targets)
    
    
    return None



if __name__ == '__main__':
    main()