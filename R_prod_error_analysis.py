import pandas as pd
from plotnine import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, norm, lognorm
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

from M_prod_model import hubbert_model, femp, femp_deriv, hubbert_deriv
from util import save_fig_plotnine , save_fig, df_to_latex
from D_sp_data_clean import get_data

import seaborn as sns

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
                        + labs(x='Year', y='Residual (Mt)')
                )
            
            # dont show legend
            plot = plot + theme(legend_position='none')
            
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 10:
                plot = plot + geom_smooth(color = '#b2182b')

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


    data['Residual'] = data['Residual'] / 10**6 # Convert to Mt
    data['Observed'] = data['Observed'] / 10**6 # Convert to Mt

    data['Prop_id'] = data['Prop_id'].astype('category')

    for t in targets:
        for m in data.Model.unique():
            plot = (ggplot(data[(data['Target_var'] == t) & (data['Model'] == m)], aes(x='Observed', y='Residual'))
                        + geom_point()
                        + facet_wrap('~Prop_id', scales='free')
                        + theme_minimal()
                        + labs(x='Observed (Mt)', y='Residual (Mt)')
                        
                )
            
            # do geom smooth only if sample size is sufficient
            if len(data[(data['Target_var'] == t) & (data['Model'] == m)]) > 15:
                # add geom smooth plus regularization
                plot = plot + geom_smooth(method= 'lm' ,color = '#b2182b')
            

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


def error_and_cum_prod_hubbert(res, targets):
    # select hubbert only
    hubbert = res[res['Model'] == 'hubbert']

    # get the hubbert evaluation for each target
    def hubbert_eval_func(group):
        years = np.arange(group['Year'].min(), group['Year'].max() + 1)
        evaluations = hubbert_model(years, group['P1_value'].iloc[0], group['P2_value'].iloc[0], group['P3_value'].iloc[0]) / 10**6 # convert to Mt
        return pd.DataFrame({'Year': years, 'Cum_prod': evaluations})

    hubbert_eval = hubbert.groupby(['Prop_id', 'Target_var']).apply(hubbert_eval_func).reset_index(level=[0,1])



    # plot the residual of the fitting and the cumulative production
    hubbert_eval = hubbert_eval.reset_index()

    sample_25 = np.random.choice(res['Prop_id'].unique(), size=25, replace=False)
    res = res[res['Prop_id'].isin(sample_25)]
    hubbert_eval = hubbert_eval[hubbert_eval['Prop_id'].isin(sample_25)]
    
    res['Residual'] = res['Residual'] / 10**6 # Convert to Mt

    for t in targets:

        plot = (ggplot(res[(res['Target_var'] == t)], aes(x='Year', y='Residual'))
                    + geom_point()
                    + facet_wrap('~Prop_id', scales='free')
                    + theme_minimal()
                    + labs(x='Year', y='Value (Mt)')
                    + geom_smooth(color = '#b2182b')
            )
        
        
        
        # include hubbert fit
        plot = plot + geom_line(hubbert_eval[(hubbert_eval['Target_var'] == t)], aes(x='Year', y='Cum_prod'), color='#2166ac')
        
        # show legend
        plot = plot + theme(legend_position='bottom')

        save_fig_plotnine(plot, f'{t}_hubbert_res_cum_prod', w=18, h=12)


    return None


def check_positive(x, replace = 1e-8):

    if x <= 0:
        return replace
    else:
        return x
def predict_femp(t, R0, C, R0_err, C_err, n_samples):
    """
    Monte Carlo sampling for confidence interval estimation of FEMP model.
    
    Parameters:
        t (np.array): Time
        R0 (float): Initial reserves
        C (float): Production-to-reserve ratio
        R0_err (float): Standard deviation of R0
        C_err (float): Standard deviation of C
        n_samples (int): Number of Monte Carlo samples
    
    Returns:
        tuple: Mean prediction, lower CI, upper CI
    """
    # Generate Monte Carlo samples for R0 and C

    # R0 is lognormal distributed

    
    R0_samples = norm.rvs(loc=check_positive(R0), scale=check_positive(R0_err), size=n_samples)
    C_samples = norm.rvs(loc=check_positive(C), scale=check_positive(C_err), size=n_samples)

    # Evaluate the FEMP model derivative for each sample
    predictions = [femp_deriv(t, R0_sample, C_sample) for R0_sample, C_sample in zip(R0_samples, C_samples)]

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    lower_ci = np.percentile(predictions, 2.5, axis=0)
    upper_ci = np.percentile(predictions, 97.5, axis=0)

    return mean_prediction, lower_ci, upper_ci

def predict_hubbert(t, L, k, t0, L_err, k_err, t0_err, n_samples):
    """
    Monte Carlo sampling for confidence interval estimation of Hubbert model.
    
    Parameters:
        t (np.array): Time
        L (float): Maximum production
        k (float): Growth rate
        t0 (float): Time of peak production
        L_err (float): Standard deviation of L
        k_err (float): Standard deviation of k
        t0_err (float): Standard deviation of t0
        n_samples (int): Number of Monte Carlo samples
    
    Returns:
        tuple: Mean prediction, lower CI, upper CI
    """
    # Generate Monte Carlo samples for L, k, and t0
    
    # L is log normal distributed


    L_samples = norm.rvs(loc=check_positive(L), scale=check_positive(L_err), size=n_samples)


    k_samples = norm.rvs(loc=check_positive(k), scale=check_positive(k_err), size=n_samples)
    
    #t0 is also lognormal distributd
    t0_samples = norm.rvs(loc=check_positive(t0), scale=check_positive(t0_err), size=n_samples)

    # Evaluate the Hubbert model derivative for each sample
    predictions = [hubbert_deriv(t, L_sample, k_sample, t0_sample) for L_sample, k_sample, t0_sample in zip(L_samples, k_samples, t0_samples)]


    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    lower_ci = np.percentile(predictions, 2.5, axis=0)
    upper_ci = np.percentile(predictions, 97.5, axis=0)

    return mean_prediction, lower_ci, upper_ci


# Apply to DataFrame
def calculate_prod_row(row):
    if row['Model'] == 'femp':
        return femp_deriv(row['Year'], row['P1_value'], row['P2_value'])
    elif row['Model'] == 'hubbert':
        return hubbert_deriv(row['Year'], row['P1_value'], row['P2_value'], row['P3_value'])
    
def calculate_stock_row(row):
    if row['Model'] == 'femp':
        return femp(row['Year'], row['P1_value'], row['P2_value'])
    elif row['Model'] == 'hubbert':
        return hubbert_model(row['Year'], row['P1_value'], row['P2_value'], row['P3_value'])



def obs_pred_femp_hubbert(df, modelres, targets, fig_manu = False):
    # Convert observed values in df to mega tonnes
    df['Observed'] = df['Observed'] / 1e6  # Assuming data is in tonnes
    df['Residual'] = df['Residual'] / 1e6  # Assuming data is in tonnes

    # Create a DataFrame with all possible years for each Prop_id, Target_var, and Model
    df_pred = (
        df.groupby(['Prop_id', 'Target_var', 'Model'])
        .apply(lambda x: pd.DataFrame({'Year': np.arange(0, x['Year'].max() + 1)}))
        .reset_index(level=[0, 1, 2])
    )


    # Merge predictions with results
    df_pred = df_pred.merge(modelres, on=['Prop_id', 'Target_var', 'Model'], how='left')

    # Calculate predicted values and confidence intervals
    df_pred['Prod_pred'] = df_pred.apply(calculate_prod_row, axis=1)

    df_pred['Prod_lower_95ci'] = df_pred['Prod_pred'] - 1.96 * df_pred['RMSE']
    df_pred['Prod_high_95ci'] = df_pred['Prod_pred'] + 1.96 * df_pred['RMSE']


    df_pred['Stock_pred'] = df_pred.apply(calculate_stock_row, axis=1)

    df_pred['Stock_lower_95ci'] = df_pred['Stock_pred'] - 1.96 * df_pred['RMSE']
    df_pred['Stock_high_95ci'] = df_pred['Stock_pred'] + 1.96 * df_pred['RMSE']

    # Replace non zero confidence intervals with zero
    df_pred.loc[df_pred['Prod_lower_95ci'] < 0, 'Prod_lower_95ci'] = 0
    df_pred.loc[df_pred['Stock_lower_95ci'] < 0, 'Stock_lower_95ci'] = 0


    #Convert to Mt
    df_pred['Prod_pred'] = df_pred['Prod_pred'] / 1e6
    df_pred['Prod_lower_95ci'] = df_pred['Prod_lower_95ci'] / 1e6
    df_pred['Prod_high_95ci'] = df_pred['Prod_high_95ci'] / 1e6

    df_pred['Stock_pred'] = df_pred['Stock_pred'] / 1e6
    df_pred['Stock_lower_95ci'] = df_pred['Stock_lower_95ci'] / 1e6
    df_pred['Stock_high_95ci'] = df_pred['Stock_high_95ci'] / 1e6

    df_pred = df_pred.merge(df[['Prop_id', 'Prop_name']], on='Prop_id', how='left')

    if fig_manu:
        w= 16
        h=12
        size = 12
        # Get the top 4 mines with the most data points for the current target variable
        mine_partial_names = ['Escondida','Charcas', 'Ramu']

        # get unique prop ids where prop name contains the partial name
        mine_select = df[df['Prop_name'].str.contains('|'.join(mine_partial_names))]['Prop_id'].unique()

        type = 'manu'

    else:   
        w= 24
        h=24
        size = 8
        # get random 49 prop ids
        mine_select = np.random.choice(df['Prop_id'].unique(), size=49, replace=False)
        type = 'explo'

    # Plot for each target variable
    for t in targets:
        # Filter data for the selected target
        df_pred_t = df_pred[
            (df_pred['Prop_id'].isin(mine_select)) & 
            (df_pred['Target_var'] == t) & 
            (df_pred['Model'] == 'hubbert')
        ]
        
        if df_pred_t.empty:
            continue

        df_t = df[
            (df['Prop_id'].isin(mine_select)) & 
            (df['Target_var'] == t) & 
            (df['Model'] == 'hubbert')
        ]

        # Prepare a unified DataFrame for plotting
        df_pred_t_melted = df_pred_t.melt(
            id_vars=['Year', 'Prop_name'], 
            value_vars=['Prod_pred', 'Prod_lower_95ci', 'Prod_high_95ci'], 
            var_name='Category', 
            value_name='Value'
        )
        df_pred_t_melted['Category'] = df_pred_t_melted['Category'].replace({
            'Prod_pred': 'Predicted',
            'Prod_lower_95ci': 'Lower CI',
            'Prod_high_95ci': 'Upper CI'
        })

        df_t_melted = df_t.melt(
            id_vars=['Year', 'Prop_name'], 
            value_vars=['Observed', 'Residual'], 
            var_name='Category', 
            value_name='Value'
        )
        df_t_melted['Category'] = df_t_melted['Category'].replace({
            'Observed': 'Observed',
            'Residual': 'Residual'
        })

        # Combine the DataFrames
        df_combined = pd.concat([df_pred_t_melted, df_t_melted], ignore_index=True)

        # Define colors for categories
        color_dict = {
            'Predicted': '#542788',
            'Lower CI': '#542788',
            'Upper CI': '#542788',
            'Observed': 'black',
            'Residual': '#c51b7d'
        }

        # Create the plot
        p = (
            ggplot(df_combined, aes(x='Year', y='Value', color='Category', fill='Category'))
            + geom_line(data=df_combined[df_combined['Category'] == 'Predicted'], size=1)
            + geom_ribbon(aes(ymin='Value', ymax='Value'),
                data=df_combined[df_combined['Category'].isin(['Lower CI', 'Upper CI'])], 
                alpha=0.2
            )
            + geom_point(
                data=df_combined[df_combined['Category'] == 'Observed'], 
                size=2, 
                shape='o'
            )
            + geom_point(
                data=df_combined[df_combined['Category'] == 'Residual'], 
                size=2, 
                shape='x'
            )
            + geom_smooth(
                data=df_combined[df_combined['Category'] == 'Residual'], 
                method='lm', 
                se=True, 
                alpha=0.2,
                linetype='dashed'
            )
            + facet_wrap('~Prop_name', scales='free', ncol=3)
            + labs(
                x='Year',
                y=f'{t} (Mt)',
                color='Category',
                fill='Category'
            )
            + theme_minimal()
            + scale_color_manual(values=color_dict)
            + scale_fill_manual(values=color_dict)
            + theme(
                legend_position='bottom', 
                text=element_text(size=size), 
                subplots_adjust={"wspace": 1, "hspace": 1}
            )
        )

        

        # Save the plot
        save_fig_plotnine(p, f'{t}_prod__obs_pred_{type}.png', w=w, h=h)

    



        # p1 = (
        #     ggplot(df_pred_t, aes(x='Year', y='Stock_pred', color='Model'))
        #     + geom_line()
        #     + geom_ribbon(aes(ymin='Stock_lower_95ci', ymax='Stock_high_95ci', fill='Model'), alpha=0.2)
        #     + facet_wrap('~Prop_name', scales='free')
        #     + labs(
        #         x='Year',
        #         y=f'{t} cumulative (Mt)',
        #     )
        #     + theme_minimal()
        #     + scale_color_manual(values=color_dict)
        #     + theme(legend_position='bottom', text=element_text(size=size), subplots_adjust={"wspace": 1, "hspace": 1})
        # )

        # if fig_manu:
        #     # legend off
        #     p1 = p1 + theme(legend_position='none')
        # # Save the plot
        # save_fig_plotnine(p1, f'{t}_stock__obs_pred_{type}.png', w=w, h=h)

def hubbert_k_vs_L(modelres, targets):

    for t in targets:
        # Extract the relevant subset of data
        df = modelres[(modelres['Target_var'] == t) & (modelres['Model'] == 'hubbert')]

        # Prepare the data
        df['L'] = np.log(df['P1_value'])
        df['k'] = df['P2_value']
        df['t0'] = df['P3_value']
        df['L_err'] = np.log(df['P1_err'])
        df['k_err'] = df['P2_err']
        df['t0_err'] = df['P3_err']

        plot = (
            ggplot(df, aes(x='k', y='L'))
            + geom_point()  # Scatter plot of k vs L
            + geom_smooth( method = 'lm', color='red')  # Linear regression lin
            + labs(x='k', y='L log(t)')  # Axis labels
            + theme_minimal()  # Minimal theme
        )
        #increase text
        plot = plot + theme(text=element_text(size=16))
        save_fig_plotnine(plot, f'{t}_hubbert_k_vs_L.png', w=12, h=10)

def add_mine_context(data):

    site = get_data('site')
    # construct a feature from prop name, primary commodity and Country
    site['Prop_name'] = site['Prop_name'].str.cat(site[['Primary_commodity', 'Country_name']], sep=', ')

    # merge site
    c_data = data.merge(site[['Prop_id', 'Prop_name']], on='Prop_id', how='left')

    return c_data

def main():
    
    path = r'data\int\D_build_sample_sets\target_vars_prio_source.csv'

    model_res_path = r'data\int\data_records.csv'
    data = pd.read_csv(path)
    targets = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    res = pd.read_csv(model_res_path)
    modelres = pd.read_json(r'data\int\production_model_fits.json')

    
    merge = add_mine_context(data)
    
    #modelres = modelres.merge(site[['Prop_id', 'Prop_name', 'Primary_commodity']], on='Prop_id', how='left')

    obs_pred_femp_hubbert(merge, modelres, targets, fig_manu = True)

    return None



if __name__ == '__main__':
    main()