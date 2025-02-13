import pandas as pd
from plotnine import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore, norm, lognorm
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import seaborn as sns
from itertools import product

from M_prod_model import hubbert_model, femp, femp_deriv, hubbert_deriv
from util import save_fig_plotnine , save_fig, df_to_latex
from D_sp_data_clean import get_data




#########################################Parameters##############################################

np.random.seed(42)

# model color dict

model_color = {'femp':'#e08214', 'hubbert': '#542788'}

ids_manuscript = [28174, 33643,  26634]


prod_rename_dict = {'Tailings_production': 'TP', 'Waste_rock_production': 'WRP', 'Ore_processed_mass': 'OP', 'Concentrate_production': 'CP'}

cumprod_rename_dict = {'Tailings_production': 'CTP', 'Waste_rock_production': 'CWRP', 'Ore_processed_mass': 'COP', 'Concentrate_production': 'CCP'}

rename_prop_name = {'Neves-Corvo, Copper, Portugal': 'Type 2: Neves-Corvo, Copper, Portugal',
                    'Broken Hill, Zinc, Australia': 'Type 1: Broken Hill, Zinc, Australia',
                    'Lanfranchi, Nickel, Australia': 'Type 3: Lanfranchi, Nickel, Australia'}

#########################################Functions################################################

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


def get_random_ids(data, n):
    return np.random.choice(data['Prop_id'].unique(), size=n, replace=False )


def explo_obs_vs_sim_error_prod(type = 'prod'):

    data = prep_data()

    if type == 'prod':
        data.rename(columns={'f_p_star': 'Simulated', 
                             'f_upper_ci': 'Conf_upper',
                             'f_lower_ci': 'Conf_lower'}, inplace=True)
        
        rename_dict = prod_rename_dict
        
    elif type == 'cumprod':

        data.rename(columns={f'F_p_star': 'Simulated', 
                            f'F_upper_ci': 'Conf_upper',
                            f'F_lower_ci': 'Conf_lower'}, inplace=True)
        rename_dict = cumprod_rename_dict
        
    else:
        raise ValueError('type must be either prod or cumprod')

    data['Start_up_year'] = data['Start_up_year'].astype(int)

    data['Year'] = data['Year'] + data['Start_up_year']

    data['Prop_id'] = data['Prop_id'].astype('category')

    data['Simulated'] = data['Simulated'] / 10**6 # Convert to Mt
    data['Observed'] = data['Observed'] / 10**6 # Convert to Mt
    data['Conf_upper'] = data['Conf_upper'] / 10**6 # Convert to Mt
    data['Conf_lower'] = data['Conf_lower'] / 10**6 # Convert to Mt

    # if conf upper is bigger then 3 times the simulated value, set it to 3 times the simulated value
    data['Conf_upper'] = data.apply(lambda x: 8 * x['Simulated'] if x['Conf_upper'] > 8* x['Simulated'] else x['Conf_upper'], axis=1)
    
    
    # Loop over each target variable to create faceted plots
    for t in data['Target_var'].unique():
        # Select a random sample of 40 unique Prop_id values
        subset = data[data.Target_var == t]
        ids = get_random_ids(subset, 36)

        subset = subset[(subset['Prop_id'].isin(ids))].copy()
        subset['Target_var'] = subset['Target_var'].replace(rename_dict)
        
        # Construct the plot
        plot = (
            ggplot(subset, aes(x='Year', y='Simulated', color='Model'))
            # Confidence intervals as shaded ribbons
            + geom_ribbon(aes(ymin='Conf_lower', ymax='Conf_upper', fill='Model'),
                          alpha=0.3)
            # Simulation lines for each model
            + geom_line(size=1)
                       
            # Faceting by property name with free scales
            + facet_wrap('~Prop_name', scales='free', ncol=6)
            + theme_minimal()
            + labs(x='Year', y=f'{rename_dict[t]} (Mt)')
            + theme(
                legend_position='bottom',
                axis_text_x=element_text(size=8),  # Reduce x-axis tick text size
                axis_text_y=element_text(size=8),  # Reduce y-axis tick text size
                axis_title_x=element_text(size=8),  # Reduce x-axis label size
                axis_title_y=element_text(size=8),  # Reduce y-axis label size
                strip_text=element_text(size=8),  # Reduce facet title text size
                legend_text=element_text(size=8),  # Reduce legend text size
                legend_title=element_text(size=8), 
                figure_size=(20, 20) # Reduce legend title size
                )
            
            + scale_fill_manual(values=model_color)
            + scale_color_manual(values=model_color)
            )
        if type == 'prod':
            plot = plot + geom_point(
            data=subset.drop_duplicates(subset=['Year', 'Prop_id']),
            mapping=aes(x='Year', y='Observed'),
            color='black', size=2)
        # Save the plot to file
        save_fig_plotnine(plot, f'{t}_{type}_obs_vs_sim_conf.png', w=24, h=24)
        


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

    assert 'Prop_id' in data.columns, "Prop_id column missing in data"

    site = get_data('site')
    # construct a feature from prop name, primary commodity and Country
    site['Prop_name'] = site['Prop_name'].str.cat(site[['Primary_commodity', 'Country_name']], sep=', ')

    # merge site
    c_data = data.merge(site[['Prop_id', 'Prop_name']], on='Prop_id', how='left')

    return c_data

def transform_to_r2_text(x):
        # Extract R² values for both models
        r2_femp = x[x['Model'] == 'femp']['R2'].values[0]
        r2_hubbert = x[x['Model'] == 'hubbert']['R2'].values[0]

        nrmse_femp = x[x['Model'] == 'femp']['NRMSE'].values[0]
        nrmse_hubbert = x[x['Model'] == 'hubbert']['NRMSE'].values[0]

        
        # Format into a single string
        data = f"R2: femp ({r2_femp:.2f}), hubbert ({r2_hubbert:.2f})\n NRMSE: femp ({nrmse_femp:.2f}), hubbert ({nrmse_hubbert:.2f})"
    
        return data

def prep_data():
    """Prepares and merges data from multiple sources, adding mine context."""
    
    
    model_res_path = r'data\int\data_records_trans.csv'
    model_fits_path = r'data\int\production_model_fits_trans.json'
    ua_path = r'data\int\M_cumprod_mc_confidence\cumprod_mc_confidence.csv'

   
    rec = pd.read_csv(model_res_path)
    modelres = pd.read_json(model_fits_path)
    ua = pd.read_csv(ua_path)

    # Subset relevant columns
    rec_sub = rec[['Prop_id', 'Target_var', 'Model', 'Year', 'Observed']]

    # Ensure 'Year' is integer
    rec_sub['Year'] = rec_sub['Year'].astype(int)

    # make a column per prop id with r2 of both models for each target var

    fit_text = modelres.groupby(['Prop_id', 'Target_var']).apply(lambda x: transform_to_r2_text(x)).reset_index()
    fit_text.rename(columns={0: 'R2_text'}, inplace=True)

    ua.rename(columns={'Time_period': 'Year'}, inplace=True)
    ua['Year'] = ua['Year'].astype(int)

    # Merge uncertainty analysis (ua) with observed data records
    merged_data = ua.merge(rec_sub, on=['Prop_id', 'Target_var', 'Model', 'Year'], how='left')

    merged_data = merged_data.merge(fit_text, on=['Prop_id', 'Target_var'], how='left')

    # Add mine context
    merged_data = add_mine_context(merged_data)

    return merged_data



def manuscript_cumprod_plot():
    data = prep_data()
    melt = data.melt(
        id_vars=['Prop_id', 'Year', 'Target_var', 'Model', 'Prop_name', 'R2_text', 'Start_up_year'],
        value_vars=['Observed', 'f_p_star', 'f_upper_ci', 'f_lower_ci', 'F_p_star', 'F_lower_ci', 'F_upper_ci'],
        var_name='Var_type',
        value_name='Value'
    )

    # Strip 'f' and 'F' and create a separate column
    melt['Type'] = melt['Var_type'].str[2:]
    melt['Var_type'] = melt['Var_type'].str[0]

    melt['Var_type'].replace({'O': 'f'}, inplace=True)
    melt['Type'].replace({'p_star': 'Simulated', 'served': 'Observed'}, inplace=True)

    melt['Value'] = melt['Value'] / 10**6  # Convert to Mt

    

    piv = melt.pivot_table(
        index=['Prop_id', 'Year', 'Target_var', 'Model', 'Prop_name', 'Var_type', 'Start_up_year', 'R2_text'],
        columns='Type',
        values='Value',
        aggfunc='first'
    ).reset_index()

    
    piv['Year'] = piv['Year'] + piv['Start_up_year']

    for t in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:	
        # Select a random sample of 40 unique Prop_id values
        subset = piv[piv.Target_var == t].copy()

        subset = subset[(subset['Prop_name'].isin(rename_prop_name.keys()))].copy()
        subset['Target_var'] = subset.apply(
            lambda x: prod_rename_dict[x['Target_var']] if x['Var_type'] == 'f' else cumprod_rename_dict[x['Target_var']],
            axis=1
        )
        type_dict = {'f': 'Production', 'F': 'Cumulative Production'}
        subset['Var_type'] = subset['Var_type'].replace(type_dict)

        subset['Var_type'] = subset['Var_type'].astype('category')
        subset['Prop_name'] = subset['Prop_name'].astype('category')

        subset['Prop_name'] = subset['Prop_name'].replace(rename_prop_name)

        plot = (
            ggplot(subset, aes(x='Year')) +  # Use `subset` to ensure required variables are included
            # Simulated production and integration
            geom_line(aes(y='Simulated', color='Model'), size =1.5) +
            geom_ribbon(aes(ymin='lower_ci', ymax='upper_ci', fill='Model'), alpha=0.2) +
            # Observed data for Production
            geom_point(aes(y='Observed'), color='black', size=1.5, na_rm=True) +
            # Facet by `Prop_name` and `Var_type`
            facet_grid(cols ='Prop_name', rows = 'Var_type', space={"x": [2,1,1], "y": [1, 2]}, scales='free') +
            labs(
                x="Year",
                y="Log Value (Mt)",
            ) +
            theme_minimal() +
            theme(
                figure_size=(20, 10),
                legend_position='bottom',
                axis_text_x=element_text(rotation=45, hjust=1, size=12),
                axis_text_y=element_text(size=12),
                axis_title_x=element_text(size=12),
                axis_title_y=element_text(size=12),
                strip_text=element_text(size=12),
                legend_text=element_text(size=12),
            ) 
            + scale_color_manual(values=model_color)
            + scale_fill_manual(values=model_color)
            # log scale y axis
            + scale_y_log10()
            +  geom_text(aes(x=2005, y=.005, label='R2_text'),
                size=10, color='black',
                data = subset[subset['Var_type'] == 'Production'])

        )
        save_fig_plotnine(plot, f'{t}_manuscript_compod_plot.png', w=20, h=10)

#########################################Main Functions ##############################################



if __name__ == '__main__':
    
    manuscript_cumprod_plot()