'''

This script is used to explore the sample for ws and t.

Explored:

1. Multicollinearity
2. Distribution of the target variable
3. Distribution of the features

'''

stat_res_p = r'data\int\E_sample_explo\stat_res_rec.xlsx'


import numpy as np
import pandas as pd
import geopandas as gpd
from plotnine import *
import seaborn as sns
from scipy import stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
#import gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
#linear regriession
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
import networkx as nx


from util import save_fig_plotnine, df_to_csv_int, df_to_gpkg, save_fig, get_path, df_to_latex, append_to_excel, df_to_latex
##########################################################Purpose##########################################################################


#########################################################Params############################################################################

vars = ['Prop_id', 'Target_var', 'Cum_prod', 'Cum_prod_lower', 'Cum_prod_upper',
       'R2', 'NRMSE', 'Start_up_year', 'Prop_name', 'Polygon_count', 'Weight',
       'Unary_area', 'Unary_area_weighted', 'Convex_hull_area',
       'Convex_hull_area_weighted', 'Convex_hull_perimeter',
       'Convex_hull_perimeter_weighted', 'Compactness', 'Compactness_weighted',
       'Coalloc_mines_count', 'Primary_Chromium', 'Byprod_Chromium',
       'Primary_Cobalt', 'Byprod_Cobalt', 'Primary_Copper', 'Byprod_Copper',
       'Primary_Crude Oil', 'Byprod_Crude Oil', 'Primary_Gold', 'Byprod_Gold',
       'Primary_Indium', 'Byprod_Indium', 'Primary_Iron', 'Byprod_Iron',
       'Primary_Lead', 'Byprod_Lead', 'Primary_Manganese', 'Byprod_Manganese',
       'Primary_Molybdenum', 'Byprod_Molybdenum', 'Primary_Nickel',
       'Byprod_Nickel', 'Primary_Palladium', 'Byprod_Palladium',
       'Primary_Platinum', 'Byprod_Platinum', 'Primary_Rhenium',
       'Byprod_Rhenium', 'Primary_Silver', 'Byprod_Silver', 'Primary_Tin',
       'Byprod_Tin', 'Primary_Titanium', 'Byprod_Titanium', 'Primary_Tungsten',
       'Byprod_Tungsten', 'Primary_Uranium', 'Byprod_Uranium',
       'Primary_Vanadium', 'Byprod_Vanadium', 'Primary_Zinc', 'Byprod_Zinc',
       'ev', 'mt', 'nd', 'pa', 'pb', 'pi', 'py', 'sc', 'sm', 'ss', 'su', 'va',
       'vb', 'vi', 'wb',  'Latitude', 'Longitude']

log_vars = ['Cum_prod',
    'Unary_area', 'Unary_area_weighted', 'Convex_hull_area',
    'Convex_hull_area_weighted', 'Convex_hull_perimeter',
    'Convex_hull_perimeter_weighted']
    

num_vars = ['Cum_prod',
    'Polygon_count', 'Weight',
    'Unary_area', 'Unary_area_weighted', 'Convex_hull_area',
    'Convex_hull_area_weighted', 'Convex_hull_perimeter',
    'Convex_hull_perimeter_weighted', 'Compactness', 'Compactness_weighted',
    'Coalloc_mines_count', 'EPS_mean', 'EPS_slope', 'Latitude', 'Longitude']



id = ['Prop_id', 'Target_var', 'Prop_name']


cat_vars = ['Primary_Chromium', 'Byprod_Chromium',
    'Primary_Cobalt', 'Byprod_Cobalt', 'Primary_Copper', 'Byprod_Copper',
    'Primary_Crude Oil', 'Byprod_Crude Oil', 'Primary_Gold', 'Byprod_Gold',
    'Primary_Indium', 'Byprod_Indium', 'Primary_Iron', 'Byprod_Iron',
    'Primary_Lead', 'Byprod_Lead', 'Primary_Manganese', 'Byprod_Manganese',
    'Primary_Molybdenum', 'Byprod_Molybdenum', 'Primary_Nickel',
    'Byprod_Nickel', 'Primary_Palladium', 'Byprod_Palladium',
    'Primary_Platinum', 'Byprod_Platinum', 'Primary_Rhenium',
    'Byprod_Rhenium', 'Primary_Silver', 'Byprod_Silver', 'Primary_Tin',
    'Byprod_Tin', 'Primary_Titanium', 'Byprod_Titanium', 'Primary_Tungsten',
    'Byprod_Tungsten', 'Primary_Uranium', 'Byprod_Uranium',
    'Primary_Vanadium', 'Byprod_Vanadium', 'Primary_Zinc', 'Byprod_Zinc',
    'ev', 'mt', 'nd', 'pa', 'pb', 'pi', 'py', 'sc', 'sm', 'ss', 'su', 'va',
    'vb', 'vi', 'wb']


cum_target_var_rename = {'Concentrate_production':'CCP',
                         'Ore_processed_mass': 'COP',
                         'Tailings_production': 'CTP'}


##################################################################Functions################################################################

def corr_heat_plot(p = r'data\int\E_ml_explo\correlation_results.csv', sig_level=0.05, high = '#b35806', low = '#542788'):    
    
    cor = pd.read_csv(p)

    # filter out nan values
    cor = cor[~cor.isnull().any(axis=1)]

    # pivot vars 
    cor['Significant'] = cor['P-Value'] < sig_level

    
    # Add and asterix label to the correlation if it is significant
    cor['Label'] = cor.apply(lambda x: f"{x['Correlation']:.2f}" +'*' if x['Significant'] else f"{x['Correlation']:.2f}", axis=1)


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = cor[cor.Target_var == name]

        # Create the heatmap
        plot = (
            ggplot(t, aes(x='Variable 1', y='Variable 2', fill='Correlation'))
            + geom_tile(color="white", size=0.2)
            + scale_fill_gradient2(low=low, mid="white", high=high, midpoint=0)
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1, size=7),
                axis_text_y=element_text(size=7),
                axis_title=element_blank(),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                legend_position="right"
            )
            + coord_fixed()
            + geom_text(aes(label='Label'), size=6, color="black")
        )
            
        # Save the plot
        save_fig_plotnine(plot, f'{name}_correlation_matrix_heatmap.png', w=24, h=24)

    pass

def hist_per_variable(df, cat_vars, num_vars):
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

def unit_rename(df, log_vars, num_vars, units):
    for c in units:
        if c in log_vars:
            units[c] = f'log {units[c]}'
            
    
    df.rename(columns={col: f'{col} {units[col]}' for col in units}, inplace=True)

    # rename numerical variables to include the units
    num_vars = [f'{col} {units[col]}' for col in num_vars]
    
    return df, num_vars

def hist_per_var_type():

    df = get_data()
    # Create a DataFrame to store the results

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = df[df.Target_var == name]

        t= clean_and_imput(t)

        t.set_index('Prop_id', inplace=True)
    
        # Create histogram for numerical variables
        df_num = t[num_vars].reset_index().melt(id_vars='Prop_id', var_name='Variable', value_name='Value')
        
        df_num['Variable'] = pd.Categorical(df_num['Variable'])
        
        plot_num = (
            ggplot(df_num, aes(x='Value', fill='Variable'))
            + geom_histogram(bins=30, color='black', alpha=0.5)
            + facet_wrap('~Variable', scales='free')
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1),
                axis_title=element_blank(),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                legend_position="none"
            )
        )

        save_fig_plotnine(plot_num, f'{name}_histogram_numerical_variables.png', w=14, h=10)

        df_log = df_num.copy()

        df_log['Value'] = np.log10(df_log['Value'])

        plot_log = (
            ggplot(df_log, aes(x='Value', fill='Variable'))
            + geom_histogram(bins=30, color='black', alpha=0.5)
            + facet_wrap('~Variable', scales='free')
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1),
                axis_title=element_blank(),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                legend_position="none"
            )
        )
    
        # Save numerical variables histogram
        save_fig_plotnine(plot_log, f'{name}_histogram_numerical_variables_log.png', w=14, h=10)
    
    

        df_cat = t[cat_vars].reset_index().melt(id_vars='Prop_id', var_name='Variable', value_name='Value')
        df_cat['Prop_id'] = pd.Categorical(df_cat['Prop_id'])
        df_cat['Variable'] = pd.Categorical(df_cat['Variable'])
        df_cat['Value'] = df_cat['Value'].astype(int)
        # Ensure the values are 0 or 1
        assert df_cat['Value'].isin([0, 1]).all(), "Categorical variables must have values 0 or 1."

        # Cast the 'value' column to categorical
        df_cat['Value'] = pd.Categorical(df_cat['Value'], categories=[0, 1])

        plot_cat = (
            ggplot(df_cat, aes(x='Value', fill='Variable'))
            + geom_bar(position='dodge', color='black', alpha=0.7)
            + facet_wrap('~Variable', scales='free')
            + theme_minimal()
            + theme(
                axis_text_x=element_text(angle=45, hjust=1),
                axis_title=element_blank(),
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
                legend_position="none"
            )
        )
        # Save categorical variables histogram
        save_fig_plotnine(plot_cat, f'{name}_histogram_categorical_variables.png', w=14, h=10)

    pass

def test_normality(alpha=0.05):
   
    df = get_data()
    # Create a DataFrame to store the results
    res = []
    

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = df[df.Target_var == name]

        t= clean_and_imput(t)

        t = t[num_vars]

        for var in num_vars:
                       
            # Perform the Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(t[var])
            
            data_log = np.log(t[var])
            shapiro_stat_log, shapiro_p_log = stats.shapiro(data_log)

            res.append(pd.DataFrame({'Variable': var, 'Target_var': name,  'Norm_stat': shapiro_stat, 'Norm_p': shapiro_p,
                                     'Log_stat':shapiro_stat_log, 'Log_p':shapiro_p_log }, index=[0]))
            
    res = pd.concat(res, ignore_index=True)
    
    res['Target_var'].replace(cum_target_var_rename, inplace=True)

    res[['Norm_stat', 'Norm_p', 'Log_stat', 'Log_p']] = res[['Norm_stat', 'Norm_p', 'Log_stat', 'Log_p']].apply(lambda x: x.round(3))

    # format to three digits
    res['Norm_stat'] = res['Norm_stat'].apply(lambda x: f"{x:.3f}")
    
    # Save the results to a LaTeX table
    df_to_csv_int(res, 'normality_test_results')

    return 

def fit_random_forest(df, name, log_vars, cat_vars, num_vars, target_vars):
    '''
    Fit a Random Forest model with hyperparameter tuning using RandomizedSearchCV, 
    evaluate performance on train/test split, and return feature importances.
    '''
    # Transform numerical variables to log scale
    df[log_vars] = df[log_vars].apply(np.log10)

    # Split the data into features (X) and target (y)
    X = df.drop(target_vars, axis=1)
    y = pd.DataFrame(df[target_vars].iloc[:, 0])


    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features and target
    X_train_scaled = MinMaxScaler().fit_transform(X_train)
    X_test_scaled = MinMaxScaler().fit_transform(X_test)
    y_train_scaled = MinMaxScaler().fit_transform(y_train)
    y_test_scaled = MinMaxScaler().fit_transform(y_test)

    # Apply PCA to reduce dimensionality (retain components explaining > 80% variance)
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Take components that explain > 80% of the variance
    index_80 = np.argmax(pca.explained_variance_ratio_.cumsum() > 0.85)
    X_train_pca = X_train_pca[:, :index_80]
    X_test_pca = X_test_pca[:, :index_80]

    
    gb = RandomForestRegressor()

    gb.fit(X_train_pca, y_train_scaled)


    # Evaluate the best model on training data
    r2_train = gb.score(X_train_pca, y_train_scaled)
    rmse_train = np.sqrt(np.mean((gb.predict(X_train_pca) - y_train_scaled) ** 2))
    coef_var_train = rmse_train / y_train_scaled.mean()

    # Evaluate the best model on test data
    r2_test = gb.score(X_test_pca, y_test_scaled)
    rmse_test = np.sqrt(np.mean((gb.predict(X_test_pca) - y_test_scaled) ** 2))
    coef_var_test = rmse_test / y_test_scaled.mean()

    # Print the evaluation metrics for both train and test sets
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Train Coefficient of Variation: {coef_var_train:.4f}")
    print(f"Test Coefficient of Variation: {coef_var_test:.4f}")

    # Return feature importances from the best model
    return gb.feature_importances_

def immpute_vars(df,cat_vars, num_vars):
    
    df = df.copy().sort_values('Active_years')

    df[['EPS_mean', 'EPS_slope']] = df[['EPS_mean', 'EPS_slope']].fillna(df[['EPS_mean', 'EPS_slope']].mean())
    # impute missing values using the sorted numerical values per active years 
    df[num_vars] = df[num_vars].fillna(method='ffill')

    df[cat_vars] = df[cat_vars].apply(lambda x: x.fillna(0))

    df.dropna(subset='geometry', inplace=True)
    return df

def summary_stats(df, name,  cat_vars, num_vars, units):
    # percentage of missing values per var relative to samples
    missing = (df.isnull().mean().round(2) / len(df)) * 100
    df_to_latex(missing, f'{name}_missing_values.tex')

    df = immpute_vars(df, cat_vars, num_vars)
    
    numeric_sum = df[num_vars].describe().round(2).T

    numeric_sum['Unit'] = [units[col] for col in numeric_sum.index]
    # unit should appear direchtly after index name
    numeric_sum = numeric_sum[['Unit', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    cat_sum = pd.DataFrame()
    cat_sum['Count'] = df[cat_vars].sum()
    cat_sum['Percentage'] = (cat_sum['Count'] / len(df) * 100).round(2)

    df_to_latex(numeric_sum, f'{name}_numeric_summary.tex')
    df_to_latex(cat_sum, f'{name}_categorical_summary.tex')

    return None

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2, p_val = stats.chi2_contingency(confusion_matrix)[0:2]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), p_val


def clean_and_imput(df):
    # Drop cols
    df.drop(columns= ['Unnamed: 0'], inplace = True)

    # Remove Zero cols
    df = df[~df.isnull()]

    # impute 
    df[['EPS_mean', 'EPS_slope']] = df[['EPS_mean', 'EPS_slope']].fillna(df[['EPS_mean', 'EPS_slope']].mean())

    return df



def corr_calc():
    """
    Calculate correlations for:
    - numerical vs numerical
    - categorical vs numerical
    - categorical vs categorical (using point biserial correlation)
    Includes p-values for significance testing, and returns all results in a single DataFrame.
    """
    # Transform numerical variables to log scale
    
    df = get_data()

    df[log_vars] = df[log_vars].apply(np.log10)
    
    results = []


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        t = df[df.Target_var == name]

        t = clean_and_imput(t)

        # Numerical vs Numerical
        for var1 in num_vars:
            for var2 in num_vars:
                if var1 != var2:
                    corr, p_val = stats.pearsonr(t[var1], t[var2])
                    results.append({
                        'Target_var': name,
                        'Variable 1': var1,
                        'Variable 2': var2,
                        'Type 1': 'numerical',
                        'Type 2': 'numerical',
                        'Test_Type': 'Pearson',
                        'Correlation': corr,
                        'P-Value': p_val
                    })

        # Categorical vs Numerical
        for var1 in cat_vars:
            for var2 in num_vars:
                if var1 != var2:
                    corr, p_val = stats.pointbiserialr(t[var1].astype(int), t[var2])
                    results.append({
                        'Target_var': name,
                        'Variable 1': var1,
                        'Variable 2': var2,
                        'Type 1': 'binary',
                        'Type 2': 'numerical',
                        'Test_Type': 'Point Biserial',
                        'Correlation': corr,
                        'P-Value': p_val
                    })
                    # Out of consistency the inverse is also included thi is in the case of num-num etc. already by definition included.
                    results.append({
                        'Target_var': name,
                        'Variable 1': var2,
                        'Variable 2': var1,
                        'Type 1': 'numerical',
                        'Type 2': 'binary',
                        'Test_Type': 'Point Biserial',
                        'Correlation': corr,
                        'P-Value': p_val
                    })

        # cat vs cat
        for var1 in cat_vars:
            for var2 in cat_vars:
                if var1 != var2 and df[var1].nunique() == 2 and df[var2].nunique() == 2:
                    contingency = pd.crosstab(t[var1].astype(int), t[var2].astype(int))
                    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                    n = contingency.sum().sum()
                    phi = (chi2 / n) ** 0.5
                    results.append({
                        'Target_var': name,
                        'Variable 1': var1,
                        'Variable 2': var2,
                        'Type 1': 'binary',
                        'Type 2': 'binary',
                        'Test_Type': 'Phi Coefficient',
                        'Correlation': phi,
                        'P-Value': p_val
                    })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    df_to_csv_int(results_df, 'correlation_results')

    
    pass


def vif(df, name, cat_vars, num_vars, target_vars, units):
    '''
    Calculate the Variance Inflation Factor (VIF) for all variables in the DataFrame. Extract also the pvalue
    H0: the r2 is not significantly different from 0

    Lets only do it fro numerical variables

    '''
    # Transform numerical variables to log scale
    
    df.drop(['geometry', 'Prop_id', 'id_data_source', 'COU', 'continent', 'iso3', 'Unnamed: 0'], axis=1, inplace=True)
    
    df = immpute_vars(df, cat_vars, num_vars)
    log_vars = test_normality(df, name, num_vars)
    df[log_vars] = df[log_vars].apply(np.log10)

    
    df.drop(target_vars, axis=1, inplace=True)

    num_vars = [col for col in num_vars if col not in target_vars]

    df, num_vars = unit_rename(df, log_vars, num_vars, units)

    # calculate vif
    vif_data = pd.DataFrame(index = df[num_vars].columns)
    vif_data["VIF"] = [variance_inflation_factor(df[num_vars].values, i) for i in range(df[num_vars].shape[1])]
    vif_data = vif_data.sort_values(by='VIF', ascending=False)

    # Save to csv
    append_to_excel(stat_res_p, vif_data, f'{name}_vif_results')

    return None

def pairplot():
    df = get_data()

    comb_vars = num_vars + cat_vars

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:

        t = df[df.Target_var == name]

        t= clean_and_imput(t)

        f, ax = plt.subplots(figsize=(24, 24))

        sns.pairplot(t[comb_vars])

        save_fig(f'pairplot_{name}.png')

        plt.show()


def pca():
    '''
    Perform PCA on the numerical variables of the DataFrame and return the results in a DataFrame.
    '''
    # Transform numerical variables to log scale

    df = get_data()
    df[log_vars] = df[log_vars].apply(np.log10)

    vars = num_vars + cat_vars

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        
        

        t = df[df.Target_var == name][vars]

        # Standardize the data
        scaler = MinMaxScaler()
        t_scaled = scaler.fit_transform(t)

        # Perform PCA
        pca = PCA()
        pca_results = pca.fit_transform(t)

        # explained variance
        explained_variance = pca.explained_variance_ratio_

        explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
        explained_variance_df.index = [i+1 for i in range(pca_results.shape[1])]
        explained_variance_df = explained_variance_df.round(4)
        explained_variance_df['Cummulative Explained Variance'] = explained_variance_df['Explained Variance'].cumsum()

        # reset index and rename to component
        explained_variance_df.reset_index(inplace=True)
        explained_variance_df.rename(columns={'index': 'Component'}, inplace=True)

        # plot explained variance - plotnine scatter
        plot = (
            ggplot(explained_variance_df, aes(x='Component', y='Cummulative Explained Variance'))
            + geom_point()
            + theme_minimal()
            + labs(x='Component', y='Cummulative Explained Variance (%)')
        )
        # Make a vertical line at 80% explained variance
        plot += geom_hline(yintercept=0.8, linetype='dashed', color='red')

        # Save the plot
        save_fig_plotnine(plot, f'{name}_explained_variance.png', w=10, h=6)

        return None

def bin_network(df, cat_vars, num_vars, name):
    # Impute missing values for categorical and numerical variables
    df = immpute_vars(df, cat_vars, num_vars)

    com_vars = [
        'Primary_Chromium', 'Byprod_Chromium', 'Primary_Cobalt', 'Byprod_Cobalt', 
        'Primary_Copper', 'Byprod_Copper', 'Primary_Crude Oil', 'Byprod_Crude Oil', 
        'Primary_Gold', 'Byprod_Gold', 'Primary_Indium', 'Byprod_Indium', 
        'Primary_Iron', 'Byprod_Iron', 'Primary_Lead', 'Byprod_Lead', 
        'Primary_Manganese', 'Byprod_Manganese', 'Primary_Molybdenum', 
        'Byprod_Molybdenum', 'Primary_Nickel', 'Byprod_Nickel', 'Primary_Palladium', 
        'Byprod_Palladium', 'Primary_Platinum', 'Byprod_Platinum', 
        'Primary_Rhenium', 'Byprod_Rhenium', 'Primary_Silver', 'Byprod_Silver', 
        'Primary_Tin', 'Byprod_Tin', 'Primary_Titanium', 'Byprod_Titanium', 
        'Primary_Tungsten', 'Byprod_Tungsten', 'Primary_Uranium', 'Byprod_Uranium', 
        'Primary_Vanadium', 'Byprod_Vanadium', 'Primary_Zinc', 'Byprod_Zinc'
    ]
    
    # Convert selected columns to binary format (0 or 1)
    df_b = df[com_vars].astype(int)

    # Filter columns where the sum of non-zero values is greater than zero
    df_b = df_b.loc[:, (df_b.sum(axis=0) != 0)]

    # Compute the co-occurrence matrix
    co_occurrence_matrix = df_b.T.dot(df_b)

    # Step 2: Build a NetworkX graph
    G = nx.Graph()

    # Add nodes from the columns of df_b (commodities)
    G.add_nodes_from(df_b.columns)

    # Define a dictionary to assign colors to primary and byproduct commodities
    color_map = {}
    for col in df_b.columns:
        if 'Primary' in col:
            color_map[col] = 'lightblue'  # Color for primary commodities
        elif 'Byprod' in col:
            color_map[col] = 'lightgreen'  # Color for byproduct commodities

    # Add edges based on co-occurrence matrix
    for i in range(len(co_occurrence_matrix.columns)):
        for j in range(i, len(co_occurrence_matrix.columns)):
            weight = co_occurrence_matrix.iloc[i, j]
            # Skip self-loops (i.e., no edge where i == j) and edges with zero weight
            if weight > 0 and i != j:
                G.add_edge(co_occurrence_matrix.columns[i], co_occurrence_matrix.columns[j], weight=weight)


    # Plot the network
    plot_network(G, name)

    return None


def plot_network(G, name):


    # Define a dictionary to assign colors to primary and byproduct commodities
    color_map = {}
    for col in G.nodes():
        if 'Primary' in col:
            color_map[col] = 'lightblue'  # Color for primary commodities
        elif 'Byprod' in col:
            color_map[col] = 'lightgreen'  # Color for byproduct commodities

    pos =  nx.circular_layout(G) # Positioning the nodes
    plt.figure(figsize=(10, 8))

    # Extract edge weights
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]  # Edge weights for visualization

     # min max scale weights
    weights = StandardScaler().fit_transform(np.array(weights).reshape(-1, 1)).flatten()

    # Draw nodes and edges with the color map for nodes
    node_colors = [color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    nx.draw_networkx_labels(G, pos, font_size=6)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, edge_color='gray')

    
    plt.axis('off')  # Hide axes
    plt.tight_layout()

    # Save the figure
    save_fig(f'{name}_network_colored.png')

    # Show the plot
    plt.show()


def sample_characteristics():
    d = get_data()

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        t = d[d.Target_var == name]
        t = clean_and_imput(t)

        var_vars = num_vars + cat_vars

        stat = t[var_vars].describe().round(2).T

        # include for all variable names a \ before the _
        stat.index = stat.index.str.replace('_', '\_')

        stat.columns = stat.columns.str.replace('%', '\%')

        df_to_latex(stat, f'{name}_stat.tex', longtable=True)

        



    return None

def spatial_plot(df, target_vars, cat_vars, num_vars, wb):


    df = immpute_vars(df, cat_vars, num_vars)
    # transform to degree crs

    df = df.to_crs('EPSG:4326')
    wb = wb.to_crs('EPSG:4326')


    for t in target_vars:

        fig, ax = plt.subplots(figsize=(14, 7))

        df[t] = df[t].apply(np.log10)

        df.rename(columns={f'{t}': f'{t} log t', 'Area_mine': 'Area_mine km2'} , inplace=True)

        # Plot the target variable as a choropleth (continuous data)
        sns.scatterplot(data=df,  x=df.geometry.x, y=df.geometry.y, ax=ax, size = 'Area_mine km2',  hue=f'{t} log t', palette='viridis', legend=True, edgecolor=None, sizes=(10, 200))      

        # plot the world boundaries wiht grey background and black borders
        wb.boundary.plot(ax=ax, color='black', linewidth=0.4)

        # get legend into left lower corner
        ax.legend(loc='lower left', bbox_to_anchor=(0, 0), frameon=False)

        ax.set_axis_off()
        # Display the plot with the legend
        plt.tight_layout()
    
        save_fig(f'{t}_spatial_plot.png')
        plt.close()

        
    return None

def get_data():

    d_path = 'data\int\D_ml_sample\ml_sample.csv'
    
    d = pd.read_csv(d_path)
    return d


################################################################Main#######################################################################




if __name__ == '__main__':
    pca()