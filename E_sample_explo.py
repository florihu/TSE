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
from scipy import stats
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from util import save_fig_plotnine, data_to_csv_int, df_to_gpkg, save_fig, get_path, df_to_latex
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

def multi_cor(df, name, log_vars, unit_conv):

    df[log_vars]  = df[log_vars].apply(np.log10)
    df.rename(columns={col: f'{col} {unit_conv[col]}' for col in log_vars}, inplace=True)

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
    save_fig_plotnine(plot, f'{name}_correlation_matrix_heatmap.png', w=18, h=18)
    return None

# def hist_per_variable(df, cat_vars, num_vars):
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


def hist_per_variable(df, name,  log_vars,  cat_vars, num_vars, unit_conv):

    # Transform numerical variables to log scale
    if log_vars:       
        # rename and add the unit to the name
        df[log_vars] = df[log_vars].apply(np.log10)
        rename_prep = {col: f'{col} {unit_conv[col]}' for col in unit_conv}
        df.rename(columns=rename_prep, inplace=True)

        #change num vars
        num_vars = [rename_prep[col] if col in rename_prep.keys() else col for col in num_vars]

    # Create histogram for numerical variables
    if num_vars:
        df_num = df[num_vars].reset_index().melt(id_vars='Prop_id', var_name='variable', value_name='value')
        df_num['Prop_id'] = pd.Categorical(df_num['Prop_id'])
        df_num['variable'] = pd.Categorical(df_num['variable'])
        
        plot_num = (
            ggplot(df_num, aes(x='value', fill='variable'))
            + geom_histogram(bins=30, color='black', alpha=0.5)
            + facet_wrap('~variable', scales='free')
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
        save_fig_plotnine(plot_num, f'{name}_histogram_numerical_variables.png', w=14, h=14)
    
    # Create histogram for categorical variables
    if cat_vars:

        df_cat = df[cat_vars].reset_index().melt(id_vars='Prop_id', var_name='variable', value_name='value')
        df_cat['Prop_id'] = pd.Categorical(df_cat['Prop_id'])
        df_cat['variable'] = pd.Categorical(df_cat['variable'])

        # Ensure the values are 0 or 1
        assert df_cat['value'].isin([0, 1]).all(), "Categorical variables must have values 0 or 1."

        # Cast the 'value' column to categorical
        df_cat['value'] = pd.Categorical(df_cat['value'], categories=[0, 1])

        plot_cat = (
            ggplot(df_cat, aes(x='value', fill='variable'))
            + geom_bar(position='dodge', color='black', alpha=0.7)
            + facet_wrap('~variable', scales='free')
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
        save_fig_plotnine(plot_cat, f'{name}_histogram_categorical_variables.png', w=14, h=14)

    return None


def test_normality(df, name, num_vars):
   
    # Create a DataFrame to store the results
    normality_results = pd.DataFrame(index=num_vars, columns=['Shapiro_Wilk_stat', 'Shapiro_Wilk_p', 'Log_Shapiro_Wilk_stat', 'Log_Shapiro_Wilk_p'])
    
    # Test the normality of the numerical variables
    for var in num_vars:
        # Get the data for the variable
        data = df[var].dropna()
        
        # Perform the Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(data)
        normality_results.loc[var, 'Shapiro_Wilk_p'] = shapiro_p
        normality_results.loc[var, 'Shapiro_Wilk_stat'] = shapiro_stat

        
        data_log = np.log(data)
        shapiro_stat_log, shapiro_p_log = stats.shapiro(data_log)
        normality_results.loc[var, 'Log_Shapiro_Wilk_p'] = shapiro_p_log
        normality_results.loc[var, 'Log_Shapiro_Wilk_stat'] = shapiro_stat_log

    

    # Save the results to a LaTeX table
    df_to_latex(normality_results, f'{name}_normality_test.tex')
    
    return None

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


def summary_stats(df, name, target_vars, cat_vars, num_vars, unit_conv):

    numeric_sum = df[num_vars].describe().round(2).T


    cat_sum = pd.DataFrame()
    cat_sum['Count'] = df[cat_vars].sum()
    cat_sum['Percentage'] = (cat_sum['Count'] / len(df) * 100).round(2)

    df_to_latex(numeric_sum, f'{name}_numeric_summary.tex')
    df_to_latex(cat_sum, f'{name}_categorical_summary.tex')

    return None


def prep_dset(df):
    # Drop unnecessary columns
    df.drop(['geometry', 'id_data_source'], axis=1, inplace=True)
    df.set_index('Prop_id', inplace=True)
    # Drop columns where the sum of the columns is 0 
    df = df.loc[:, (df.sum(axis=0) != 0)]

    cat_vars = [col for col in df.columns if col.startswith('Primary') or col.startswith('Byprod')]
    num_vars = [col for col in df.columns if col not in cat_vars]
    
    df = df.dropna(subset='Area_mine')

    df['EPI'] = df['EPI'].fillna(df['EPI'].mean())

    df[cat_vars] = df[cat_vars].astype(int)
    df[num_vars] = df[num_vars].astype(float)

    return df, cat_vars, num_vars


def unit_renaming(log_col, unit_conv):
    '''
    Rename the columns of the DataFrame to include the units of the variables.
    '''
    for c in unit_conv.keys():
        if c in log_col:
            unit_conv[c] = f'log {unit_conv[c]}'

    return unit_conv


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

def corr(df, name, log_vars, cat_vars, num_vars):

    """
    Calculate correlations for:
    - numerical vs numerical
    - categorical vs numerical
    - categorical vs categorical (using point biserial correlation)
    Includes p-values for significance testing, and returns all results in a single DataFrame.
    """
    # Transform numerical variables to log scale
    df[log_vars] = df[log_vars].apply(lambda x: np.log10(x.clip(lower=1e-10)))

    # Initialize a list to collect results
    results = []

    # Numerical vs Numerical
    for var1 in num_vars:
        for var2 in num_vars:
            if var1 != var2:
                corr, p_val = stats.pearsonr(df[var1], df[var2])
                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Type 1': 'numerical',
                    'Type 2': 'numerical',
                    'Test_Type': 'Pearson',
                    'Correlation': corr,
                    'P-Value': p_val
                })

    # Categorical vs Numerical
    for cat_var in cat_vars:
        for num_var in num_vars:
            corr, p_val = stats.pointbiserialr(df[var1], df[var2])
            results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Type 1': 'categorical',
                    'Type 2': 'numerical',
                    'Test_Type': 'Point Biserial',
                    'Correlation': corr,
                    'P-Value': p_val
                })

    # Categorical vs Categorical (Point Biserial Correlation)
    for var1 in cat_vars:
        for var2 in cat_vars:
            if var1 != var2:
                # Calculate point biserial correlation directly
                corr, p_val = stats.pointbiserialr(df[var1], df[var2])
                results.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Type 1': 'categorical',
                    'Type 2': 'categorical',
                    'Test_Type': 'Point Biserial',
                    'Correlation': corr,
                    'P-Value': p_val
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    data_to_csv_int(results_df, f'{name}_correlation_results')

    return results_df

def vif(df, name, log_vars, target_vars):
    '''
    Calculate the Variance Inflation Factor (VIF) for all variables in the DataFrame. Extract also the pvalue
    H0: the r2 is not significantly different from 0

    '''
    # Transform numerical variables to log scale

    df[log_vars] = df[log_vars].apply(np.log10)
    df.drop(target_vars, axis=1, inplace=True)

    # calculate vif
    vif_data = pd.DataFrame(index = df.columns)
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif_data = vif_data.sort_values(by='VIF', ascending=False)

    # Save to csv
    data_to_csv_int(vif_data, f'{name}_vif_results')

    return None

def pairplot(df, name, log_vars, unit_conv):
    df[log_vars] = df[log_vars].apply(np.log10)
    df.rename(columns={col: f'{col} {unit_conv[col]}' for col in unit_conv}, inplace=True)

    # sns pairplot
    f, ax = plt.subplots(figsize=(20, 20))
    sns.pairplot(df)
    save_fig(f'{name}_pairplot.png')
    return None

def pca(df, name, log_vars, num_vars, target_vars):
    '''
    Perform PCA on the numerical variables of the DataFrame and return the results in a DataFrame.
    '''
    # Transform numerical variables to log scale
    df[log_vars] = df[log_vars].apply(np.log10)

    df.drop(target_vars, axis=1, inplace=True)

    # Standardize the data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Perform PCA
    pca = PCA()
    pca_results = pca.fit_transform(df_scaled)

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



def main():

    unit_conv = {'Active_years':'y', 'Concentrate_production':'t', 'Tailings_production': 't',
        'Polygon_count': 'count', 'Area_mine': 'km2', 'Area_mine_weighted':'km2',
        'Convex_hull_area': 'km2', 'Convex_hull_area_weighted': 'km2',
            'Convex_hull_perimeter': 'km2', 'Convex_hull_perimeter_weighted': 'km2',
            'Compactness': 'ratio', 'Compactness_weighted': 'ratio', 'Weight': 'ratio',
            'Waste_rock_production': 't', 'Ore_processed_mass': 't', 'Count': 'count', }
    

    for name in ['tailings', 'waste_rock']:
        
        if name == 'tailings':
            path = get_path('tailings.gpkg')
            log_vars = ['Active_years', 'Concentrate_production', 'Tailings_production', 'Convex_hull_area', 'Convex_hull_perimeter', 'Convex_hull_perimeter_weighted']
            target_vars = ['Tailings_production', 'Concentrate_production']

            # unit_conv is in log var add log to the unit
            unit_conv = unit_renaming(log_vars, unit_conv)


        if name == 'waste_rock':
            path = get_path('waste_rock.gpkg')
            log_vars = ['Active_years', 'Ore_processed_mass', 'Polygon_count', 'Weight', 'Area_mine', 'Area_mine_weighted','Convex_hull_area','Convex_hull_area_weighted' ,'Convex_hull_perimeter', 'Convex_hull_perimeter_weighted']
            target_vars = ['Waste_rock_production', 'Ore_processed_mass']
            unit_conv = unit_renaming(log_vars, unit_conv)
                   
        
        sample = gpd.read_file(path)
        sample, cat_vars, num_vars = prep_dset(sample)
        summary_stats(sample, name, target_vars, cat_vars, num_vars, unit_conv)
    

    return None


if __name__ == '__main__':
    main()