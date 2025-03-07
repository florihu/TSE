
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from plotnine import *
import shap
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson

from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

from E_pca import get_data_per_var
from M_ml_train_loop import get_comb, pre_pipe, y_pipe, r2_calc
from util import df_to_csv_int, save_fig_plotnine





########################################################Purpose#######################################################################



########################################################Params#######################################################################

random_state = 43
test_size = 0.2
scale_tonnes = 10**-6

outlier_thres = 3


model_paths = {'Tailings_production': 'models\SVR_Tailings_production.pkl', 'Concentrate_production': 'models\SVR_Concentrate_production.pkl', 'Ore_processed_mass': 'models\GradientBoostingRegressor_Ore_processed_mass.pkl'}

rename_dict = {'Tailings_production': 'CTP', 'Concentrate_production': 'CCP', 'Ore_processed_mass': 'COP'}

#######################################################Functions#######################################################################


def calc_std_errors():
    """Computes standardized prediction errors for train and test samples across all variables."""
    
    res_df = []

    for variable in model_paths.keys():
        data = get_data_per_var(variable)

        y = data["Cum_prod"]
        X = data.drop(columns=["Cum_prod"])
        comb = get_comb(data)

        model = joblib.load(model_paths[variable])
        indices = np.arange(len(X))

        train_idx, test_idx = train_test_split(indices, stratify =comb, random_state=random_state, test_size = test_size)
        # # Predictions

        y_train = y_pipe.fit_transform(y.values[train_idx].reshape(-1, 1)).flatten()
        y_test = y_pipe.transform(y.values[test_idx].reshape(-1, 1)).flatten()
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #inv_trans
        y_train_pred = y_pipe.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_test_pred = y_pipe.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
        y_train = y_pipe.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_pipe.inverse_transform(y_test.reshape(-1, 1)).flatten()


        print(f'R2 for {variable} train: {r2_calc(y_train, y_train_pred)}')
        print(f'R2 for {variable} test: {r2_calc(y_test, y_test_pred)}')

        # calc rmse
        rmse_train = np.sqrt(np.mean((y_train - y_train_pred)**2))
        rmse_test = np.sqrt(np.mean((y_test - y_test_pred)**2))
        print(f'RMSE for {variable} train: {rmse_train}')
        print(f'RMSE for {variable} test: {rmse_test}')
      
        #train sharpio
        stat, p = shapiro(y_train - y_train_pred)

        
        print(f'Shapiro p-value for {variable} train: {p}')

        #test sharpio
        stat, p = shapiro(y_test - y_test_pred)
        print(f'Shapiro p-value for {variable} test: {p}')


        # Durbin-Watson test
        dw_train = durbin_watson(y_train - y_train_pred)
        dw_test = durbin_watson(y_test - y_test_pred)
        print(f'Durbin-Watson for {variable} train: {dw_train}')
        print(f'Durbin-Watson for {variable} test: {dw_test}')

        # Breusch-Pagan test

        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

        # add constant to X_train
        X_train = sm.add_constant(X.iloc[train_idx])
        X_test = sm.add_constant(X.iloc[test_idx])

        bp_train = het_breuschpagan(y_train - y_train_pred, X_train)
        
        bp_test = het_breuschpagan(y_test - y_test_pred, X_test)  
        print(f'Breusch-Pagan for {variable} train: {dict(zip(labels, bp_train))}')
        print(f'Breusch-Pagan for {variable} test: {dict(zip(labels,bp_test))}')
        

        # Standardized errors
        std_error_train = StandardScaler().fit_transform((y_train - y_train_pred).reshape(-1, 1)).flatten()
        std_error_test = StandardScaler().fit_transform((y_test - y_test_pred).reshape(-1, 1)).flatten()
        
        # Append results
        res_df.append(
            pd.DataFrame({
                "Variable": [variable] * len(y_train),
                "Sample": ["Train"]*len(y_train),
                "Prod_id": data.index[train_idx],
                "Y_pred": y_train_pred,
                "Y_obs": y_train,
                "Std_error": std_error_train,
            }))
        
        res_df.append(
            pd.DataFrame({
                "Variable": [variable] * len(y_test),
                "Sample": ["Test"] * len(y_test),
                "Prod_id": data.index[test_idx],
                "Y_pred": y_test_pred,
                "Y_obs": y_test,
                "Std_error": std_error_test,
            }))
    
    res = pd.concat(res_df, ignore_index=True, axis = 0)

    # Save results
    df_to_csv_int(res, "std_errors_hype_model")

    pass

def plot_res(p= 'data\int\R_ml_hype_error\std_errors_hype_model.csv'):
    df = pd.read_csv(p)

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        
        df_v = df[df.Variable == name]

        quant_90 = df_v.Y_obs.quantile(0.90)
        quant_75 = df_v.Y_obs.quantile(0.75)
        quant_50 = df_v.Y_obs.quantile(0.50)

        # plot std_error vs Y_obs and color Train Test
        p = (ggplot(df_v, aes(x='Y_obs', y='Std_error', color='Sample')) 
        + geom_point() 
        + geom_smooth(method='loess', se=True)
        + geom_hline(yintercept=[-3, 3], linetype='dashed', color = 'black', size =.5) 
        + geom_vline(xintercept=[quant_50, quant_75, quant_90], linetype='dashed', color = 'black', size =.5)
        + labs(x=f'Log {rename_dict[name]} observed (t)', y='Standardized Error')
        + theme_minimal()
        + scale_x_log10()
        )


        save_fig_plotnine(p, f'{name}_std_error_vs_Y_obs')





def qqplot_std_error():
    p = 'data\int\R_ml_hype_error\std_errors_hype_model.csv'
    df = pd.read_csv(p)

    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        
        df_v = df[df.Variable == name]

        # plot qqplot
        p = (ggplot(df_v, aes(sample='Std_error', color='Sample')) 
        + geom_qq(aes(sample='Std_error')) 
        + geom_qq_line(aes(sample='Std_error')) 
        + labs(title=f'QQ-plot of standardized errors for {rename_dict[name]}')
        + theme_minimal()
        )

        save_fig_plotnine(p, f'{name}_qqplot_error_vs_obs')

    pass


if __name__ == '__main__':
    plot_res()