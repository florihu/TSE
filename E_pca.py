
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from plotnine import *


from util import save_fig_plotnine
from E_ml_explo import get_data, clean_and_imput, log_vars, num_vars, cat_vars



################################################Purpose#############################################



################################################Params###############################################


logs_features = [i for i in log_vars if i is not 'Cum_prod']






def get_data_per_var(name, out_remove = False, thres_out = None):

    d = get_data()

    geo_p = 'data\int\M_geography_feature\geo_sim.csv.csv'

    geo = pd.read_csv(geo_p)
    
    d = d[d.Target_var == name]
    geo = geo[geo.Target_var == name]

    geo.dropna(inplace=True, axis =1)

    geo.drop('Unnamed: 0', axis=1, inplace=True)

    m = d.merge(geo, on=['Prop_id', 'Target_var'], how='left')

    m.set_index('Prop_id', inplace=True)

    m = clean_and_imput(m)

    m[logs_features] = np.log(m[logs_features])

    geo_cols = [col for col in m.columns if 'Cluster' in col]

    vars = num_vars + cat_vars + geo_cols

    if out_remove:
        assert thres_out is not None, 'Please provide a threshold for outlier removal.'
        p = r'data\int\R_ml_hype_error\std_errors_hype_model.csv'
        out = pd.read_csv(p)
        out = out[out.Variable == name]
        out_ids = out[np.abs(out.Std_error) > thres_out].Prod_id
        m = m[~m.index.isin(out_ids)]

    return m[vars]



def pca_explo():
    '''
    Perform PCA on the numerical variables of the DataFrame and return the results in a DataFrame.
    '''
    # Transform numerical variables to log scale


    for name in ['Ore_processed_mass', 'Concentrate_production', 'Tailings_production']:
        
        t = get_data_per_var(name)

        t.drop(columns = 'Cum_prod', inplace = True)

        # Standardize the data
        scaler = MinMaxScaler()
        t_scaled = scaler.fit_transform(t)

        # Perform PCA
        pca = PCA()
        pca_results = pca.fit_transform(t_scaled)

        # explained variance
        explained_variance = pca.explained_variance_ratio_

        explained_variance_df = pd.DataFrame(explained_variance, columns=['Explained Variance'])
        explained_variance_df.index = [i+1 for i in range(pca_results.shape[1])]
        explained_variance_df = explained_variance_df.round(4)
        explained_variance_df['Cummulative Explained Variance'] = explained_variance_df['Explained Variance'].cumsum()

        # reset index and rename to component
        explained_variance_df.reset_index(inplace=True)
        explained_variance_df.rename(columns={'index': 'Component'}, inplace=True)

        #print 85\% and 95\% number of feature to keep
        print(name)
        print(explained_variance_df[explained_variance_df['Cummulative Explained Variance'] <= 0.85].shape[0])
        print(explained_variance_df[explained_variance_df['Cummulative Explained Variance'] <= 0.95].shape[0])

        # plot explained variance - plotnine scatter
        plot = (
            ggplot(explained_variance_df, aes(x='Component', y='Cummulative Explained Variance'))
            + geom_point()
            + theme_minimal()
            + labs(x='Component', y='Cummulative Explained Variance (%)')
        )
        # Make a vertical line at 80% explained variance
        plot += geom_hline(yintercept=0.85, linetype='dashed', color='#c51b7d')
        plot += geom_hline(yintercept=0.95, linetype='dashed', color='#4d9221')

        plot += geom_vline(xintercept=explained_variance_df[explained_variance_df['Cummulative Explained Variance'] <= 0.85].shape[0], linetype='dashed', color='#c51b7d')
        plot += geom_vline(xintercept=explained_variance_df[explained_variance_df['Cummulative Explained Variance'] <= 0.95].shape[0], linetype='dashed', color='#4d9221')

        # Save the plot
        save_fig_plotnine(plot, f'{name}_explained_variance.png', w=10, h=6)

    pass

################################################Functions#############################################




#################################################Main#################################################

if __name__ == '__main__':
    pca_explo()
    