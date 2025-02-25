
from imblearn.over_sampling import SMOTE, SMOTENC
import pandas as pd
import smogn
import numpy as np
from plotnine import *

from E_pca import get_data_per_var
from M_ml_train_loop import get_comb, pre_pipe, y_pipe, r2_calc
from E_ml_explo import cat_vars
from util import df_to_csv_int, save_fig_plotnine



######################################################################Params#######################################################################
random_state = 43

replace_target = {'Tailings_production': 'CTP', 'Concentrate_production': 'CCP', 'Ore_processed_mass': 'COP'}



def explo_oversampling():
    
    """Computes standardized prediction errors for train and test samples across all variables."""
    res = []
    for variable in ["Tailings_production", "Concentrate_production", "Ore_processed_mass"]:

        
        data = get_data_per_var(variable)

        data['Cum_prod'] = np.log10(data['Cum_prod'])

        data.reset_index(inplace=True, drop=True)


        rg_mtrx = [
            [data['Cum_prod'].quantile(0.50), 0, 0],   # Below median: No oversampling
            [data['Cum_prod'].quantile(0.75), 1, 0],  # Moderate relevance in upper quartile
            [data['Cum_prod'].quantile(0.90), 1, 0],  # High relevance above 90% quantile
            [data['Cum_prod'].max(), 0, 0]  # Ensure extreme max values are captured
        ]
        
        smog = smogn.smoter(
            data, 
            y='Cum_prod', 
            k=5, 
            samp_method='balance', 
            rel_method="manual", 
            rel_thres=0.9,
            rel_ctrl_pts_rg = rg_mtrx

        )


        res.append(pd.DataFrame({'Var':[variable]*len(smog['Cum_prod']),  'Type':['synth']*len(smog['Cum_prod']), 'y': smog['Cum_prod'] }))
        
        res.append(pd.DataFrame({'Var':[variable]*len(data['Cum_prod']), 'Type':['original']*len(data['Cum_prod']), 'y': data['Cum_prod'] }))


    res_df = pd.concat(res, ignore_index=True)

    df_to_csv_int(res_df, 'oversampling_results')
    pass



def plot_overs_explo(p=r'data\int\M_oversampling\oversampling_results.csv' ):

    df = pd.read_csv(p)
    df['Var'] = df['Var'].map(replace_target)

    p = (ggplot(df, aes(x='y', fill='Type')) + geom_density(alpha=0.5) + facet_wrap('Var', scales='free_y', ncol=3) + theme_minimal()+ labs(x='log10(Cum_prod)', y='Density'))

    save_fig_plotnine(p, 'oversampling_explo', w=10, h=4, dpi=800)
    pass



if __name__ == '__main__':
    explo_oversampling()
    plot_overs_explo()
    pass