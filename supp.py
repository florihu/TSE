
import pandas as pd
import os


from util import df_to_excel
# Purpose

'''
Bind all the intermediate data into a supplementary excel

'''

# Params



file_dict = {
    
    'S1_cumulative_production_model': [
        {'path': r'data\int\D_build_sample_sets\target_vars_prio_source.csv',
                 'sheet_name': 'Priorization_data_source',
                 'note': 'Priorization flag for data from Werner et al., or Captial IQ.'}, 

                {'path': r'data\int\M_cumprod_mc_confidence\cumprod_mc_confidence.csv',
                 'sheet_name': 'Cumprod_mc_confidence',
                 'note': 'Confidence intervals for the cumulative production model derived by Monte Carlo method.'},

    ],

    'S2_machine_learning_model': [
        {'path': r'data\int\E_ml_explo\correlation_results.csv',
                 'sheet_name': 'Correlation_results',
                 'note': 'Correlation results for the exploration data.'},

                 {'path': r'data\int\E_ml_explo\normality_test_results.csv',
                 'sheet_name': 'Normality_test_results',
                 'note': 'Normality test results for the variables in the exploration data.'},

                 {'path': r'data\int\M_geography_feature\geo_sim_X_pred.csv.csv',
                  'sheet_name': 'Geo_sim_pred',
                  'note': 'The Geographical similarties for the prediction set'},

                    {'path': r'data\int\M_geography_feature\geo_sim.csv.csv',
                     'sheet_name': 'Geography_featue_train_test',
                     'note': 'The Geographical similarties for the training and testing set'},

                     {'path': r'data\int\M_ml_hypeopt\ml_hype_opt_results_synth.csv',
                      'sheet_name': 'Hype_opt_results',
                      'note': 'Hyperparameter optimization results for the synthetic data.'},

                    {'path': r'data\int\M_ml_hypeopt\ml_hype_opt_results.csv',
                     'sheet_name': 'Hype_opt_results_synth',
                     'note': 'Hyperparameter optimization results for the synthetic data.'},

                     {'path':r'data\int\M_ml_train_loop\ml_train_loop_result.csv',
                      'sheet_name': 'Train_loop_result',
                      'note': 'Results of the training loop.'},

                      {'path':r'data\int\M_ml_train_loop\ml_train_loop_result_synth.csv',
                       'sheet_name': 'Train_loop_result_synth',
                       'note': 'Results of the training loop for the synthetic data.'},

                    {'path':r'data\int\M_oversampling\oversampling_results.csv',
                     'sheet_name': 'Oversampling_results',
                     'note': 'Results of the oversampling target variable specific.'},

                     {'path':r'data\int\R_ml_hype_error\std_errors_hype_model.csv',
                      'sheet_name': 'Std_errors_hype_model',
                      'note': 'Standard errors for the hyperparameter model.'},      

                      {'path':r'data\int\R_prediction\best_model_prediction.csv.csv',
                       'sheet_name': 'Best_model_prediction',
                       'note': 'Predictions of the best model.'},            

                    ],

    'S3_allocation':[

        {'path': r'data\int\M_alloc\alloc_com.csv.csv',
                    'sheet_name': 'Alloc_to_com',
                    'note': 'Allocation factors for commodity instances.'},


                ]
                 
                 }  

def bind_all_into_supp_excel():
    '''
    Bind all the intermediate data into a supplementary excel
    '''
    # create the supp folder if not existent

    for key, value in file_dict.items():

        # make for every df a sheet with sheet name and note
        df_note = pd.DataFrame({'Sheet_name': [file['sheet_name'] for file in value], 'Note': [file['note'] for file in value]})
        df_to_excel(key, df_note, 'Sheet_overview')

        for file in value:
            p = rf'{file["path"]}'

            df = pd.read_csv(p)
            df_to_excel(key, df, file['sheet_name'])

            print(f'File {file["path"]} has been added to the excel file {key} in sheet {file["sheet_name"]}')
        


    return None


if __name__ == '__main__':
    bind_all_into_supp_excel()