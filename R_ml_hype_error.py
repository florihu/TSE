
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from E_pca import get_data_per_var
from M_ml_train_loop import get_comb
from util import df_to_csv_int
########################################################Purpose#######################################################################



########################################################Params#######################################################################




model_paths = {'Tailings_production': 'models\SVR_Tailings_production.pkl', 'Concentrate_production': 'models\SVR_Concentrate_production.pkl', 'Ore_processed_mass': 'models\SVR_Ore_processed_mass.pkl'}



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

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(indices, comb):
            # Predictions
            y_train_pred, y_test_pred = model.predict(X.iloc[train_idx]), model.predict(X.iloc[test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Standardized errors
            std_error_train = StandardScaler().fit_transform((y_train - y_train_pred).values.reshape(-1, 1)).flatten()
            std_error_test = StandardScaler().fit_transform((y_test - y_test_pred).values.reshape(-1, 1)).flatten()

            # Append results
            res_df.append(
                pd.DataFrame({
                    "Variable": [variable] * len(y_train),
                    "Sample": ["Train"]*len(y_train),
                    "Y_pred": y_train_pred,
                    "Y_true": y_train.values,
                    "Std_error": std_error_train,
                }))
            
            res_df.append(
                pd.DataFrame({
                    "Variable": [variable] * len(y_test),
                    "Sample": ["Test"] * len(y_test),
                    "Y_pred": y_test_pred,
                    "Y_true": y_test.values,
                    "Std_error": std_error_test,
                }))
    
    res = pd.concat(res_df, ignore_index=True, axis = 0)

    # Save results
    df_to_csv_int(res, "std_errors_hype_model")

    pass


if __name__ == '__main__':
    calc_std_errors()