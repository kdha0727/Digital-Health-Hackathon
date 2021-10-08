"""
Final DataFrame Formation
For process.R
"""

import numpy as np
import pandas as pd


if __name__ != '__main__':
    raise ImportError("This module cannot be imported")


# Load CSV Files
treatment = pd.read_csv('./data/Treatment.csv', index_col=0)
clinical_vars = pd.read_csv('./data/Clinical_Variables.csv', index_col=0)
genetic_vars = pd.read_csv('./data/Genetic_alterations.csv', index_col=0)
survival = pd.read_csv('./data/survival_time_event.csv', index_col=0)

treatment[treatment == 0] = 'No'
treatment[treatment == 1] = 'Yes'

genetic_vars[genetic_vars == 0] = 'No'
genetic_vars[genetic_vars == 1] = 'Yes'

extrapolated_survival_time = []
for row_idx in range(survival.shape[0]):
    survival_time = survival.loc[row_idx, "time"]
    if survival.loc[row_idx, "event"] == 0:
        new_survival_time = survival.loc[
            (survival["event"] == 1) &
            (survival["time"] > survival_time), "time"].mean()
        extrapolated_survival_time.append(new_survival_time)
    else:
        extrapolated_survival_time.append(survival_time)
survival["extrapolated"] = extrapolated_survival_time

final_df = pd.concat([
    treatment.iloc[:,0],
    clinical_vars.iloc[:,0:],
    genetic_vars.iloc[:,0:],
    survival.iloc[:,-1]
], axis=1)
final_df.rename(columns={'extrapolated': 'time'}, inplace=True)

final_df.to_csv('./data_processed/bayesian_preprocess.csv', index=False)
