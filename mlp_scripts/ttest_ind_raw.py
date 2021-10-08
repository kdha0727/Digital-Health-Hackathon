import numpy as np
import pandas as pd
from scipy import stats

from data_prep_utils import get_raw_data

def extrapolate(survival_time_event, scale_time=False):
    if "extrapolated" in survival_time_event.columns:
        return survival_time_event
    extrapolated_survival_time = []
    for row_idx in range(survival_time_event.shape[0]):
        survival_time = survival_time_event.loc[row_idx, "time"]
        if survival_time_event.loc[row_idx, "event"] == 0:
            new_survival_time = survival_time_event.loc[
                (survival_time_event["event"] == 1) &
                (survival_time_event["time"] > survival_time), "time"].mean()
            extrapolated_survival_time.append(new_survival_time)
        else:
            extrapolated_survival_time.append(survival_time)
    if scale_time:
        from sklearn.preprocessing import MinMaxScaler
        extrapolated_survival_time = MinMaxScaler(feature_range=(-1, 1), copy=True, clip=False).fit_transform(np.array(extrapolated_survival_time).reshape(-1, 1))
    survival_time_event["extrapolated"] = extrapolated_survival_time
    # survival_time_event.to_csv('./data/survival_time_extrapolation.csv')
    return survival_time_event


def get_pandas_data(scale_time=False):
    # import libraries
    clinical_variables, generic_alterations, survival_time_event, treatment = get_raw_data()
    # extrapolate survival 
    survival_time_event = extrapolate(survival_time_event, scale_time=scale_time)
    # concat
    full = pd.concat([treatment, clinical_variables, generic_alterations, survival_time_event[['extrapolated']]], axis=1)
    # return processed dataset
    return full
  
  
def pretest(scale_time=False, levene_threshold=0.05):

    samples = get_pandas_data(scale_time=scale_time)
    gene_idx = tuple(map(lambda key: 'G{}'.format(key), range(1, 300 + 1)))
    
    index_col = []
    levene_pvalue_col = []
    equal_var_col = []
    ttest_tvalue_col = []
    ttest_pvalue_col = []
    average_gene_pos_col = []
    average_gene_neg_col = []
    tau1_col = []

    for gene in gene_idx:

        df = samples[[gene, 'Treatment', 'extrapolated']]
        unique_df = df[[gene, 'Treatment']].drop_duplicates()

        if len(unique_df[unique_df.Treatment == 1]) == 2:  # tau1
            looking = df[df.Treatment == 1]
            
            gene_1 = looking[looking[gene] == 1].extrapolated
            gene_2 = looking[looking[gene] == 0].extrapolated
            
            levene = stats.levene(gene_1, gene_2)
            equal_var = (levene.pvalue >= levene_threshold)
            ttest = stats.ttest_ind(gene_1, gene_2)
            
            if ttest.pvalue < 0.05:
                pos = gene_1.mean()
                neg = gene_2.mean()
                tau1 = pos - neg
                
                index_col.append(gene)
                levene_pvalue_col.append(levene.pvalue)
                equal_var_col.append(equal_var)
                ttest_tvalue_col.append(ttest.statistic)
                ttest_pvalue_col.append(ttest.pvalue)
                average_gene_pos_col.append(pos)
                average_gene_neg_col.append(neg)
                tau1_col.append(tau1)
                
        else:
            raise RuntimeError

    df = pd.DataFrame(
        {
            'Levene\'s test p-value': levene_pvalue_col,
            'Assert variance equailty': equal_var_col,
            't-test t-value': ttest_tvalue_col,
            't-test p-value': ttest_pvalue_col,
            'Pos Gene Average': average_gene_pos_col,
            'Neg Gene Average': average_gene_neg_col,
            'Tau1 Score': tau1_col,
        },
        index=index_col
    )
    
    df = df.sort_values(by="Tau1 Score", ascending=False)

    return df


if __name__ == '__main__':
    pretest().to_csv('ttest_ind_raw.csv')
    pretest(scale_time=True).to_csv('ttest_ind_raw_scaled_time.csv')
