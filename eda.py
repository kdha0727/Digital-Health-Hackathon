import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


#######
# PCA #
#######

def get_data():
    clinical_variables = pd.read_csv('./data/Clinical_Variables.csv')
    generic_alterations = pd.read_csv('./data/Genetic_alterations.csv')
    survival_time_event = pd.read_csv('./data/Survival_time_event.csv')
    treatment = pd.read_csv('./data/Treatment.csv')

    return clinical_variables, generic_alterations, survival_time_event, treatment


# get data
clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
n_data = treatment.shape[0]  # row size
n_genes = generic_alterations.shape[1] - 1  # gene count

# group people by their survived time & treatment
sorted_survival_time_event = survival_time_event.sort_values(by="time")
n_groups = 2
group_label = [0 for _ in range(n_data)]
for k in range(n_groups):
    start = int(n_data / n_groups * k)
    end = int(n_data / n_groups * (k + 1))
    for idx in range(start, end):
        had_treatment = treatment.loc[idx, "Treatment"]
        group_label[sorted_survival_time_event.iloc[idx, 0]] = f'{k}_treatment' if had_treatment else k

# remove alive data
generic_alterations = generic_alterations.loc[survival_time_event["event"] == 1]
treatment = treatment.loc[survival_time_event["event"] == 1]
group_label = [group_label[idx] for idx in range(len(group_label)) if idx in treatment.index]

generic_alterations = generic_alterations.iloc[:, 1:]

# pca
pca = PCA(n_components=2)
principal_components = pca.fit_transform(generic_alterations)
principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
principal_df["group"] = group_label

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
groups = set(group_label)
for group in groups:
    indices_to_keep = principal_df["group"] == group
    ax.scatter(principal_df.loc[indices_to_keep, 'pc1'], principal_df.loc[indices_to_keep, 'pc2'], s=20)
ax.legend(groups)
plt.savefig("outputs/pca.png")


#########
# WGCNA #
#########

def find_rel_nodes(node_idx, group):
    vt[node_idx] = 1
    for other_node_idx in range(300):
        if not vt[other_node_idx] and W[node_idx][other_node_idx] > 0.9:
            group.append(other_node_idx)
            find_rel_nodes(other_node_idx, group)
    return group


clinical_variables, generic_alterations, survival_time_event, treatment = get_data()
A = [[0 for _ in range(300)] for _ in range(300)]
beta = 12
amp = 1
for gene_idx in range(300):
    gene_name = f'G{gene_idx + 1}'
    gene_col = generic_alterations[gene_name]
    for comparing_gene_idx in range(300):
        comparing_gene_name = f'G{comparing_gene_idx + 1}'
        comparing_gene_col = generic_alterations[comparing_gene_name]
        corr = gene_col.corr(comparing_gene_col)
        A[gene_idx][comparing_gene_idx] = (0.5 + corr / 2) ** beta * amp

L = np.dot(A, A)
k = [sum(row) for row in A]
W = [[0 for _ in range(300)] for _ in range(300)]
for i in range(300):
    L[i][i] -= 1
for i in range(300):
    for j in range(300):
        W[i][j] = (L[i][j] + A[i][j]) / (min(k[i], k[j]) + 1 - A[i][j])
        if i != j:
            W[i][j] *= 1000

vt = [0 for _ in range(300)]
groups = []
for idx in range(300):
    if not vt[idx]:
        groups.append(find_rel_nodes(idx, [idx]))
