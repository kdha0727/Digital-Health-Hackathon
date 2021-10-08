import torch


@torch.no_grad()
def test():

    from SALib.sample import saltelli
    from SALib.analyze import sobol

    import torch
    import torch.nn as nn
    import pandas as pd

    import matplotlib.pyplot as plt
    import seaborn as sns

    from mlp_scripts import config
    from mlp_scripts.models import get_model

    sns.set_style('white')
    sns.set_context('talk')

    class Ensemble(nn.Sequential):

        def __init__(self, state_dicts):
            models = []
            for state_dict in state_dicts:
                model = get_model(channels=config.CHANNELS)
                model.load_state_dict(state_dict)
                models.append(model)
            super().__init__(*models)
            self.eval()

        @torch.no_grad()
        def forward(self, x):
            result = [net(x) for net in self]
            return sum(result) / len(result)

    def get_problem_and_param_values(n=1024):
        problem = {
            'num_vars': 311,
            'names': [
                'Treatment',
                *('Var{}'.format(i) for i in range(1, 10 + 1)),
                *('G{}'.format(i) for i in range(1, 300 + 1))
            ],
            'bounds': [
                          (0, 2)
                      ] + [
                          (0, 1)
                      ] * 10 + [
                          (0, 2)
                      ] * 300
        }
        param_values = saltelli.sample(problem, n, calc_second_order=True)
        for i in (0, *range(11, 300 + 11)):
            param_values[:, i] = param_values[:, i] >= 1
        return problem, param_values

    def simulate(model, param_values):
        import torch
        device = torch.device('cuda')
        x = torch.from_numpy(param_values).float().to(device)
        model.to(device)
        with torch.no_grad():
            y = model(x)
        return y.cpu().numpy().reshape(-1)

    model = Ensemble(torch.load('state_dict_list.pt'))

    problem, param_values = get_problem_and_param_values()
    Y = simulate(model, param_values)
    Si = sobol.analyze(problem, Y)
    Si_df = pd.DataFrame({k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}, index=problem['names'])
    Si_df.sort_values(by='ST', ascending=False).to_csv('sensitivity_analysis.csv')

    fig, ax = plt.subplots(1)
    indices = Si_df[['S1', 'ST']]
    err = Si_df[['S1_conf', 'ST_conf']]
    indices.plot.bar(yerr=err.values.T, ax=ax)
    fig.set_size_inches(160, 16)
    plt.savefig('sensitivity_analysis.png')

    def get_input(num_gene, force_variable_ident=False):
        assert isinstance(num_gene, int) and 1 <= num_gene <= 300

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if force_variable_ident:
            from mlp_scripts.data_prep_utils import get_processed_data
            df = get_processed_data().samples
            mean = df.mean(0)[['Var{}'.format(i) for i in range(1, 10 + 1)]]
            mean_np = mean.to_numpy().reshape(-1)
            std = df.std(0)[['Var{}'.format(i) for i in range(1, 10 + 1)]]
            std_np = std.to_numpy().reshape(-1)

        gene_idx = num_gene + 10
        param_treat = param_values[param_values[:, 0] == 1]

        for has_gene in (1, 0):
            param_gene = param_treat.copy()
            param_gene[:, gene_idx] = int(has_gene)
            if force_variable_ident:
                param_gene[:, 1:11] = mean_np
            yield torch.from_numpy(param_gene).float().to(device)

    def extract_genes(model, force_variable_ident=False):

        import pandas as pd
        from scipy import stats

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)

        index_col = []
        ttest_tvalue_col = []
        ttest_pvalue_col = []
        average_gene_pos_col = []
        average_gene_neg_col = []
        tau1_col = []

        for gene in range(1, 300 + 1):
            print(f"\r{gene}", end="")

            pos_input, neg_input = get_input(gene, force_variable_ident=force_variable_ident)

            with torch.no_grad():
                pos_result = model(pos_input).cpu().squeeze().numpy()
                neg_result = model(neg_input).cpu().squeeze().numpy()

                # levene = stats.levene(gene_1, gene_2)
                # equal_var = (levene.pvalue >= 0.05)
                ttest = stats.ttest_rel(pos_result, neg_result)
                # print(f"{(gene, ttest.pvalue)})")

                if ttest.pvalue < 0.05:
                    pos = pos_result.mean()
                    neg = neg_result.mean()
                    tau1 = (pos_result - neg_result).mean()

                    index_col.append('G{}'.format(gene))
                    ttest_tvalue_col.append(ttest.statistic)
                    ttest_pvalue_col.append(ttest.pvalue)
                    average_gene_pos_col.append(pos)
                    average_gene_neg_col.append(neg)
                    tau1_col.append(tau1)
                    # print((pos_result - neg_result)[:5])

        df = pd.DataFrame(
            {
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

    return extract_genes(model, force_variable_ident=True).to_csv('result.csv')


if __name__ == '__main__':
    test()
