def train():

    import os
    import uuid
    import glob

    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    import torch
    import torch.nn.functional as f
    from torch.utils.data import SubsetRandomSampler

    from mlp_scripts import config
    from mlp_scripts.data_prep_utils import get_processed_data, get_loader
    from mlp_scripts.models import get_model, get_optimizer_from_config, get_lr_scheduler_from_config
    from mlp_scripts.train_utils import visualize_regression, visualize_learning, RegressionTrainer

    def initialize_trainer(snapshot_dir):
        n = get_model()
        o = get_optimizer_from_config(n)
        s = get_lr_scheduler_from_config(o)
        f = RegressionTrainer(
            n, torch.nn.MSELoss(), o, s, epoch=config.EPOCH_PER_K_FOLD, snapshot_dir=snapshot_dir,
            verbose=True, progress=False, log_interval=1
        )
        f.to(device)
        return f

    dataset = get_processed_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("training by oof...\n")

    kf = KFold(n_splits=config.NUM_K_FOLD, random_state=777, shuffle=True)

    predictions = []
    ground_truth = []
    model_state_dicts = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        print("<Fold %s>" % fold)
        snapshot_dir = os.path.join(checkpoint_dir, f'{uuid.uuid4()}')
        os.makedirs(snapshot_dir, exist_ok=True)
        fitter = initialize_trainer(snapshot_dir=snapshot_dir)
        train_result, test_result = fitter.fit(
            get_loader(dataset, sampler=SubsetRandomSampler(train_idx), batch_size=10, drop_last=True),
            get_loader(dataset, sampler=SubsetRandomSampler(val_idx)),
            split_result=True
        )
        best_snapshot = sorted(glob.glob(os.path.join(snapshot_dir, 'best_checkpoint_epoch_*.pt')))[-1]
        state_dict = torch.load(best_snapshot)
        model_state_dicts.append(state_dict['model'])
        fitter.load_state_dict(state_dict)
        model = fitter.model
        model.eval()

        with torch.no_grad():
            for x, y in get_loader(dataset, sampler=SubsetRandomSampler(val_idx), batch_size=50):
                t = model(x.to(device))
                y = y.to(device)
                predictions.append(t)
                ground_truth.append(y)
            predictions = torch.cat(predictions, dim=0).view(-1, 1).cpu()
            ground_truth = torch.cat(ground_truth, dim=0).view(-1, 1).cpu()
            print(f"Whole data: {len(predictions)} predictions\n")
            visualize_learning(
                train_result,
                test_result,
                title=f"Fold {fold} fitting curve",
                figsize=(15, 12),
                filename=f"output_loss_fold{fold}.jpg",
                show=True,
            )

    with torch.no_grad():
        # transform to original
        sc_val = dataset.y_proc.scaler_
        label = sc_val.inverse_transform(ground_truth.cpu().numpy())  # numpy
        prediction = sc_val.inverse_transform(predictions.cpu().numpy())  # numpy
        outputs = torch.from_numpy(prediction).to(device).view(-1, 1)  # pytorch
        labels = torch.from_numpy(label).to(device).view(-1, 1)  # pytorch

        mse = f.mse_loss(outputs, labels).item()
        mae = f.l1_loss(outputs, labels).item()

    r2 = r2_score(label, prediction)

    with open("output.txt", "w") as f:
        f.write(f'MSE: {mse:.4f}\tMAE: {mae:.4f}\tTrain R2 score: {r2:.4f}\n')

    visualize_regression(
        label, prediction, mse_score=None, mae_score=None, r2_score=None,
        plot_max=200, plot_min=0, vmax=40., vmin=0., alpha=0.5, figsize=(15, 12),
        xlabel='Value', ylabel='Prediction', title='Year of Survival Regression',
        filename=f"output_train_regression.png", show=True
    )

    torch.save(model_state_dicts, 'state_dict_list.pt')


if __name__ == '__main__':
    train()
