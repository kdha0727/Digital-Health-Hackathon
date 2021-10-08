import matplotlib.pyplot as plt
import numpy as np


def visualize_learning(
        # Learning arguments
        train_result,
        test_result,
        # Plot arguments
        title=None,
        figsize=None,
        # Save arguments
        filename=None,
        show=True,
):
    """Plot Learning Curve to check over fitting"""
    if figsize is None:  # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(train_result) + 1), train_result, label='Training Loss')
    if train_result is not test_result:
        plt.plot(range(1, len(test_result) + 1), test_result, label='Test Loss')
    min_val_loss_idx = test_result.index(min(test_result))
    plt.axvline(min_val_loss_idx + 1, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.annotate('epoch: {}\nloss : {:.4f}'.format(min_val_loss_idx + 1, test_result[min_val_loss_idx]),
                 xy=[min_val_loss_idx + 1, test_result[min_val_loss_idx]],
                 xytext=[min_val_loss_idx + 20, test_result[min_val_loss_idx] + 1e-2],
                 bbox=dict(boxstyle='square', color='grey'), fontsize=13, arrowprops=dict(facecolor='blue'))

    if title is not None:
        plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


def visualize_early_stopping_epochs(
        # Learning arguments
        epochs,
        # Plot arguments
        title=None,
        figsize=None,
        # Save arguments
        filename=None,
        show=True,
):
    if figsize is None:  # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')
    cm = (np.array(range(len(epochs), 0, -1)) - 0.5) / len(epochs)
    plt.figure(figsize=figsize)
    plt.bar(range(1, len(epochs) + 1), epochs, color=plt.cm.rainbow(cm), label='Early Stopping Epochs')
    plt.axhline(sum(epochs) / len(epochs), linestyle='--', color='r', label='Average Epoch')

    if title is not None:
        plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Average Epoch")
    plt.grid(True)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()


def visualize_regression(
        # Regression arguments
        label,
        prediction,
        mse_score=None,
        mae_score=None,
        r2_score=None,
        # Regression range
        plot_max=1,
        plot_min=0,
        # Plot arguments
        vmax=None,
        vmin=None,
        alpha=0.5,
        figsize=None,
        xlabel=None,
        ylabel=None,
        title=None,
        # Save arguments
        filename=None,
        show=True,
):

    if figsize is None:  # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if vmax is None:
        vmax = vmax or (plot_max - plot_min) / 5

    if vmin is None:
        vmin = 0

    plt.figure(figsize=figsize)

    plt.plot([plot_min, plot_max], [plot_min, plot_max], color='black')
    plt.scatter(
        label, prediction,
        c=np.abs(label - prediction), vmax=vmax, vmin=vmin, cmap='jet', alpha=alpha
    )
    plt.colorbar(label='Error')

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    text_args = []
    if mse_score is not None:
        text_args.append("MSE score: {:.4f}".format(mse_score))
    if mae_score is not None:
        text_args.append("MAE score: {:.4f}".format(mae_score))
    if r2_score is not None:
        text_args.append("R2 score: {:.4f}".format(r2_score))

    plt.text(
        (plot_max + plot_min) / 2, plot_min,
        '\n'.join(text_args),
        fontsize=25,
        bbox={'boxstyle': 'square', 'ec': (1,1,1), 'fc': (1,1,1), 'linestyle': '--', 'color': 'black'}
    )

    plt.grid(True)
    plt.minorticks_on()

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
