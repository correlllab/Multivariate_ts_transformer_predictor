import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

from utilities.utils import set_size


# function to add value labels
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, f'{y[i]:.2f}%', ha='center',
                 bbox=dict(facecolor='white', alpha=.5))


def nc_bar_plot(labels, vals, save_plot=True):
    # Setup
    plt.style.use('seaborn')
    # From Latex \textwidth
    fig_width = 345
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 14,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)

    plt.figure(figsize=set_size(fig_width))
    plt.tight_layout()
    plt.bar(x=labels, height=vals)
    add_labels(names, values)

    plt.xlabel('Model')
    plt.ylabel('Percentage of NC')
    plt.title(
        'NC rate of the models with unseen data during training, 0.9 threshold for decision making (preemptive dataset)'
    )

    if save_plot:
        plt.savefig('../saved_data/imgs/evaluation/models_NC_rates.png')
        plt.clf()
    else:
        plt.show()


def stacked_conf_mat_bar_plot(conf_mat: dict, model_name: str, save_plot: bool = True):
    # Setup
    plt.style.use('seaborn')
    # From Latex \textwidth
    fig_width = 345
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 14,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)

    x = ['No classification', 'Actual Positives', 'Actual negatives']
    fig, axes = plt.subplots(figsize=set_size(fig_width))
    fig.tight_layout()
    nc = [conf_mat['NC'], 0, 0]
    tp = np.array([0, conf_mat['TP'], 0])
    fn = np.array([0, conf_mat['FN'], 0])
    fp = np.array([0, 0, conf_mat['FP']])
    tn = np.array([0, 0, conf_mat['TN']])
    axes.bar(x, nc, alpha=0.8, label='NC')
    axes.bar(x, tp, alpha=0.8, bottom=nc, label='TP')
    axes.bar(x, fn, alpha=0.8, bottom=nc + tp, label='FN')
    axes.bar(x, tn, alpha=0.8, bottom=nc + tp + fn, label='TN')
    axes.bar(x, fp, alpha=0.8, bottom=nc + tp + fn + tn, label='FP')

    for bar in axes.patches:
        if bar.get_height() != 0:
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_y() - (bar.get_height() / 2),
                f'{bar.get_height() * 100:.2f}%',
                ha='center',
                bbox=dict(facecolor='white', alpha=.5),
                fontsize=15
            )

    axes.legend()
    plt.ylim(top=1.1)
    plt.title(f'{model_name} confusion matrix (preemptive dataset)')

    if save_plot:
        plt.savefig(f'../saved_data/imgs/evaluation/{model_name}_stacked_conf_mat_preemptive.png')
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    model_names = ['FCN', 'RNN', 'VanillaTransformer', 'OOP_Transformer_small', 'OOP_Transformer']
    confusion_matrices = dict.fromkeys(model_names)

    counter = 0
    values = []
    names = []
    for model_name in model_names:
        try:
            with open(f'../saved_data/evaluation_results/{model_name}_conf_matrix.json', 'r') as f:
                confusion_matrices[model_name] = json.load(f)
            counter += 1
            values.append(confusion_matrices[model_name]['NC'] * 100)
            names.append(model_name)
        except FileNotFoundError as e:
            print(f'\nERROR: {e} (confusion matrix does not exist for this model)\n')

    print(confusion_matrices)

    nc_bar_plot(labels=names, vals=values)
    for name, conf_mat in confusion_matrices.items():
        if conf_mat is not None:
            stacked_conf_mat_bar_plot(conf_mat=conf_mat, model_name=name)
