import json
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex

sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

from utilities.utils import set_size


model_name_table = {
    'FCN': 'FCN', 
    'RNN': 'RNN',
    'GRU': 'GRU',
    'LSTM': 'LSTM',
    'Transformer': 'Transformer',
    'VanillaTransformer': 'Vanilla\nTransformer',
    'OOP_Transformer_small': 'OOP Transformer\n(small)',
    'OOP_Transformer': 'OOP Transformer (big)'
    }


def plot_model_sizes(sizes: dict, out_name: str = ''):
    # Setup
    # plt.style.use('seaborn')
    # From Latex \textwidth
    fig_width = 600
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
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

    fig, ax = plt.subplots(figsize=set_size(fig_width))
    fig.tight_layout()
    # sort dict (greater to lower)
    sizes = sorted(sizes.items(), key=lambda item: item[1], reverse=False)
    # colors = ['r', 'g', 'b', 'y']
    cmap = cm.get_cmap('Spectral', len(sizes))
    colors = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    space = 10000
    extra_spacing = 0
    for i in range(len(sizes)):
        if i > 0:
            extra_spacing += 2 * sizes[i-1][1] + space

        ax.add_patch(
            plt.Circle(
                (extra_spacing + sizes[i][1], 0),
                sizes[i][1],
                color=colors[i],
                label=sizes[i][0],
                alpha=0.7
            )
        )
        ax.annotate(
            f'{model_name_table[sizes[i][0]]}\n{sizes[i][1]}',
            xy=(extra_spacing + sizes[i][1], 0),
            fontsize=15,
            ha='center',
            va='center'
        )

    ax.set_aspect('equal')
    plt.axis('off')
    plt.xlim((-space, extra_spacing + 2 * sizes[-1][1] + space))
    plt.ylim((-sizes[-1][-1] * 1.15, sizes[-1][-1] * 1.15))
    # plt.legend()
    plt.title('Number of parameters')
    plt.tight_layout()
    if out_name == '':
        plt.savefig('../saved_data/imgs/evaluation/model_sizes.png')
    else:
        plt.savefig(f'../saved_data/imgs/evaluation/{out_name}.png')


if __name__ == '__main__':
    with open('../saved_data/model_sizes.json', 'r') as f:
        model_sizes = json.load(f)

    plot_model_sizes(sizes=model_sizes)
