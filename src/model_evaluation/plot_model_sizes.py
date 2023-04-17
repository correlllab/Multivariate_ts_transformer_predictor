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


model_name_table = {
    'FCN': 'FCN', 
    'RNN': 'RNN',
    'VanillaTransformer': 'Vanilla\nTransformer',
    'OOP_Transformer_small': 'OOP Transformer\n(small)',
    'OOP_Transformer': 'OOP Transformer (big)'
    }


def plot_model_sizes(sizes: dict):
    fig, ax = plt.subplots(figsize=(15, 10))
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
    plt.savefig('../saved_data/imgs/evaluation/model_sizes.png')


if __name__ == '__main__':
    with open('../saved_data/model_sizes.json', 'r') as f:
        model_sizes = json.load(f)

    plot_model_sizes(sizes=model_sizes)
