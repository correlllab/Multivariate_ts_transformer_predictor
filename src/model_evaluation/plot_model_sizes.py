import json
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed


def plot_model_sizes(sizes: dict):
    fig, ax = plt.subplots(figsize=(15, 10))
    # sort dict (greater to lower)
    sizes = sorted(sizes.items(), key=lambda item: item[1], reverse=False)
    colors = ['b', 'r', 'g', 'y']
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
            f'{sizes[i][0]}\n{sizes[i][1]}',
            xy=(extra_spacing + sizes[i][1], 0),
            fontsize=15,
            ha='center',
            va='center'
        )

    ax.set_aspect('equal')
    plt.axis('off')
    plt.xlim((-space, extra_spacing + 2 * sizes[-1][1] + space))
    plt.ylim((-sizes[0][-1] * 2, sizes[0][-1] * 2))
    # plt.legend()
    plt.title('Number of parameters')
    plt.savefig('../saved_data/imgs/evaluation/model_sizes.png')


if __name__ == '__main__':
    with open('../saved_data/model_sizes.json', 'r') as f:
        model_sizes = json.load(f)

    plot_model_sizes(sizes=model_sizes)
