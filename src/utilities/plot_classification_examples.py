import os, sys, json
sys.path.append(os.path.realpath('../'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf

from utilities.makespan_utils import scan_output_for_decision
from utilities.utils import set_size


def classify(model: tf.keras.Model, episode: np.ndarray, true_label: float, window_width: int, confidence: float = 0.9, ts_s: float = 20.0/1000.0):
    with tf.device('/GPU:0'):
        time_steps = episode.shape[0]
        n_windows = time_steps - window_width + 1
        episode_X_list = []
        for i in range(window_width, n_windows):
            # episode_window = episode[i:i + window_width, :]
            episode_window = episode[i - window_width:i, :]
            episode_X_list.append(episode_window[:, 1:7])
        episode_X = tf.stack(episode_X_list)
        prediction = model.predict(episode_X, use_multiprocessing=True)
        ans, t_c, row = scan_output_for_decision(
            np.array(prediction),
            np.array([tf.keras.utils.to_categorical(true_label, num_classes=2)] * len(prediction)),
            threshold=confidence
        )

    # Return the classification answer and the time (in seconds) at which the classification happened
    return ans, (window_width * ts_s) + (t_c * ts_s), prediction


def plot_ft_classification_for_model(model_names, models, episodes, confidence=0.9):
    ts_s = 20.0 / 1000.0
    ts_ms = 20
    rolling_window_width = int(7.0 * 50.0)
    win_width = 10
    FzCol = 3
    spikeThresh = 0.05

    fig_width = 1000
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
    fig, axes = plt.subplots(5, 3, figsize=set_size(fig_width, subplots=(5, 3)))

    i = 0
    for episode in episodes:
        episode_steps = episode.shape[0]

        fx = episode[:, 1]
        fy = episode[:, 2]
        fz = episode[:, 3]
        tx = episode[:, 4]
        ty = episode[:, 5]
        tz = episode[:, 6]

        axes[0, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), fx, label='F_x')
        axes[0, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), fy, label='F_y')
        axes[0, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), fz, label='F_z')
        axes[0, i].legend()
        axes[0, i].set_title('Force -vs- Time')
        axes[0, i].set_xlabel('Time in milliseconds')
        axes[0, i].set_ylabel('Force')

        axes[1, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), tx, label='T_x')
        axes[1, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), ty, label='T_y')
        axes[1, i].plot(np.arange(0, episode_steps * ts_ms, ts_ms), tz, label='T_z')
        axes[1, i].legend()
        axes[1, i].set_title('Torque -vs- Time')
        axes[1, i].set_xlabel('Time in milliseconds')
        axes[1, i].set_ylabel('Torque')

        j = 2
        for model_name, model in zip(model_names, models):
            n_steps = episode.shape[0]
            episode_time = n_steps * ts_s
            chopDex = 0
            for bgn in range(int(1.5*50), n_steps - win_width + 1):
                end     = bgn+win_width
                FzSlice = episode[ bgn:end, FzCol ].flatten()
                if np.abs( np.amax( FzSlice ) - np.amin( FzSlice ) ) >= spikeThresh:
                    chopDex = end
                    break
            if (chopDex * ts_s) < 15.0:
                ep_matrix = episode[chopDex:, :]
                if len(ep_matrix) - rolling_window_width + 1 > rolling_window_width:
                    true_label = 0.0
                    if ep_matrix[0, 7] == 0.0:
                        true_label = 1.0
                    ans, t_c, preds = classify(
                        model=model,
                        episode=ep_matrix,
                        true_label=true_label,
                        window_width=rolling_window_width,
                        confidence=confidence,
                        ts_s=ts_s
                    )

            N = preds.shape[0]
            X = np.arange(0, episode_steps*ts_ms, ts_ms)
            L = preds.transpose()
            n_empty_values = X.shape[0] - L.shape[1]
            for _ in range(n_empty_values):
                L = np.insert(L, 0, 0, axis=1)
            # print(L.shape, X.shape)
            axes[j, i].stackplot(X, *L, labels=('Pr(Pass)', 'Pr(Fail)'), baseline='zero')
            axes[j, i].axhline(y=0.9, color='black', linestyle='--')
            axes[j, i].axhline(y=0.1, color='black', linestyle='--')
            axes[j, i].legend(loc='center left')
            axes[j, i].set_title(f'{model_name}: {ans}')
            axes[j, i].set_xlabel('Time in milliseconds')
            axes[j, i].set_ylabel('Probability')


            j += 1
        i += 1

    plt.tight_layout()
    plt.savefig('../saved_data/imgs/ft_classification.png')


if __name__ == '__main__':
    DATA = ['reactive', 'training']
    DATA_DIR = f'../../data/data_manager/{"_".join(DATA)}'
    print('Loading data from file...', end='')
    with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)
    print('DONE')

    print(f'Number of episodes in test data = {len(test_data)}')

    ep_indices = [0, 6, 8]
    episode_list = []
    for idx in ep_indices:
        episode_list.append(test_data[idx])

    plot_ft_classification_for_model(
        model_names=[],
        models=None,
        episodes=episode_list
    )