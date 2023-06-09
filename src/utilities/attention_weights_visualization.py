import sys
import os
import shutil
import numpy as np
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from data_management.data_preprocessing import DataPreprocessing
from model_builds.OOPTransformer import OOPTransformer
from utilities.utils import CounterDict
from utilities.makespan_utils import *


class EpisodePerf:
    """ Container class for per-episode performance """
    def __init__( self, trueLabel = -1, answer = 'NC', runTime = float('nan'),
                  TTP = float('nan'), TTN = float('nan'), TTS = float('nan'), TTF = float('nan') ):
        self.trueLabel = trueLabel
        self.answer    = answer
        self.runTime   = runTime
        self.TTP       = TTP
        self.TTN       = TTN
        self.TTS       = TTS
        self.TTF       = TTF


def scan_output_for_decision( output, trueLabel, threshold = 0.90 ):
    for i, row in enumerate( output ):
        if np.amax( row ) >= threshold:
            if np.dot( row, trueLabel[i] ) >= threshold:
                ans = 'T'
            else:
                ans = 'F'
            if row[0] > row[1]:
                ans += 'P'
            else:
                ans += 'N'
            return ans, i, row
    return 'NC', output.shape[0], row


def monitored_makespan( MTS, MTF, MTN, P_TP, P_FN, P_TN, P_FP, P_NCS, P_NCF ):
    """ Closed form makespan for monitored case, symbolic simplification from Mathematica """
    return (1 + MTF*(P_FP + P_NCF) + MTS*(P_NCS + P_TP) + MTN*(P_FN + P_TN) ) / (1 - P_FN - P_FP - P_TN - P_NCF)


def plot_attention_weights(attention_heads, episode_num, time_step, res, values):
    for h, head in enumerate(attention_heads[0]):
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))

        shw = axes.matshow(head)
        bar = plt.colorbar(shw)

        axes.set_title(f'Episode {episode_num}, Head {h+1}\nPredicted {res} at time {time_step} ms ({values})')

        plt.tight_layout()
        dir_path = f'../saved_data/imgs/attention/episode_{episode_num}/'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        plt.savefig(dir_path + f'attention_mat_ep_{episode_num}_head_{h}.png')
        plt.clf()
        plt.close("all")


def plot_attention_on_time_series(episode, attention_heads, episode_num, time_step_ms, ts_s, ts_ms):
    print(f'Classification time step = {time_step_ms} ms')
    
    # Big-picture plot
    fx = episode[:, 1]
    fy = episode[:, 2]
    fz = episode[:, 3]
    tx = episode[:, 4]
    ty = episode[:, 5]
    tz = episode[:, 6]

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ax[0].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), fx, linewidth=1, linestyle='solid', label='Force x')
    ax[0].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), fy, linewidth=1, linestyle='dotted', label='Force y')
    ax[0].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), fz, linewidth=1, linestyle='dashed', label='Force z')
    ax[0].axvspan(time_step_ms - (350 * ts_ms), time_step_ms, color='gray', alpha=0.3)
    ax[0].legend()

    ax[1].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), tx, linewidth=1, linestyle='solid', label='Torque x')
    ax[1].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), ty, linewidth=1, linestyle='dotted', label='Torque y')
    ax[1].plot(np.arange(0, episode.shape[0] * ts_ms, ts_ms), tz, linewidth=1, linestyle='dashed', label='Torque z')
    ax[1].axvspan(time_step_ms - (350 * ts_ms), time_step_ms, color='gray', alpha=0.3)
    ax[1].legend()
    plt.savefig(f'../saved_data/imgs/attention/episode_{episode_num}/ep_{episode_num}_big_picture.png')
    plt.clf()
    plt.close("all")

    time_step = int(time_step_ms / 20.0)

    # Zoomed plot with attention visualization
    fx_trunc = episode[time_step - 350:time_step, 1]
    fy_trunc = episode[time_step - 350:time_step, 2]
    fz_trunc = episode[time_step - 350:time_step, 3]
    tx_trunc = episode[time_step - 350:time_step, 4]
    ty_trunc = episode[time_step - 350:time_step, 5]
    tz_trunc = episode[time_step - 350:time_step, 6]

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    ax[0].plot(fx_trunc, linewidth=1, color='gray', linestyle='solid', label='Force x')
    ax[0].plot(fy_trunc, linewidth=1, color='gray', linestyle='dotted', label='Force y')
    ax[0].plot(fz_trunc, linewidth=1, color='gray', linestyle='dashed', label='Force z')
    ax[0].legend()

    ax[1].plot(tx_trunc, linewidth=1, color='gray', linestyle='solid', label='Torque x')
    ax[1].plot(ty_trunc, linewidth=1, color='gray', linestyle='dotted', label='Torque y')
    ax[1].plot(tz_trunc, linewidth=1, color='gray', linestyle='dashed', label='Torque z')
    ax[1].legend()

    heads = [None for _ in range(attention_heads.shape[1])]
    colors = ['blue', 'green', 'yellow', 'red']
    for head_num in range(attention_heads.shape[1]):
        head = np.array(attention_heads[0, head_num, :, :])
        max_val = np.max(head)
        max_val_coords = np.unravel_index(head.argmax(), head.shape)
        max_val_row = head[:, max_val_coords[0]]
        max_val_col = head[:, max_val_coords[1]]

        # for i, value in enumerate(max_val_row):
        #     if value >= 0.9 * max_val:
        #         ax[0].axvspan(i, i, color=colors[head_num], alpha=0.7, label=f'Head {head_num}')
        #         ax[1].axvspan(i, i, color=colors[head_num], alpha=0.7, label=f'Head {head_num}')

        first = True
        for i, value in enumerate(max_val_col):
            if value >= 0.9 * max_val and first:
                heads[head_num] = ax[0].axvspan(i, i, color=colors[head_num], alpha=0.7)
                ax[1].axvspan(i, i, color=colors[head_num], alpha=0.7)
                first = False
            elif value >= 0.9 * max_val:
                ax[0].axvspan(i, i, color=colors[head_num], alpha=0.7)
                ax[1].axvspan(i, i, color=colors[head_num], alpha=0.7)

    fig.legend(heads, ['Head 1', 'Head 2', 'Head 3', 'Head 4'])
    fig.suptitle(f'Episode {episode_num}, Head {head_num}: At time step {max_val_coords[1]}, attention is focused on colored strips')
    plt.savefig(f'../saved_data/imgs/attention/episode_{episode_num}/head_{head_num}_ep_{episode_num}_truncated.png')
    plt.clf()
    plt.close("all")


def clean_imgs_dir(directory='../saved_data/imgs/attention/'):
    for folder in os.listdir(directory):
        shutil.rmtree(directory + folder)


class Predictor(tf.Module):
    def __init__(self, model, name='Predictor'):
        super().__init__(name)
        self.model = model

    def __call__(self, window):
        # TODO: more fancy behavior?
        prediction, attention_weights = self.model.evaluate(window)
        # attention_weights = self.model.encoder.last_attn_scores

        return prediction, attention_weights


DATA = ['reactive', 'training']
DATA_DIR = f'../../data/instance_data/{"_".join(DATA)}'

if __name__ == '__main__':
    print('\nLoading data from files...', end='')

    with open(f'{DATA_DIR}/{"_".join(DATA)}_X_train.npy', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)

    # transformer = tf.saved_model.load('../fcn_vs_transformer/models/OOP_transformer')
    transformer_net = OOPTransformer(model_name='Small_Transformer')

    num_layers = 4
    d_model = 6
    ff_dim = 256
    num_heads = 4
    head_size = 128
    dropout_rate = 0.2
    mlp_dropout = 0.4
    mlp_units = [128]

    transformer_net.build(
        X_sample=X_train[:64],
        num_layers=num_layers,
        d_model=d_model,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_size=head_size,
        dropout_rate=dropout_rate,
        mlp_dropout=mlp_dropout,
        mlp_units=mlp_units,
        verbose=False
    )
    transformer_net.model.load_weights('../saved_models/OOP_Transformer_small/').expect_partial()
    predictor = Predictor(model=transformer_net.model)

    print(type(transformer_net.model))

    print(f'Cleaning images directory...', end='')
    clean_imgs_dir()
    print('DONE')
    print(f'Contents of dir: {os.listdir("../saved_data/imgs/attention/")}')

    ts_s = 20.0 / 1000.0
    ts_ms = 20
    rolling_window_width = int(7.0 * 50.0)
    win_width = 10
    FzCol = 3
    spikeThresh = 0.05
    ep_indices = [1, 6, 36, 61, 96]
    for ep_index in ep_indices:
        with tf.device('/GPU:0'):
            print(f'----------------------------\nProcessing episode {ep_index}')
            episode = test_data[ep_index]
            n_steps = episode.shape[0]
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
                    n_windows = ep_matrix.shape[0] - rolling_window_width + 1
                    for i in range(rolling_window_width, n_windows):
                        episode_window = ep_matrix[i - rolling_window_width:i, :]
                        pred = transformer_net.model(tf.expand_dims(episode_window[:, 1:7], axis=0), training=False)
                        ans, t_c, row = scan_output_for_decision(
                            output=np.array(pred),
                            trueLabel=np.array([tf.keras.utils.to_categorical(true_label, num_classes=2)]),
                            threshold=0.9
                        )
                        if ans != 'NC':
                            print(f'{i}: {ans}')
                            break

                    t_c_ms = (i * ts_ms)

                    attn_weights = transformer_net.model.encoder.last_attn_scores
                    plot_attention_weights(
                        attention_heads=attn_weights,
                        episode_num=ep_index,
                        time_step=t_c_ms,
                        res=ans,
                        values=row
                    )
                    plot_attention_on_time_series(
                        episode=episode,
                        attention_heads=attn_weights,
                        episode_num=ep_index,
                        time_step_ms=t_c_ms,
                        ts_s=ts_s,
                        ts_ms=ts_ms
                    )

    # ep = None
    # episode_predictions = []
    # rolling_window_width = int(7.0 * 50)
    # print( f"Each window is {rolling_window_width} timesteps long!" )
    # for ep_index, episode in enumerate(dp.truncData):
    #     with tf.device('/GPU:0'):
    #         print(f'Episode {ep_index}')
    #         time_steps = episode.shape[0]
    #         n_windows = time_steps - rolling_window_width + 1
            
    #         if episode[0, 7] == 0.0:
    #             true_label = 1.0

    #             for i in range(n_windows):
    #                 # TODO: make the window centered (i.e. i +- (window_width / 2)) ??
    #                 episode_window = episode[i:i + rolling_window_width, :]
    #                 # episode_X_list.append(episode_window[:, 1:7])
    #                 pred = transformer_net.model(tf.expand_dims(episode_window[:, 1:7], axis=0), training=False)
    #                 ans, ad_x, row = scan_output_for_decision(
    #                     output=np.array(pred),
    #                     trueLabel=np.array([tf.keras.utils.to_categorical(true_label, num_classes=2)]),
    #                     threshold=0.9
    #                 )
    #                 if ans != 'NC':
    #                     break

    #             attn_weights = transformer_net.model.encoder.last_attn_scores
    #             plot_attention_weights(
    #                 attention_heads=attn_weights,
    #                 episode_num=ep_index,
    #                 time_step=ad_x,
    #                 res=ans,
    #                 values=row
    #             )
    #             plot_attention_on_time_series(
    #                 episode=episode,
    #                 attention_heads=attn_weights,
    #                 episode_num=ep_index,
    #                 time_step=ad_x
    #             )
    #         else:
    #             continue
