import sys, os
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from data_management.data_preprocessing import DataPreprocessing
from Transformer import Transformer


class Predictor(tf.Module):
    def __init__(self, model, name='Predictor'):
        super().__init__(name)
        self.model = model

    def __call__(self, window):
        # TODO: more fancy behavior?
        prediction = self.model(window)
        attention_weights = self.model.encoder.last_attn_scores

        return prediction, attention_weights


class ExportModel(tf.Module):
    def __init__(self, model, name=None):
        super().__init__(name)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 350, 6), dtype=tf.float64)])
    def __call__(self, window):
        (result, attention_weights) = self.model(window)

        return result, attention_weights



def plot_attention_weights(window, attention_heads):
    for h, head in enumerate(attention_heads[0]):
        fx = window[0][:, h]
        g = sns.JointGrid(x=fx, y=fx, height=10)
        sns.heatmap(head, ax=g.ax_joint, xticklabels=False, yticklabels=False, cmap="YlGnBu", cbar_kws=dict(location="left"))
        sns.lineplot(fx, ax=g.ax_marg_x)
        sns.lineplot(x=fx, y=list(range(len(fx))), ax=g.ax_marg_y)
        g.ax_marg_x.set_xlabel('Time')
        g.ax_marg_y.set_ylabel('Time')
        g.ax_marg_y.xaxis.tick_top()
        g.ax_marg_y.set_xlim(np.min(fx), np.max(fx))
        g.ax_marg_y.set_ylim(len(fx), 0)
        g.ax_joint.set_xlabel('Time')
        g.ax_joint.set_ylabel('Time')
        g.ax_joint.set_title(f'Head {h+1}')
        sns.despine()
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.9)
        plt.savefig(f'../fcn_vs_transformer/imgs/attention/attention_mat_head_{h}.png')
        plt.clf()

        # fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        # axes.axis('off')

        # # ax = fig.add_subplot(2, 2, h+1)

        # # plot_attention_head(window=window, attention=head)
        # axes.matshow(head)
        # # bar = plt.colorbar(shw)

        # axes.plot(fx, np.zeros_like(fx)-0.5, lw=2)
        # axes.plot(np.zeros_like(fx)-0.5, fx, lw=2)

        # # axes.set_xticks(range(len(fx)))
        # # axes.set_yticks(range(len(fx)))

        # axes.set_title(f'Head {h+1}')
        # # axes.set(xlabel=None)
        # # axes.set(ylabel=None)

        # plt.tight_layout()
        # plt.savefig(f'../fcn_vs_transformer/imgs/attention/attention_mat_head_{h}.png')


if __name__ == '__main__':
    num_layers = 4
    d_model = 6
    dff = 512
    num_heads = 4
    dropout_rate = 0.1
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=dff,
        input_space_size=6,
        target_space_size=2,
        dropout_rate=dropout_rate,
        pos_encoding=True
    )
    # learning_rate = CustomSchedule()
    opt = tf.keras.optimizers.legacy.Adam(1e-4, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)
    transformer.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    transformer.load_weights('../fcn_vs_transformer/models/OOP_transformer').expect_partial()
    predictor = Predictor(model=transformer)
    predictor = ExportModel(model=predictor)

    dp = DataPreprocessing(sampling='none')
    dp.shuffle = False
    dp.load_data(verbose=True)
    dp.scale_data(verbose=True)
    dp.set_episode_beginning(verbose=True)

    print(type(transformer))

    ep = None
    episode_predictions = []
    rolling_window_width = int(7.0 * 50)
    print( f"Each window is {rolling_window_width} timesteps long!" )
    for ep_index, episode in enumerate(dp.truncData):
        with tf.device('/GPU:0'):
            print(f'Episode {ep_index}')
            time_steps = episode.shape[0]
            n_windows = time_steps - rolling_window_width + 1
            episode_X_list = []
            # episode_label_list = []
            true_label = None
            for i in range(n_windows):
                # TODO: make the window centered (i.e. i +- (window_width / 2)) ??
                episode_window = episode[i:i + rolling_window_width, :]
                episode_X_list.append(episode_window[:, 1:7])
                # Need to change this because it is also done in training data
                # if episode_window[0, 7] == 0.0:
                #     episode_label_list.append(1.0)
                # elif episode_window[0, 7] == 1.0:
                #     episode_label_list.append(0.0)
                # episode_X = episode_window[:, 1:7]
                # episode_label = episode_window[0, 7]

            episode_X = tf.stack(episode_X_list)
            # episode_label = tf.stack(tf.keras.utilities.to_categorical(episode_label_list, num_classes=2))
            if episode_window[0, 7] == 0.0:
                true_label = 1.0
            elif episode_window[0, 7] == 1.0:
                true_label = 0.0

            ep = tf.expand_dims(episode_X[0], axis=0)

            # prediction, attn_weigths = predictor(tf.expand_dims(episode_X[0], axis=0))
            # prediction, attn_weights = transformer(tf.expand_dims(episode_X[0], axis=0), training=False)
            # plot_attention_weights(window=episode_X[0], attention_heads=transformer.encoder.last_attn_scores[0])
            break

    prediction, attn_weigths = predictor(ep)
    plot_attention_weights(window=ep, attention_heads=attn_weigths)