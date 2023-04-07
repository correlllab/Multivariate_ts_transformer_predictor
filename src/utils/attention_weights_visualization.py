import sys, os
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import matplotlib.pyplot as plt

from data_management.data_preprocessing import DataPreprocessing
# from CustomSchedule import CustomSchedule
from Transformer.Transformer import Transformer



def plot_attention_head(window, attention):
    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(window)))
    ax.set_yticks(range(len(window)))


def plot_attention_weights(window, attention_heads):
    for h, head in enumerate(attention_heads[0]):
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))

        # ax = fig.add_subplot(2, 2, h+1)

        # plot_attention_head(window=window, attention=head)
        shw = axes.matshow(head)
        bar = plt.colorbar(shw)
        # axes.xticks(range(len(window)))
        # axes.yticks(range(len(window)))

        axes.set_title(f'Head {h+1}')
        # axes.set(xlabel=None)
        # axes.set(ylabel=None)

        plt.tight_layout()
        plt.savefig(f'./imgs/attention/attention_mat_head_{h}.png')



class Predictor(tf.Module):
    def __init__(self, model, name='Predictor'):
        super().__init__(name)
        self.model = model

    def __call__(self, window):
        # TODO: more fancy behavior?
        prediction, attention_weights = self.model.evaluate(window)
        # attention_weights = self.model.encoder.last_attn_scores

        return prediction, attention_weights



if __name__ == '__main__':
    # transformer = tf.saved_model.load('../fcn_vs_transformer/models/OOP_transformer')
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
            # episode_label = tf.stack(tf.keras.utils.to_categorical(episode_label_list, num_classes=2))
            if episode_window[0, 7] == 0.0:
                true_label = 1.0
            elif episode_window[0, 7] == 1.0:
                true_label = 0.0

            ep = tf.expand_dims(episode_X[0], axis=0)

            # prediction, attn_weigths = predictor(tf.expand_dims(episode_X[0], axis=0))
            # prediction, attn_weights = transformer(tf.expand_dims(episode_X[0], axis=0), training=False)
            # plot_attention_weights(window=episode_X[0], attention_heads=transformer.encoder.last_attn_scores[0])
            break

    prediction, attn_weigths = transformer(ep)
    plot_attention_weights(window=episode_X[0], attention_heads=attn_weigths)

