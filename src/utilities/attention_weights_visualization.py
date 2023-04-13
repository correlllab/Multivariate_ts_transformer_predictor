import sys
import os
import numpy as np
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import matplotlib.pyplot as plt

from data_management.data_preprocessing import DataPreprocessing
from model_builds.OOPTransformer import OOPTransformer
from utilities.utils import CounterDict


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


def expected_makespan( MTF, MTN, MTS, P_TP, P_FN, P_TN, P_FP ):
    """ Return the expected makespan for the given parameters """
    return -( MTF*P_FP + MTN*P_FN + MTN*P_TN + MTS*P_TP + 1) / (P_FN + P_FP + P_TN - 1)


def plot_attention_weights(window, attention_heads, episode_num):
    for h, head in enumerate(attention_heads[0]):
        fig, axes = plt.subplots(1, 1, figsize=(20, 10))

        shw = axes.matshow(head)
        bar = plt.colorbar(shw)

        axes.set_title(f'Episode {episode_num}, Head {h+1}')

        plt.tight_layout()
        dir_path = f'../saved_data/imgs/attention/episode_{episode_num}/'
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        plt.savefig(dir_path + f'attention_mat_head_{h}.png')


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
    dp = DataPreprocessing(sampling='none', data='reactive')
    dp.shuffle = False
    # dp.load_data(verbose=True)
    # dp.scale_data(verbose=True)
    # dp.set_episode_beginning(verbose=True)
    dp.run(save_data=False, verbose=True)

    # transformer = tf.saved_model.load('../fcn_vs_transformer/models/OOP_transformer')
    transformer_net = OOPTransformer()

    num_layers = 8
    d_model = 6
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    mlp_units = [128, 256, 64]

    transformer_net.build(
        X_sample=dp.X_train_sampled[:32],
        num_layers=num_layers,
        d_model=d_model,
        dff=dff,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        mlp_units=mlp_units,
        save_model=True
    )
    # learning_rate = CustomSchedule()
    # opt = tf.keras.optimizers.legacy.Adam(1e-4, beta_1=0.9, beta_2=0.98,
    #                                       epsilon=1e-9)
    transformer_net.compile()
    transformer_net.model.load_weights('../saved_models/OOP_Transformer/').expect_partial()
    predictor = Predictor(model=transformer_net.model)

    print(type(transformer_net.model))

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

            episode_X = tf.stack(episode_X_list)
            # episode_label = tf.stack(tf.keras.utilities.to_categorical(episode_label_list, num_classes=2))
            if episode_window[0, 7] == 0.0:
                true_label = 1.0
            elif episode_window[0, 7] == 1.0:
                true_label = 0.0

            ep = tf.expand_dims(episode_X[0], axis=0)

            # prediction, attn_weigths = predictor(tf.expand_dims(episode_X[0], axis=0))
            # prediction, attn_weights = transformer_net(tf.expand_dims(episode_X[0], axis=0), training=False)
            # plot_attention_weights(window=episode_X[0], attention_heads=transformer_net.encoder.last_attn_scores[0])
            break


    prediction = transformer_net.model(ep)
    attn_weights = transformer_net.model.encoder.last_attn_scores
    plot_attention_weights(window=episode_X[0], attention_heads=attn_weights, episode_num=0)

