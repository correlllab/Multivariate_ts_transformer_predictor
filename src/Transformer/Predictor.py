import os, sys, glob
sys.path.insert(1, os.path.realpath('../fcn_vs_transformer'))
print(sys.version)
print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preprocessing import DataPreprocessing


def plot_attention_head(window, attention):
  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(window)))
  ax.set_yticks(range(len(window)))


def plot_attention_weights(window, attention_heads):
  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(window, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()



class Predictor(tf.Module):
    def __init__(self, model, name='Predictor'):
        super().__init__(name)
        self.model = model

    def __call__(self, window):
        # TODO: more fancy behavior?
        prediction = self.model.evaluate(window)
        attention_weights = self.model.encoder.last_attn_scores

        return prediction, attention_weights


if __name__ == '__main__':
    transformer = tf.saved_model.load('../fcn_vs_transformer/models/OOP_transformer')
    predictor = Predictor(model=transformer)

    dp = DataPreprocessing(sampling='none')
    dp.shuffle = False
    dp.load_data(verbose=True)
    dp.scale_data(verbose=True)
    dp.set_episode_beginning(verbose=True)

    print(type(transformer))

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

            prediction, attn_weigths = predictor(episode_X)
            plot_attention_weights(window=episode_X, attention_heads=attn_weigths[0])
            break

