import os, sys, yaml, re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from YamlLoader import YamlLoader

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
        # TODO: np.sin np.cos seem to not be in np 1.18.5
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    ) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, space_size, d_model, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.d_model = d_model
        self.embedding = layers.Embedding(input_dim=space_size, output_dim=d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)


    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)


    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        # TODO: tf.math, tf.cats, tf.newaxis can't be imported for some reason? (tf.newaxis seems to not be in tf 2.3)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x