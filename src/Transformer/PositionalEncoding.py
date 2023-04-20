import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def positional_encoding(self, s):
        aux = np.zeros(s)
        mat = np.arange(s[0], dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, s[-1], 2, dtype=np.float32) / s[-1])
        aux[:, 0::2] = np.sin(mat)
        aux[:, 1::2] = np.cos(mat)
        pe = tf.convert_to_tensor(aux, dtype=tf.float32)

        return pe


    def call(self, x):
        x += self.positional_encoding(x.shape[-2:])
        return x