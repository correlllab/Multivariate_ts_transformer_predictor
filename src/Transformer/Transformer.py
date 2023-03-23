import os, sys, yaml, re
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from YamlLoader import YamlLoader
from MultiHeadAttention import MultiHeadAttention
from Encoder import Encoder
from Decoder import Decoder

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# Full transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, ff_dim, input_space_size,
                 target_space_size, pos_encoding=True, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                                num_heads=num_heads, ff_dim=ff_dim,
                                space_size=input_space_size,
                                dropout_rate=dropout_rate,
                                pos_encoding=pos_encoding)

        self.flatten = layers.Flatten()
        self.final_layer = layers.Dense(target_space_size)

    def call(self, inputs):
        # print(f'In Transformer call, shape = {inputs.shape}')
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # x = tf.keras.Input(shape=inputs.shape)

        x = self.encoder(inputs)  # (batch_size, context_len, d_model)

        # Final linear layer output.
        x = self.flatten(x)
        logits = self.final_layer(x)  # (batch_size, target_len, target_space_size)
        # print(f'==> In Transformer call, logits.shape = {logits.shape}')

        # Return the final output and the attention weights.
        return logits
