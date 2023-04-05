import os, sys, yaml, re
import numpy as np
import tensorflow as tf
# from YamlLoader import YamlLoader
# from MultiHeadAttention import MultiHeadAttention
from Encoder import Encoder
# from Decoder import Decoder

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# Full transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, ff_dim, mlp_units,
                 input_space_size, target_space_size, training, pos_encoding=True, dropout_rate=0.1):
        super().__init__()

        self.training = training

        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               ff_dim=ff_dim,
                               space_size=input_space_size,
                               dropout_rate=dropout_rate,
                               pos_encoding=pos_encoding)
        
        self.mlp = tf.keras.Sequential()
        for dim in mlp_units:
            self.mlp.add(tf.keras.layers.Dense(dim, activation='relu'))
            self.mlp.add(tf.keras.layers.Dropout(dropout_rate))

        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')
        self.flatten = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(target_space_size, activation='softmax')

    def call(self, inputs):
        # print(f'In Transformer call, shape = {inputs.shape}')
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        # x = tf.keras.Input(shape=inputs.shape)

        x = self.encoder(inputs, self.training)  # (batch_size, context_len, d_model)

        # Global average pooling for temporal data
        x = self.global_average_pooling(x)

        # Flattening data
        # x = self.flatten(x)

        # MLP net
        x = self.mlp(x)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_space_size)
        # print(f'==> In Transformer call, logits.shape = {logits.shape}')

        # Return the final output and the attention weights.
        return logits
