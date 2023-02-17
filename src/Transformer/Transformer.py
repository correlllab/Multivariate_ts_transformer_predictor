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
                 target_space_size, dropout_rate=0.1, decode=True):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                                num_heads=num_heads, ff_dim=ff_dim,
                                space_size=input_space_size,
                                dropout_rate=dropout_rate)

        self.decode = decode
        if self.decode:
            self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                                    num_heads=num_heads, ff_dim=ff_dim,
                                    space_size=target_space_size,
                                    dropout_rate=dropout_rate)

        self.final_layer = layers.Dense(target_space_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x  = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        if self.decode:
            x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_space_size)

        if self.decode:
            try:
                # Drop the keras mask, so it doesn't scale the losses/metrics.
                # b/250038731
                del logits._keras_mask
            except AttributeError:
                pass

        # Return the final output and the attention weights.
        return logits
