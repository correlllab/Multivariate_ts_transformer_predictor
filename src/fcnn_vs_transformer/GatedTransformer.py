import os, sys
sys.path.insert(1, os.path.realpath('..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

from Transformer.AttentionLayers import *
from Transformer.Decoder import *
from Transformer.Encoder import *
from Transformer.FeedForwardLayer import *
from Transformer.PositionalEncoding import *
from Transformer.TimeStepEncoder import *

class GatedTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, input_space_size,
                 target_space_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers, d_model=d_model,
            num_heads=num_heads, ff_dim=ff_dim,
            space_size=input_space_size,
            dropout_rate=dropout_rate
        )

        self.time_step_encoder = TimeStepEncoder(
            num_layers=num_layers, d_model=d_model,
            num_heads=num_heads, ff_dim=ff_dim,
            space_size=input_space_size,
            dropout_rate=dropout_rate
        )

        # TODO: create Gate layer
        self.gate = None

        self.linear_layer = layers.Dense(target_space_size, activation='relu')

        self.softmax_layer = layers.Softmax()

    def call(self, inputs):
        # inputs = tf.keras.Input(shape=inputs.shape)
        context, x = inputs

        print(f'\ncontext shape = {context.shape}; x shape = {x.shape}\n')
        # Time channel tower
        context = self.encoder(context)

        # Time step tower
        x = self.time_step_encoder(x, context)

        # TODO
        # x = self.gate(x)

        x = self.linear_layer(x)

        logits = self.softmax_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        return logits