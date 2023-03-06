import os, sys, yaml, re
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from YamlLoader import YamlLoader
from MultiHeadAttention import MultiHeadAttention

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# FeedForward net for both encoder and decoder
class FeedForward(layers.Layer):
    def __init__(self, d_model, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.seq = tfk.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate)
        ])
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, x):
        print(f'X shape = {x.shape}; self.seq(x) shape = {self.seq(x).shape}')
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x