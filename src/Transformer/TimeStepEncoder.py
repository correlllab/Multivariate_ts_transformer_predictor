import os, sys, yaml, re
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from YamlLoader import YamlLoader
from MultiHeadAttention import MultiHeadAttention
from Transformer.AttentionLayers import BaseAttention, GlobalSelfAttention, CrossAttention, MultiHeadAttention, CausalSelfAttention
from Transformer.FeedForwardLayer import FeedForward
from Transformer.PositionalEncoding import PositionalEmbedding


class TimeStepEncoderLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()

        self.masked_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, ff_dim)


    def call(self, x):
        x = self.masked_attention(x)
        x = self.ffn(x)
        return x


class TimeStepEncoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 ff_dim, space_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            space_size=space_size,
            d_model=d_model
        )

        self.enc_layers = [
            TimeStepEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)


    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x
