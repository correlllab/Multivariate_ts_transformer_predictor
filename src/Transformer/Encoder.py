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

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# Encoder layer
class EncoderLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, ff_dim)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.self_attention.last_attn_scores

        return x


# Full encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                ff_dim, space_size, dropout_rate=0.1, pos_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.encode = pos_encoding
        if self.encode:
            self.pos_embedding = PositionalEmbedding(
                space_size=space_size,
                d_model=d_model
            )

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                            num_heads=num_heads,
                            ff_dim=ff_dim,
                            dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)

        self.last_attn_scores = None

    def call(self, x):
        # print(f'In Encoder call, shape = {x.shape}')
        if self.encode:
            # `x` is token-IDs shape: (batch, seq_len)
            x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        # print(f'==> In Encoder call, last x.shape = {x.shape}')

        self.last_attn_scores = self.enc_layers[-1].last_attn_scores

        return x  # Shape `(batch_size, seq_len, d_model)`.