import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, SimpleRNN, LSTM, GRU, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

import os, sys
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import seaborn as sns
import pandas as pd

from utils import CounterDict
from helper_functions import scan_output_for_decision, graph_episode_output


class Transformer:
    def __init__(self) -> None:
        self.model = None
        self.history = None

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


    def build_model(
        self,
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=2
    ):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)
    

    def fit(self, X_train, Y_train, X_test, Y_test, trainWindows, epochs=200, save_model=True):
        input_shape = X_train.shape[1:]
        self.build_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["categorical_accuracy"],
        )
        self.model.summary()

        