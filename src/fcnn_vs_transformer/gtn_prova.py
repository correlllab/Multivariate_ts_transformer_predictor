import os, sys
sys.path.insert(1, os.path.realpath('..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

import tensorflow_lattice as tfl

from Transformer.AttentionLayers import *
from Transformer.Decoder import *
from Transformer.Encoder import *
from Transformer.FeedForwardLayer import *
from Transformer.PositionalEncoding import *
from Transformer.TimeStepEncoder import *
from Transformer.Embedding import *


class GTN:
    def __init__(self) -> None:
        self.model = None
        self.history = None
        self.evaluation = None
        self.file_name = './models/GTN.keras'
        self.imgs_path = './imgs/GTN/'

        self.timestep_embedding = Embedding(d_feature=16, d_timestep=128, dropout=0,
                                            d_model=128, wise='timestep')
        self.feature_embedding = Embedding(d_feature=16, d_timestep=128, dropout=0,
                                            d_model=128, wise='feature')
        self.gate = tfl.layers.Linear(num_input_dims=(128 * 512 + 16 * 512), units=2)
        self.linear_out = tfl.layers.Linear(num_input_dims=(128 * 512 + 16 * 512), units=2)


    def gtn_timestep_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, use_causal_mask=True)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res
    

    def gtn_feature_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, use_causal_mask=False)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
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
        n_classes=2,
        stage='train' or 'test'
    ):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        x_timestep, _ = self.timestep_embedding(x)
        x_feature, _ = self.feature_embedding(x)

        for _ in range(num_transformer_blocks):
            x_timestep = self.gtn_timestep_encoder(x, head_size, num_heads, ff_dim, dropout)
            x_feature = self.gtn_feature_encoder(x, head_size, num_heads, ff_dim, dropout)

        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
        x_feature = x_feature.reshape(x_feature.shape[0], -1)

        gate = tf.nn.softmax(self.gate(tf.concat([x_timestep, x_feature], axis=-1)), axis=-1)

        gate_out = tf.concat([x_timestep * gate[:, 0:1], x_feature * gate[:, 1:2]], axis=-1)

        out = self.linear_out(gate_out)

        self.model = tf.keras.Model(inputs, out)


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
            loss="binary_focal_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["categorical_accuracy"],
        )
        self.model.summary()

        checkpoint_filepath = './models/tmp/checkpoints/'
        callbacks = [
            # Early stopping might xause overfitting
            #tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='val_categorical_accuracy',
                mode='max',
                save_best_only=True
            )
        ]

        self.history = self.model.fit(
            X_train,
            Y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
        )

        self.evaluation = self.model.evaluate(X_test, Y_test, verbose=1)

        if save_model:
            self.model.save(self.file_name)