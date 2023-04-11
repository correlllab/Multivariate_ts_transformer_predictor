import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils import CounterDict
from utils.helper_functions import scan_output_for_decision, graph_episode_output


class VanillaTransformer:
    def __init__(self) -> None:
        self.model_name = 'VanillaTransformer'
        self.model = None
        self.history = None
        self.evaluation = None
        self.last_attn_scores = None
        self.file_name = f'../saved_models/{self.model_name}.keras'
        self.imgs_path = f'../saved_data/imgs/{self.model_name}/'
        self.histories_path = f'../saved_data/histories/{self.model_name}_history'


    def positional_encoding(self, s):
        # x = np.zeros(s)
        # pe = tf.ones_like(x, dtype=tf.float32)
        # position = tf.expand_dims(tf.range(0., x.shape[0]), axis=-1)
        # temp = tf.range(0., x.shape[-1], delta=2.)
        # temp = tf.multiply(temp, -(tf.divide(math.log(10000), x.shape[-1])))
        # temp = tf.expand_dims(tf.math.exp(temp), axis=0)
        # temp = tf.linalg.matmul(position, temp)  # shape:[input, d_model/2]
        # pe = tf.Variable(pe, dtype=tf.float32, validate_shape=False)
        # pe[:, 0::2].assign(tf.math.sin(temp))
        # pe[:, 1::2].assign(tf.math.cos(temp))
        # pe = tf.convert_to_tensor(pe, dtype=tf.float32)

        # return pe

        aux = np.zeros(s)
        mat = np.arange(s[0], dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, s[-1], 2, dtype=np.float32) / s[-1])
        aux[:, 0::2] = np.sin(mat)
        aux[:, 1::2] = np.cos(mat)
        pe = tf.convert_to_tensor(aux, dtype=tf.float32)

        # return tf.keras.layers.Add()([x, pe])
        return pe


    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        x, attn_scores = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, return_attention_scores=True)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        # x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res, attn_scores


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
        x += self.positional_encoding(input_shape[-2:])
        for _ in range(num_transformer_blocks):
            x, attn_scores = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        self.last_attn_scores = attn_scores

        x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        for dim in mlp_units:
            x = tf.keras.layers.Dense(dim, activation="relu")(x)
            x = tf.keras.layers.Dropout(mlp_dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
        self.model = tf.keras.Model(inputs, outputs)


    def fit(self, X_train, Y_train, X_test, Y_test, trainWindows, batch_size=64, epochs=200, save_model=True):
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

        checkpoint_filepath = '../saved_models/tmp/vanilla_transformer_checkpoints/'
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.2
            )
        ]

        if save_model:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=False,
                    monitor='val_categorical_accuracy',
                    mode='max',
                    save_best_only=True
                )
            )

        self.history = self.model.fit(
            x=X_train,
            y=Y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data  = (X_test, Y_test),
            steps_per_epoch = len(X_train) // batch_size,
            validation_steps = len(X_test) // batch_size
        )

        # self.evaluation = self.model.evaluate(X_test, Y_test, verbose=1)

        if save_model:
            self.model.save(self.file_name)

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)
