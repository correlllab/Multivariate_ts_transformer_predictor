import sys, os, pickle, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from keras import backend as K

from utils.utils import CounterDict
from utils.helper_functions import scan_output_for_decision, graph_episode_output
from Transformer.AttentionLayers import *
from data_management.data_preprocessing import DataPreprocessing



class GTN:
    def __init__(self) -> None:
        self.model = None
        self.history = None
        self.evaluation = None
        self.file_name = './models/Transformer.keras'
        self.imgs_path = './imgs/transformer/'


    def timestep_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs, use_causal_mask=True)
        # res = tf.keras.layers.LayerNormalization(tf.keras.layers.Add()([x, inputs]))
        res = tf.keras.layers.LayerNormalization()(x + inputs)

        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        # return tf.keras.layers.LayerNormalization(tf.keras.layers.Add()([x, res]))
        return tf.keras.layers.LayerNormalization()(x + res)


    def feature_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        x = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        # res = tf.keras.layers.LayerNormalization(tf.keras.layers.Add()([x, inputs]))
        res = tf.keras.layers.LayerNormalization()(x + inputs)

        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        # return tf.keras.layers.LayerNormalization(tf.keras.layers.Add()([x, res]))
        return tf.keras.layers.LayerNormalization()(x + res)


    def position_encode(self, x):
        aux = np.zeros(x.shape)
        mat = np.arange(x.shape[1], dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, x.shape[-1], 2, dtype=np.float32) / x.shape[-1])
        aux[:, :, 0::2] = tf.math.sin(mat)
        aux[:, :, 1::2] = tf.math.cos(mat)
        pe = tf.convert_to_tensor(aux, dtype=tf.float32)

        return x + pe


    def embedding(self, x, d_model, d_feature, d_timestep, wise):
        embedding = None
        if wise == 'feature':
            embedding =  tf.keras.layers.Dense(d_model, input_shape=(d_timestep,), activation=None)(x)
        elif wise == 'timestep':
            embedding = tf.keras.layers.Dense(d_model, input_shape=(d_feature,), activation=None)(x)
            embedding = self.position_encode(embedding)

        return embedding


    def build_model(
        self,
        d_model,
        d_feature,
        d_timestep,
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        batch_size,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        class_num=2
    ):
        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32, batch_size=batch_size)

        x_timestep = self.embedding(x=tf.transpose(inputs, perm=[0,2,1]), d_model=d_model, d_feature=d_feature, d_timestep=d_timestep, wise='timestep')
        x_feature = self.embedding(x=inputs, d_model=d_model, d_feature=d_feature, d_timestep=d_timestep, wise='feature')

        for _ in range(num_transformer_blocks):
            x_timestep = self.timestep_encoder(inputs=x_timestep, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim)
            x_feature = self.feature_encoder(inputs=x_feature, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim)

        x_timestep = tf.reshape(x_timestep, shape=(x_timestep.shape[0], -1))
        x_feature = tf.reshape(x_feature, shape=(x_feature.shape[0], -1))

        gate = tf.nn.softmax(
            tf.keras.layers.Dense(2, input_shape=(d_timestep * d_model + d_feature * d_model,), activation=None)(
                tf.concat([x_timestep, x_feature], axis=-1)
            ), axis=-1
        )
        gate_out = tf.concat([tf.multiply(x_timestep, gate[:, 0:1]), tf.multiply(x_feature, gate[:, 1:2])], axis=-1)
        print(f'gate_out.shape = {gate_out.shape}')
        out =  tf.keras.layers.Dense(class_num, input_shape=(d_timestep * d_model + d_feature * d_model,), activation=None)(gate_out)

        self.model = tf.keras.Model(inputs, out)


    def fit(self, X_train, Y_train, X_test, Y_test, epochs=200, save_model=True):
        batch_size = 32
        input_shape = X_train.shape[1:]
        self.build_model(
            d_model=256,
            d_feature=6,
            d_timestep=7,
            input_shape=input_shape,
            head_size=64,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            batch_size=batch_size,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        self.model.compile(
            loss='binary_focal_crossentropy',
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
            metrics=['categorical_accuracy']
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
            batch_size=batch_size,
            callbacks=callbacks,
        )

        self.evaluation = self.model.evaluate(X_test, Y_test, verbose=1)

        if save_model:
            self.model.save(self.file_name)




if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print( f"Found {len(gpus)} GPUs!" )
    for i in range( len( gpus ) ):
        try:
            tf.config.experimental.set_memory_growth(device=gpus[i], enable=True)
            tf.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*3 )
            print( f"\t{tf.config.experimental.get_device_details( device=gpus[i] )}" )
        except RuntimeError as e:
            print( '\n', e, '\n' )

    devices = tf.config.list_physical_devices()
    print( "Tensorflow sees the following devices:" )
    for dev in devices:
        print( f"\t{dev}" )

    dp = DataPreprocessing()
    dp.run(verbose=True)

    gtn = GTN()
    print(dp.X_train_under.shape)
    print(dp.Y_train_under.shape)
    print(dp.X_test.shape)
    print(dp.Y_test.shape)
    gtn.fit(X_train=dp.X_train_under, Y_train=dp.Y_train_under, X_test=dp.X_test, Y_test=dp.Y_test,
            epochs=200, save_model=False)
    gtn.plot_acc_loss()
    gtn.compute_confusion_matrix(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest, plot=True)
    gtn.make_probabilities_plots(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest)