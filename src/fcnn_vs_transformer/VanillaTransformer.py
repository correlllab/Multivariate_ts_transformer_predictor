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
        self.evaluation = None
        self.file_name = './models/Transformer.keras'
        self.imgs_path = './imgs/transformer/'

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
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
        print(f'X_train.shape = {X_train.shape}')
        print(f'X_train.shape[1:] = {X_train.shape[1:]}')
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


    def plot_acc_loss(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        accuracy = self.history.history['categorical_accuracy']
        val_accuracy = self.history.history['val_categorical_accuracy']

        fig, axes = plt.subplots(2, 2, figsize=(20, 8))
        axes[0, 0].plot(accuracy, label='Training accuracy')
        axes[0, 0].title.set_text('Training accuracy over epochs')
        axes[0, 1].plot(np.array(loss), label='Training loss', color='orange')
        axes[0, 1].title.set_text('Training loss over epochs')



        axes[1, 0].plot(val_accuracy, label='Validation accuracy')
        axes[1, 0].title.set_text('Valiadation accuracy over epochs')
        axes[1, 1].plot(np.array(val_loss), label='Validation loss', color='orange')
        axes[1, 1].title.set_text('Valiadation loss over epochs')


        plt.savefig(self.imgs_path + 'acc_loss_plots.png')
        plt.clf()



    def compute_confusion_matrix(self, X_winTest, Y_winTest, plot=False):
        perf = CounterDict()

        for epNo in range( len( X_winTest ) ):    
            print( '>', end=' ' )
            with tf.device('/GPU:0'):
                res = self.model.predict( X_winTest[epNo] )
                ans, aDx = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = 0.90 )
                perf.count( ans )
                
        print( '\n', self.file_name, '\n', perf )

        confMatx = {
            # Actual Positives
            'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            # Actual Negatives
            'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'NC' : (perf['NC'] if ('NC' in perf) else 0) / len( X_winTest ),
        }

        print( confMatx )

        if plot:
            arr = [
                [confMatx['TP'], confMatx['FP']],
                [confMatx['FN'], confMatx['TP']]
                ]
            conf_mat = sns.heatmap(arr, annot=True).get_figure()
            conf_mat.savefig(self.imgs_path + 'confusion_matrix.png')


    def make_probabilities_plots(self, X_winTest, Y_winTest):
        for epNo in range( len( X_winTest ) ):
            with tf.device('/GPU:0'):
                res = self.model.predict( X_winTest[epNo] )
                print( Y_winTest[epNo][0], '\n' )
                out_decision = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = 0.90 )
                print(out_decision)
                graph_episode_output( res=res, index=epNo, ground_truth=Y_winTest[epNo][0], out_decision=out_decision, net='transformer', ts_ms = 20, save_fig=True )
                print()