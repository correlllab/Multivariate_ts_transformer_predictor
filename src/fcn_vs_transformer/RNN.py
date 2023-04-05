import os, sys, pickle
print(sys.version)
print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
from random import choice
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_preprocessing import DataPreprocessing
from utils import CounterDict
from helper_functions import scan_output_for_decision, graph_episode_output


class RNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.evaluation = None
        self.file_name = './models/RNN.keras'
        self.imgs_path = './imgs/rnn/'


    def build_model(self, input_shape, lstm_dim=128, dense_dim=2):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.LSTM(lstm_dim))
        model.add(tf.keras.layers.Dense(dense_dim, activation='softmax'))

        self.model = model


    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=64, epochs=200, save_model=True):
        self.build_model(
            input_shape=X_train.shape[1:],
            lstm_dim=128,
            dense_dim=2
        )

        self.model.compile(
            loss="binary_focal_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["categorical_accuracy"]
        )
        self.model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.2
            )
        ]

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

        if save_model:
            self.model.save(self.file_name)

            with open('./histories/RNN_history', 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)


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

        try:
            confMatx = {
                # Actual Positives
                'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
                'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
                # Actual Negatives
                'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
                'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
                'NC' : (perf['NC'] if ('NC' in perf) else 0) / len( X_winTest ),
            }
        except ZeroDivisionError as e:
            print(f'Building VanillaTransformer confusion matrix: {e}')
            confMatx = None
            plot = False

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
        print

    dp = DataPreprocessing(sampling='none')
    dp.run(save_data=False, verbose=True)

    X_train = dp.X_train_sampled
    Y_train = dp.Y_train_sampled
    X_test = dp.X_test
    Y_test = dp.Y_test
    batch_size = 64
    epochs = 200

    rnn = RNN()
    rnn.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        batch_size=batch_size,
        epochs=epochs
    )

    with open('./RNN_history', 'wb') as file_pi:
        pickle.dump(rnn.history.history, file_pi)

    rnn.model.save('./models/RNN.keras')