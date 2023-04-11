import os, sys, pickle
print(sys.version)
print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
from random import choice
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data_management.data_preprocessing import DataPreprocessing
from utils.utils import CounterDict
from utils.helper_functions import scan_output_for_decision, graph_episode_output


class RNN:
    def __init__(self):
        self.model_name = 'RNN'
        self.model = None
        self.history = None
        self.evaluation = None
        self.file_name = f'../saved_models/{self.model_name}.keras'
        self.imgs_path = f'../saved_data/imgs/{self.model_name}/'
        self.histories_path = f'../saved_data/histories/{self.model_name}_history'


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

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)



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

    # with open(f'./RNN_history', 'wb') as file_pi:
    #     pickle.dump(rnn.history.history, file_pi)

    # rnn.model.save('./models/RNN.keras')