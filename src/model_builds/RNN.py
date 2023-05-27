import os, sys, pickle
sys.path.append(os.path.realpath('../'))
print(sys.version)
print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing


class BaseRNN:
    def __init__(self, name: str = 'RNN'):
        self.model_name = name
        self.model = None
        self.history = None
        self.evaluation = None
        self.file_name = f'../saved_models/{self.model_name}.keras'
        self.imgs_path = f'../saved_data/imgs/{self.model_name}/'
        self.histories_path = f'../saved_data/histories/{self.model_name}_history'

    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=1024, epochs=200, save_model=True, verbose=False):
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        self.build_model(
            input_shape=X_train.shape[1:],
            dim=128,
            dropout=0.2,
            dense_dim=2
        )

        learning_rate = 1e-4
        opt = tf.keras.optimizers.legacy.Adam(learning_rate)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        metric = tf.keras.metrics.CategoricalAccuracy()

        self.model.compile(
            loss=loss_object,
            optimizer=opt,
            metrics=[metric]
        )
        if verbose:
            self.model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.1
            )
        ]

        self.history = self.model.fit(
            x=X_train,
            y=Y_train,
            # validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(X_test, Y_test),
            steps_per_epoch=len(X_train) // batch_size,
            validation_steps=len(X_test) // batch_size
        )

        if save_model:
            self.model.save(self.file_name)

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)


class RNN(BaseRNN):
    def __init__(self, name: str = 'RNN'):
        super().__init__(name)


    def build_model(self, input_shape, dim=128, dropout=0.2, dense_dim=2):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.SimpleRNN(units=dim, dropout=dropout))
        model.add(tf.keras.layers.Dense(dense_dim, activation='softmax'))

        self.model = model


class GRU(BaseRNN):
    def __init__(self, name: str = 'GRU'):
        super().__init__(name)


    def build_model(self, input_shape, dim=128, dropout=0.2, dense_dim=2):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.GRU(units=dim, dropout=dropout))
        model.add(tf.keras.layers.Dense(dense_dim, activation='softmax'))

        self.model = model


class LSTM(BaseRNN):
    def __init__(self, name: str = 'LSTM'):
        super().__init__(name)


    def build_model(self, input_shape, dim=128, dropout=0.2, dense_dim=2):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=input_shape))
        model.add(tf.keras.layers.LSTM(units=dim, dropout=dropout))
        model.add(tf.keras.layers.Dense(dense_dim, activation='softmax'))

        self.model = model


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
        print()

    dp = DataPreprocessing(sampling='none', data='reactive')
    dp.run(save_data=False, verbose=True)

    X_train = dp.X_train_sampled
    Y_train = dp.Y_train_sampled
    X_test = dp.X_test
    Y_test = dp.Y_test
    batch_size = 512
    epochs = 200

    rnn = RNN()
    rnn.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        batch_size=batch_size,
        epochs=epochs,
        verbose=True
    )

    # with open(f'./RNN_history', 'wb') as file_pi:
    #     pickle.dump(rnn.history.history, file_pi)

    # rnn.model.save('./models/RNN.keras')