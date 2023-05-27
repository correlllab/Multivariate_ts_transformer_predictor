import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers


class FCN:
    def __init__(self, rolling_window_width) -> None:
        self.model_name = 'FCN'
        self.model = None
        self.history = None
        self.file_name = f'../saved_models/{self.model_name}.keras'
        self.imgs_path = f'../saved_data/imgs/{self.model_name}/'
        self.histories_path = f'../saved_data/histories/{self.model_name}_history'
        self.rollWinWidth = rolling_window_width


    def build(self, verbose=False):
        mirrored_strategy = tensorflow.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            self.model = Sequential()

            # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
            self.model.add(Conv1D( # "Dilated Convolution Layer 1 (No. of filters = 8, Filter Size = 2, Dilation Rate = 2)"
                filters       = 16, #32 #16 #8, 
                kernel_size   =  4, #2, 
                dilation_rate =  4, #2,
                activation  = 'relu', # -------- "a sequence of layers of dilated convolutional layers with ReLU"
                input_shape = (self.rollWinWidth, 6,) #(n_steps, n_features)
            ))

            self.model.add(Conv1D( # "Dilated Convolution Layer 2 (No. of filters = 8, Filter Size = 2, Dilation Rate = 4)"
                filters       = 16, #32 #16 # 8, 
                kernel_size   =  4, #2, 
                dilation_rate =  8, #4,
                activation  = 'relu', # -------- "a sequence of layers of dilated convolutional layers with ReLU"
                input_shape = (self.rollWinWidth, 6,) #(n_steps, n_features)
            ))

            # https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
            self.model.add( MaxPooling1D( pool_size = 4 ) ) #"Max Pooling Layer"

            self.model.add( Flatten() )

            self.model.add( Dense(50) ) # 50 # 200 #100 # fully connected layer
            self.model.add( Activation('relu') )

            self.model.add( Dropout( rate = 0.5 ) ) # "We used drop out at a rate of 0.5"

            self.model.add( Dense( 2 ) ) # "The softmax layer computes the probability values for the predicted class."
            self.model.add( Activation( "softmax" ) )

            opt = tensorflow.keras.optimizers.SGD(learning_rate=0.0010, momentum=0.125)
            loss_object = tensorflow.keras.losses.CategoricalCrossentropy()
            metric = tensorflow.keras.metrics.CategoricalAccuracy()
        if verbose:
            self.model.summary()

        self.model.compile(
            # optimizer=Adam(
            #     beta_1 = 0.9, #0.7,#0.8, #0.9, # - "β1 = 0.9"
            #     beta_2 = 0.999, #0.799,#0.85, #0.999, # "β2 = 0.999"
            # ),
            optimizer = opt,
            loss      = loss_object, #'MSE' 
            metrics=[metric]
        )


    def fit(self, X_train, Y_train, X_test, Y_test, batch_size=2048, epochs=200, save_model=True):
        self.build(verbose=False)

        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.1
            )
        ]

        with tensorflow.device('/GPU:0'):
            self.history = self.model.fit( 
                # X_train, Y_train.reshape((-1,2)), 
                X_train, Y_train, 
                # validation_data  = (X_test, Y_test.reshape((-1,2)) ),
                validation_data  = (X_test, Y_test),
                batch_size       = batch_size, 
                epochs           = epochs, #250, #50, #250, # 2022-09-12: Trained for 250 total
                verbose          = True, 
                # validation_split = 0.2,
                # steps_per_epoch  = int(trainWindows/batch_size), # https://stackoverflow.com/a/49924566
                steps_per_epoch = len(X_train) // batch_size,
                validation_steps = len(X_test) // batch_size,
                callbacks        = callbacks
            )
        
        if save_model:
            self.model.save(self.file_name)

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)

