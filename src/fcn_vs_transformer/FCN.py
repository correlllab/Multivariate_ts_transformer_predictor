import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, SimpleRNN, LSTM, GRU, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

import os, sys, pickle
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import seaborn as sns
import pandas as pd

from utils import CounterDict
from helper_functions import scan_output_for_decision, graph_episode_output


class FCN:
    def __init__(self, rolling_window_width) -> None:
        self.model = None
        self.history = None
        self.file_name = './models/FCN.keras'
        self.imgs_path = './imgs/fcn/'
        self.rollWinWidth = rolling_window_width


    def build(self):
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
        self.model.summary()

        self.model.compile(
            # optimizer=Adam(
            #     beta_1 = 0.9, #0.7,#0.8, #0.9, # - "β1 = 0.9"
            #     beta_2 = 0.999, #0.799,#0.85, #0.999, # "β2 = 0.999"
            # ),
            optimizer = SGD( 
                learning_rate = 0.0010, # 0.0015 #0.002 # 0.004
                momentum      = 0.125 #0.25 # 0.50 # 0.75
            ),
            loss      = 'binary_focal_crossentropy', #'MSE' 
            metrics=['categorical_accuracy']
        )


    def fit(self, X_train, Y_train, X_test, Y_test, trainWindows, epochs=200, save_model=True):
        batchSize = 256 #128 #256 #512 (out of mem) #256 #128

        callbacks = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.2
            )
        ]

        with tensorflow.device('/GPU:0'):
            self.history = self.model.fit( 
                # X_train, Y_train.reshape((-1,2)), 
                X_train, Y_train, 
                # validation_data  = (X_test, Y_test.reshape((-1,2)) ),
                validation_data  = (X_test, Y_test),
                batch_size       = batchSize, 
                epochs           = epochs, #250, #50, #250, # 2022-09-12: Trained for 250 total
                verbose          = True, 
                validation_split = 0.1,
                # steps_per_epoch  = int(trainWindows/batchSize), # https://stackoverflow.com/a/49924566
                steps_per_epoch = len(X_train) // batchSize,
                validation_steps = len(X_test) // batchSize,
                callbacks        = callbacks
            )
        
        if save_model:
            self.model.save(self.file_name)

            with open('./histories/FCN_history', 'wb') as file_pi:
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
            with tensorflow.device('/GPU:0'):
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
            print(f'Building FCN confusion matrix: {e}')
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
            with tensorflow.device('/GPU:0'):
                res = self.model.predict( X_winTest[epNo] )
                print( Y_winTest[epNo][0], '\n' )
                out_decision = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = 0.90 )
                print(out_decision)
                graph_episode_output( res=res, index=epNo, ground_truth=Y_winTest[epNo][0], out_decision=out_decision, net='fcn', ts_ms = 20, save_fig=True )
                print()
