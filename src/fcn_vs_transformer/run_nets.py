import os, sys
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np

import tensorflow as tf

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

from data_preprocessing import DataPreprocessing
from FCN import FCN
from VanillaTransformer import Transformer
from Transformer.CustomSchedule import CustomSchedule


load_data_from_file = False
save_data = False

dp = DataPreprocessing(sampling='none')
if load_data_from_file:
    # with open('X_pos_encoded_data.npy', 'r') as f:
    #     X_pos_encoded_data = np.load(f, allow_pickle=True)

    # with open('train_sampled_data.npy', 'r') as f:
    #     train_sampled_data = np.load(f, allow_pickle=True)

    # with open('test_data.npy', 'r') as f:
    #     test_data = np.load(f, allow_pickle=True)

    with open('preprocessing_data/X_data.npy', 'rb') as f:
        X_data = np.load(f)

    with open('preprocessing_data/Y_data.npy', 'rb') as f:
        Y_data = np.load(f)

    with open('preprocessing_data/win_data.npy', 'rb') as f:
        win_data = np.load(f, allow_pickle=True)

    x_train_sampled = X_data[2]
    y_train_sampled = Y_data[0]
    x_test = X_data[3]
    y_test = Y_data[1]
    x_train_enc = X_data[0]
    x_test_enc = X_data[1]
    x_win_test = win_data[0]
    y_win_test = win_data[1]
else:
    dp.run(save_data=save_data, verbose=True)

fcn_net = FCN(rolling_window_width=dp.rollWinWidth)
fcn_net.build()
fcn_net.fit(X_train=dp.X_train_sampled, Y_train=dp.Y_train_sampled, X_test=dp.X_test, Y_test=dp.Y_test,
            trainWindows=dp.trainWindows, epochs=200, save_model=True)
fcn_net.plot_acc_loss()
fcn_net.compute_confusion_matrix(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest, plot=True)
fcn_net.make_probabilities_plots(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest)

transformer_net = Transformer()
transformer_net.fit(X_train=dp.X_train_enc, Y_train=dp.Y_train_sampled, X_test=dp.X_test_enc, Y_test=dp.Y_test,
                    trainWindows=dp.trainWindows, epochs=200, save_model=True)
transformer_net.plot_acc_loss()
transformer_net.compute_confusion_matrix(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest, plot=True)
transformer_net.make_probabilities_plots(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest)



