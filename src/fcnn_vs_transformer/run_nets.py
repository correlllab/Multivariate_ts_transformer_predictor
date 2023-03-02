import os, sys
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
print( f"Found {len(gpus)} GPUs!" )
for i in range( len( gpus ) ):
    try:
        tensorflow.config.experimental.set_memory_growth(device=gpus[i], enable=True)
        tensorflow.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*3 )
        print( f"\t{tensorflow.config.experimental.get_device_details( device=gpus[i] )}" )
    except RuntimeError as e:
        print( '\n', e, '\n' )

devices = tensorflow.config.list_physical_devices()
print( "Tensorflow sees the following devices:" )
for dev in devices:
    print( f"\t{dev}" )

from data_preprocessing import DataPreprocessing
from FCN import FCN
from Transformer import Transformer



dp = DataPreprocessing()
dp.run(verbose=True)

print('====> TESTING FOR CLASS IMBALANCE:')
print(f'Passes = {sum(dp.Y_train[:,0])}; Fails = {sum(dp.Y_train[:,1])}')

fcn_net = FCN(rolling_window_width=dp.rollWinWidth)
fcn_net.build()
fcn_net.fit(X_train=dp.X_train, Y_train=dp.Y_train, X_test=dp.X_test, Y_test=dp.Y_test,
            trainWindows=dp.trainWindows, epochs=20, save_model=True)
fcn_net.plot_acc_loss()
fcn_net.compute_confusion_matrix(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest, plot=True)
fcn_net.make_probabilities_plots(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest)

transformer_net = Transformer()
transformer_net.fit(X_train=dp.X_train, Y_train=dp.Y_train, X_test=dp.X_test, Y_test=dp.Y_test,
                    trainWindows=dp.trainWindows, epochs=3, save_model=True)
transformer_net.plot_acc_loss()
transformer_net.compute_confusion_matrix(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest, plot=True)
transformer_net.make_probabilities_plots(X_winTest=dp.X_winTest, Y_winTest=dp.Y_winTest)


