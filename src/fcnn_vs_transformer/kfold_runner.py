import os, sys
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

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



if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.run(verbose=True)

    print('====> TESTING FOR CLASS IMBALANCE:')
    print(f'Passes = {sum(dp.Y_train[:,0])}; Fails = {sum(dp.Y_train[:,1])}')

    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # Merge inputs and targets
    inputs = np.concatenate((dp.X_train, dp.X_test), axis=0)
    targets = np.concatenate((dp.Y_train, dp.Y_test), axis=0)

    print(inputs.shape, targets.shape)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    for train, test in kfold.split(inputs, targets):
        transformer_net = Transformer()
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        transformer_net.fit(X_train=inputs[train], Y_train=targets[train], X_test=inputs[test], Y_test=targets[test],
                            trainWindows=dp.trainWindows, epochs=2, save_model=False)
        print(f'Score for fold {fold_no}: {transformer_net.model.metrics_names[0]} of {transformer_net.evaluation[0]}; {transformer_net.model.metrics_names[1]} of {transformer_net.evaluation[1]*100}%')
        acc_per_fold.append(transformer_net.evaluation[1] * 100)
        loss_per_fold.append(transformer_net.evaluation[0])
        fold_no += 1

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0, 0].plot(acc_per_fold, label='Eval Accuracy per fold')
    axes[0, 0].title.set_text('Eval Accuracy per fold')
    axes[0, 1].plot(np.array(loss_per_fold), label='Eval Loss per fold', color='orange')
    axes[0, 1].title.set_text('Eval Loss per fold')
