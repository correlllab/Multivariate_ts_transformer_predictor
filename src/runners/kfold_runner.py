import sys, os, glob
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.VanillaTransformer import VanillaTransformer



def plot_histories(histories: dict, num_folds: int, save=False):
    fig, axes = plt.subplots(2*num_folds, len(histories.keys()), figsize=(20, 8))
    model_ind = 0
    for model, hist in histories.items():
        fold = 0
        for h in hist:
            axes[fold, model_ind].plot(h.history['categorical_accuracy'], label=f'{model.upper()} Training accuracy')
            axes[fold, model_ind].plot(h.history['val_categorical_accuracy'], label=f'{model.upper()} Validation accuracy')
            axes[fold, model_ind].title.set_text(f'{model.upper()} Accuracy over epochs in fold {fold}')

            axes[fold+1, model_ind].plot(h.history['loss'], label=f'{model.upper()} Training loss')
            axes[fold+1, model_ind].plot(h.history['val_loss'], label=f'{model.upper()} Validation loss')
            axes[fold+1, model_ind].title.set_text(f'{model.upper()} Loss over epochs in fold {fold}')

            fold += 2
        model_ind += 1

    if save:
        plt.savefig('imgs/kfold/metrics_over_epochs.png')
    else:
        plt.plot()


if __name__ == "__main__":
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

    dp = DataPreprocessing(sampling='under', data='reactive')
    print(dp.datadir)
    dp.run(verbose=True)

    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    print('ALL OK')

    # Merge inputs and targets
    # inputs = np.concatenate((dp.X_train_sampled, dp.X_test), axis=0)
    # targets = np.concatenate((dp.Y_train_sampled, dp.Y_test), axis=0)
    inputs = dp.X_train_sampled
    targets = dp.Y_train_sampled


    # K-fold Cross Validation model evaluation
    fold_no = 1
    histories = {'fcn': [], 'transformer': []}
    losses = {'fcn': [], 'transformer': []}
    accs = {'fcn': [], 'transformer': []}
    for train, test in kfold.split(inputs, targets):
        print(len(train), len(test))
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        fcn_net = FCN(rolling_window_width=dp.rollWinWidth)
        fcn_net.build()
        fcn_history = fcn_net.fit(X_train=inputs[train], Y_train=targets[train],
                              X_test=inputs[test], Y_test=targets[test],
                              trainWindows=dp.trainWindows, epochs=20, save_model=False)
        fcn_evaluation = fcn_net.model.evaluate(dp.X_test,dp.Y_test, verbose=0)
        print(f'Score for fold {fold_no}: {fcn_net.model.metrics_names[0]} of {fcn_evaluation[0]}; {fcn_net.model.metrics_names[1]} of {fcn_evaluation[1]*100}%')
        histories['fcn'].append(fcn_history)
        accs['fcn'].append(fcn_evaluation[1] * 100)
        losses['fcn'].append(fcn_evaluation[0])

        # vanilla_transformer = Transformer()
        # transformer_history = vanilla_transformer.fit(X_train=inputs[train], Y_train=targets[train],
        #                                               X_test=inputs[test], Y_test=targets[test],
        #                                               trainWindows=dp.trainWindows, epochs=5, save_model=False)
        # transformer_evaluation = vanilla_transformer.model.evaluate(dp.X_test, dp.Y_test, verbose=0)
        # print(f'Score for fold {fold_no}: {vanilla_transformer.model.metrics_names[0]} of {vanilla_transformer.evaluation[0]}; {vanilla_transformer.model.metrics_names[1]} of {vanilla_transformer.evaluation[1]*100}%')
        # histories['transformer'].append(transformer_history)
        # accs['transformer'].append(transformer_evaluation[1] * 100)
        # losses['transformer'].append(transformer_evaluation[0])

        fold_no += 1

    plot_histories(histories=histories, num_folds=num_folds)

    fig, axes = plt.subplots(2, 2, figsize=(20, 8))

    axes[0, 0].plot(accs['fcn'], label='FCN Eval Accuracy per fold')
    axes[0, 0].title.set_text('FCN Eval Accuracy per fold')
    axes[0, 1].plot(np.array(losses['fcn']), label='FCN Eval Loss per fold', color='orange')
    axes[0, 1].title.set_text('FCN Eval Loss per fold')

    axes[1, 0].plot(accs['transformer'], label='Transformer Eval Accuracy per fold')
    axes[1, 0].title.set_text('Transformer Eval Accuracy per fold')
    axes[1, 1].plot(np.array(losses['transformer']), label='Transformer Eval Loss per fold', color='orange')
    axes[1, 1].title.set_text('Transformer Eval Loss per fold')

    plt.savefig('imgs/kfold/averages.png')
