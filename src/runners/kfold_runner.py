import sys, os, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer



def plot_histories(histories: dict, num_folds: int, save:bool = True):
    fig, axes = plt.subplots(2*num_folds, len(histories.keys()), figsize=(25, 12))
    fig.tight_layout(pad=3.0)
    model_ind = 0
    for model, hist in histories.items():
        fold = 0
        fold_index = 1
        for h in hist:
            axes[fold, model_ind].plot(h['categorical_accuracy'], label=f'{model} Training accuracy')
            axes[fold, model_ind].plot(h['val_categorical_accuracy'], label=f'{model} Validation accuracy')
            axes[fold, model_ind].title.set_text(f'{model} Accuracy over epochs in fold {fold_index}')
            axes[fold, model_ind].legend()

            axes[fold+1, model_ind].plot(h['loss'], label=f'{model} Training loss')
            axes[fold+1, model_ind].plot(h['val_loss'], label=f'{model} Validation loss')
            axes[fold+1, model_ind].title.set_text(f'{model} Loss over epochs in fold {fold_index}')
            axes[fold+1, model_ind].legend()

            fold += 2
            fold_index += 1
        model_ind += 1

    if save:
        plt.savefig('../saved_data/imgs/kfold_crossvalidation/metrics_over_epochs.png')
        plt.clf()
    else:
        plt.show()

def plot_histories_average(histories: dict, save: bool = True):
    for model, hist in histories.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.tight_layout(pad=3.0)

        acc_mean = np.mean([item['categorical_accuracy'] for item in histories[model]], axis=0)
        val_acc_mean = np.mean([item['val_categorical_accuracy'] for item in histories[model]], axis=0)
        loss_mean = np.mean([item['loss'] for item in histories[model]], axis=0)
        val_loss_mean = np.mean([item['val_loss'] for item in histories[model]], axis=0)

        axes[0, 0].plot(acc_mean)
        for l in [item['categorical_accuracy'] for item in histories[model]]:
            axes[0, 0].plot(l, color='grey', alpha=0.5)
        axes[0, 0].title.set_text(f'{model} mean training accuracy')
        axes[0, 0].legend(['Mean training accuracy', 'Training accuracy for each fold'])
        for fold_num, fold in enumerate([item['categorical_accuracy'] for item in histories[model]]):
            axes[0, 0].text(len(fold) - 1, fold[-1], f'Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[0, 0].set_xlim((-0.1, len(fold) - 0.9))

        axes[0, 1].plot(loss_mean, color='orange')
        for l in [item['loss'] for item in histories[model]]:
            axes[0, 1].plot(l, color='grey', alpha=0.5)
        axes[0, 1].title.set_text(f'{model} mean training loss')
        axes[0, 1].legend(['Mean training loss', 'Training loss for each fold'])
        for fold_num, fold in enumerate([item['loss'] for item in histories[model]]):
            axes[0, 1].text(len(fold) - 1, fold[-1], f'Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[0, 1].set_xlim((-0.1, len(fold) - 0.9))

        axes[1, 0].plot(val_acc_mean)
        for l in [item['val_categorical_accuracy'] for item in histories[model]]:
            axes[1, 0].plot(l, color='grey', alpha=0.5)
        axes[1, 0].title.set_text(f'{model} mean validation accuracy')
        axes[1, 0].legend(['Mean validation accuracy', 'Validation accuracy for each fold'])
        for fold_num, fold in enumerate([item['val_categorical_accuracy'] for item in histories[model]]):
            axes[1, 0].text(len(fold) - 1, fold[-1], f'Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[1, 0].set_xlim((-0.1, len(fold) - 0.9))

        axes[1, 1].plot(val_loss_mean, color='orange')
        for l in [item['val_loss'] for item in histories[model]]:
            axes[1, 1].plot(l, color='grey', alpha=0.5)
        axes[1, 1].title.set_text(f'{model} mean validation loss')
        axes[1, 1].legend(['Mean validation loss', 'Validation loss for each fold'])
        for fold_num, fold in enumerate([item['val_loss'] for item in histories[model]]):
            axes[1, 1].text(len(fold) - 1, fold[-1], f'Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[1, 1].set_xlim((-0.1, len(fold) - 0.9))

        if save:
            plt.savefig(f'../saved_data/imgs/kfold_crossvalidation/{model}_mean_metrics.png')
            plt.clf()
            plt.close('all')
        else:
            plt.show()


def build_fcn(roll_win_width:int):
    # FCN
    fcn_net = FCN(rolling_window_width=roll_win_width)
    fcn_net.build()
    return fcn_net


def build_rnn():
    # RNN
    return RNN()


def build_vanilla_transformer():
    # VanillaTransformer
    return VanillaTransformer()


def build_oop_transformer(X_sample, model_type: str):
    name = 'OOP_Transformer'
    if model_type == 'small':
        name = name + '_' + model_type

    transformer_net = OOPTransformer(model_name=name)

    num_layers = 4
    d_model = 6
    ff_dim = 256
    num_heads = 8
    head_size = 256
    dropout_rate = 0.2
    mlp_dropout = 0.4
    mlp_units = [128, 256, 64]
    if model_type == 'small':
        num_layers = 4
        d_model = 6
        ff_dim = 256
        num_heads = 4
        head_size = 128
        dropout_rate = 0.2
        mlp_dropout = 0.4
        mlp_units = [128]

    transformer_net.build(
            X_sample=X_sample,
            num_layers=num_layers,
            d_model=d_model,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_size=head_size,
            dropout_rate=dropout_rate,
            mlp_dropout=mlp_dropout,
            mlp_units=mlp_units,
            verbose=False
        )

    transformer_net.compile()

    return transformer_net


def get_model(name: str, roll_win_width: int = 0, X_sample = None):
    if name == 'FCN':
        return build_fcn(roll_win_width=roll_win_width)
    elif name == 'RNN':
        return build_rnn()
    elif name == 'VanillaTransformer':
        return build_vanilla_transformer()
    elif name == 'OOP_Transformer':
        return build_oop_transformer(X_sample=X_sample, model_type='big')
    elif name == 'OOP_Transformer_small':
        return build_oop_transformer(X_sample=X_sample, model_type='small')

    return None


MODELS_TO_RUN = [
    'FCN',
    'RNN',
    'VanillaTransformer',
    'OOP_Transformer',
    'OOP_Transformer_small'
    ]
COMPUTE = True
SAVE_HISTORIES = True

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

    num_folds = 2

    if COMPUTE:
        dp = DataPreprocessing(sampling='under', data='reactive')
        print(dp.datadir)
        dp.run(verbose=True)

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, shuffle=True)

        print('ALL OK')

        # Merge inputs and targets
        inputs = np.concatenate((dp.X_train_sampled, dp.X_test), axis=0)
        targets = np.concatenate((dp.Y_train_sampled, dp.Y_test), axis=0)
        print(inputs.shape, targets.shape)

        histories = {key: [] for key in MODELS_TO_RUN}
        fold_no = 1
        for train, test in kfold.split(inputs, targets):
            print(f'\nFold {fold_no}/{num_folds}:')
            for model_name in MODELS_TO_RUN:
                model = get_model(name=model_name,
                                roll_win_width=dp.rollWinWidth,
                                X_sample=dp.X_train_sampled[:64])
                print(f'--> Training {model_name}...')
                model.fit(
                    X_train=inputs[train],
                    Y_train=targets[train],
                    X_test=inputs[test],
                    Y_test=targets[test],
                    epochs=200,
                    save_model=True
                )
                histories[model_name].append(model.history.history)

                tf.keras.backend.clear_session()

            fold_no += 1


        if not os.path.exists('../saved_data/imgs/kfold_crossvalidation/'):
            os.makedirs('../saved_data/imgs/kfold_crossvalidation/')

        print()
        print(histories)
        if SAVE_HISTORIES:
            if not os.path.exists('../saved_data/kfold_crossvalidation/'):
                os.makedirs('../saved_data/kfold_crossvalidation/') 
            with open('../saved_data/kfold_crossvalidation/histories.json', 'w') as f:
                json.dump(histories, f)
    else:
        with open('../saved_data/kfold_crossvalidation/histories.json', 'r') as f:
            histories = json.load(f)

    plot_histories_average(histories=histories, save=True)
    plot_histories(histories=histories, num_folds=num_folds, save=True)
