import sys, os, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from utilities.utils import set_size
from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN, GRU, LSTM
from model_builds.OOPTransformer import OOPTransformer



def plot_histories(histories: dict, num_folds: int, save:bool = True):
    # Setup
    plt.style.use('seaborn')
    # From Latex \textwidth
    fig_width = 345
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 14,
        "font.size": 14,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)

    fig, axes = plt.subplots(2*num_folds, len(histories.keys()), figsize=set_size(fig_width, subplots=(2*num_folds, len(histories.keys()))))
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


def find_min_max_len(values: list, num_folds: int):
    base_min = 200
    base_max = 0
    for i in range(num_folds):
        if len(values[i]) <= base_min:
            base_min = len(values[i])

        if len(values[i]) >= base_max:
            base_max = len(values[i])

    return base_min, base_max


def plot_histories_average(histories: dict, num_folds: int, save: bool = True):
    for model, hist in histories.items():
        # Setup
        plt.style.use('seaborn')
        # From Latex \textwidth
        fig_width = 345
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 14,
            "font.size": 14,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        }
        plt.rcParams.update(tex_fonts)

        fig, axes = plt.subplots(2, 2, figsize=set_size(fig_width, subplots=(2, 2)))
        fig.tight_layout(pad=3.0)

        accs = [item['categorical_accuracy'] for item in histories[model]]
        val_accs = [item['val_categorical_accuracy'] for item in histories[model]]
        losses = [item['loss'] for item in histories[model]]
        val_losses = [item['val_loss'] for item in histories[model]]

        acc_min_len, acc_max_len = find_min_max_len(values=accs, num_folds=num_folds)
        val_acc_min_len, val_acc_max_len = find_min_max_len(values=val_accs, num_folds=num_folds)
        loss_min_len, loss_max_len = find_min_max_len(values=losses, num_folds=num_folds)
        val_loss_min_len, val_loss_max_len = find_min_max_len(values=val_losses, num_folds=num_folds)

        acc_mean = np.mean([el[:acc_min_len] for el in accs], axis=0)
        val_acc_mean = np.mean([el[:val_acc_min_len] for el in val_accs], axis=0)
        loss_mean = np.mean([el[:loss_min_len] for el in losses], axis=0)
        val_loss_mean = np.mean([el[:val_loss_min_len] for el in val_losses], axis=0)

        axes[0, 0].plot(acc_mean)
        for l in [item['categorical_accuracy'] for item in histories[model]]:
            axes[0, 0].plot(l, color='grey', alpha=0.5)
        axes[0, 0].title.set_text(f'{model} mean training accuracy')
        axes[0, 0].legend(['Mean training accuracy', 'Training accuracy for each fold'])
        for fold_num, fold in enumerate([item['categorical_accuracy'] for item in histories[model]]):
            axes[0, 0].text(len(fold) - 1, fold[-1], f'  Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[0, 0].set_xlim((-acc_max_len * 0.1, acc_max_len + acc_max_len * 0.1))

        axes[0, 1].plot(loss_mean, color='orange')
        for l in [item['loss'] for item in histories[model]]:
            axes[0, 1].plot(l, color='grey', alpha=0.5)
        axes[0, 1].title.set_text(f'{model} mean training loss')
        axes[0, 1].legend(['Mean training loss', 'Training loss for each fold'])
        for fold_num, fold in enumerate([item['loss'] for item in histories[model]]):
            axes[0, 1].text(len(fold) - 1, fold[-1], f'  Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[0, 1].set_xlim((-val_acc_max_len * 0.1, val_acc_max_len + val_acc_max_len * 0.1))

        axes[1, 0].plot(val_acc_mean)
        for l in [item['val_categorical_accuracy'] for item in histories[model]]:
            axes[1, 0].plot(l, color='grey', alpha=0.5)
        axes[1, 0].title.set_text(f'{model} mean validation accuracy')
        axes[1, 0].legend(['Mean validation accuracy', 'Validation accuracy for each fold'])
        for fold_num, fold in enumerate([item['val_categorical_accuracy'] for item in histories[model]]):
            axes[1, 0].text(len(fold) - 1, fold[-1], f'  Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[1, 0].set_xlim((-loss_max_len * 0.1, loss_max_len + loss_max_len * 0.1))

        axes[1, 1].plot(val_loss_mean, color='orange')
        for l in [item['val_loss'] for item in histories[model]]:
            axes[1, 1].plot(l, color='grey', alpha=0.5)
        axes[1, 1].title.set_text(f'{model} mean validation loss')
        axes[1, 1].legend(['Mean validation loss', 'Validation loss for each fold'])
        for fold_num, fold in enumerate([item['val_loss'] for item in histories[model]]):
            axes[1, 1].text(len(fold) - 1, fold[-1], f'  Fold {fold_num}', ha='left', va='center', size='small', color='grey')
        axes[1, 1].set_xlim((-val_loss_max_len * 0.1, val_loss_max_len + val_loss_max_len * 0.1))

        if save:
            if not os.path.exists('../saved_data/imgs/kfold_crossvalidation/'):
                os.makedirs('../saved_data/imgs/kfold_crossvalidation/')
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


def build_gru():
    # GRU
    return GRU()


def build_lstm():
    # LSTM
    return LSTM()


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
    elif name == 'GRU':
        return build_gru()
    elif name == 'LSTM':
        return build_lstm()
    elif name == 'OOP_Transformer':
        return build_oop_transformer(X_sample=X_sample, model_type='big')
    elif name == 'OOP_Transformer_small':
        return build_oop_transformer(X_sample=X_sample, model_type='small')

    return None


MODELS_TO_RUN = [
    'FCN',
    'RNN',
    'GRU',
    'LSTM',
    'OOP_Transformer',
    'OOP_Transformer_small'
    ]
COMPUTE = True
DATA_MODE = 'create'
# DATA_MODE = 'load'
SAVE_HISTORIES = True
SAVE_MODEL_SIZE = True

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

    num_folds = 5

    if SAVE_HISTORIES:
        if not os.path.exists('../saved_data/kfold_crossvalidation/'):
            os.makedirs('../saved_data/kfold_crossvalidation/')

    if COMPUTE:
        if DATA_MODE == 'create':
            dp = DataPreprocessing(sampling='under', data='reactive')
            print(dp.datadir)
            dp.run(verbose=True)

            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=num_folds, shuffle=True)

            print('ALL OK')

            # Merge inputs and targets
            # inputs = np.concatenate((dp.X_train_sampled, dp.X_test), axis=0)
            # targets = np.concatenate((dp.Y_train_sampled, dp.Y_test), axis=0)
            inputs = dp.X_train_sampled
            targets = dp.Y_train_sampled
            print(inputs.shape, targets.shape)
        elif DATA_MODE == 'load':
            pass

        if os.path.exists('../saved_data/kfold_crossvalidation/histories.json'):
            with open('../saved_data/kfold_crossvalidation/histories.json') as f:
                histories = json.load(f)
            print('\nLoaded histories from file\n')
            for model_name in MODELS_TO_RUN:
                histories[model_name] = []
        else:
            histories = {key: [] for key in MODELS_TO_RUN}
        histories['num_folds'] = 0

        model_n_params = {key: [] for key in MODELS_TO_RUN}
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

                if fold_no == 1 and SAVE_MODEL_SIZE:
                    model_n_params[model.model_name] = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.model.trainable_variables]))

                tf.keras.backend.clear_session()

            if fold_no == 1 and SAVE_MODEL_SIZE:
                with open('../saved_data/model_sizes_kfold.json', 'w') as f:
                    json.dump(model_n_params, f)

            fold_no += 1
            histories['num_folds'] += 1


            if SAVE_HISTORIES:
                with open('../saved_data/kfold_crossvalidation/histories.json', 'w') as f:
                    json.dump(histories, f)


        print()
        print(histories)
    else:
        with open('../saved_data/kfold_crossvalidation/histories.json', 'r') as f:
            histories = json.load(f)


    plot_histories_average(histories=histories, num_folds=histories['num_folds'], save=True)
    # plot_histories(histories=histories, num_folds=num_folds, save=True)
