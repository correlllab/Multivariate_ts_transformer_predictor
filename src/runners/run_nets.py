import sys, os, pickle, json, time
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN, GRU, LSTM
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import plot_acc_loss, plot_evaluation_on_test_window_data


def load_keras_model(model_name: str, verbose: bool = True):
    try:
        model = tf.keras.models.load_model(f'../saved_models/{model_name}.keras')
        if verbose:
            print(f'--> Loaded {model_name}')
        return model
    except OSError as e:
        print(f'{e}: model {model_name}.keras does not exist!\n')
    

def load_keras_weights(model_build: OOPTransformer, model_name: str, verbose: bool = True):
    try:
        model_build.model.load_weights(f'../saved_models/{model_name}/').expect_partial()
        if verbose:
            print(f'--> Loaded {model_name}')
    except OSError as e:
        print(f'{e}: model weights {model_name} do not exist!')


def build_fcn(roll_win_width:int):
    # FCN
    # fcn_net = FCN(rolling_window_width=roll_win_width)
    # fcn_net.build()
    return FCN(rolling_window_width=roll_win_width)


def build_rnn():
    # RNN
    return RNN()


def build_gru():
    # GRU
    return GRU()


def build_lstm():
    # LSTM
    return LSTM()


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

    # transformer_net.compile()

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
    elif name == 'VanillaTransformer':
        return build_vanilla_transformer()
    elif name == 'OOP_Transformer':
        return build_oop_transformer(X_sample=X_sample, model_type='big')
    elif name == 'OOP_Transformer_small':
        return build_oop_transformer(X_sample=X_sample, model_type='small')

    return None


DATA = ['reactive', 'training']
# DATA = ['training']
# DATA = ['reactive']
DATA_DIR = f'../../data/data_manager/{"_".join(DATA)}'
SAVE_DATA = True
LOAD_DATA_FROM_FILES = True
MODELS_TO_RUN = [
    'FCN',
    'GRU',
    'LSTM',
    'RNN',
    'VanillaTransformer',
    'OOP_Transformer',
    'OOP_Transformer_small'
    ]


class hist_obj:
    def __init__(self, h: dict) -> None:
        self.history = h


def run_model(model, X_train, Y_train, X_test, Y_test, X_window_test, Y_window_test, model_n_params, compute: bool = True):
    if compute:
        model_start_time = time.time()
        model.fit(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            epochs=200,
            save_model=True
        )
        model_training_time = (time.time() - model_start_time) / 60.0
        print(f'\n{model_name} training time = {model_training_time} minutes\n')
    else:
        model_training_time = 0
        model.history = hist_obj(np.load(model.histories_path, allow_pickle=True))
        if model.model_name in ('FCN', 'RNN', 'GRU', 'LSTM', 'VanillaTransformer'):
            model.model = load_keras_model(model_name=model.model_name, verbose=True)
        elif model.model_name in ('OOP_Transformer', 'OOP_Transformer_small'):
            load_keras_weights(model_build=model, model_name=model.model_name, verbose=True)
    try:
        model_n_params[model.model_name] = int(np.sum([np.prod(v.get_shape().as_list()) for v in model.model.trainable_variables]))
        with open('../saved_data/model_sizes.json', 'w') as f:
            json.dump(model_n_params, f)
    except AttributeError as e:
        print(f'For model {model_name}: {e}')
    plot_acc_loss(history=model.history, imgs_path=model.imgs_path)
    preds = []
    # preds, _ = plot_evaluation_on_test_window_data(
    #     model=model,
    #     model_name=model_name,
    #     X_data=X_window_test,
    #     Y_data=Y_window_test,
    #     confidence=0.9,
    # )

    return model_n_params, model_training_time, preds


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print( f"Found {len(gpus)} GPUs!" )
    for i in range( len( gpus ) ):
        try:
            tf.config.experimental.set_memory_growth(device=gpus[i], enable=True)
            tf.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*6 )
            print( f"\t{tf.config.experimental.get_device_details( device=gpus[i] )}" )
        except RuntimeError as e:
            print( '\n', e, '\n' )

    devices = tf.config.list_physical_devices()
    print( "Tensorflow sees the following devices:" )
    for dev in devices:
        print( f"\t{dev}" )

    dp = DataPreprocessing(sampling='none', data=DATA)
    if LOAD_DATA_FROM_FILES:
        print(f'\nLoading data from files (using {DATA})...', end='')
        with open(f'{DATA_DIR}/{"_".join(DATA)}_X_train.npy', 'rb') as f:
            X_train = np.load(f, allow_pickle=True)

        with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_train.npy', 'rb') as f:
            Y_train = np.load(f, allow_pickle=True)

        with open(f'{DATA_DIR}/{"_".join(DATA)}_X_test.npy', 'rb') as f:
            X_test = np.load(f, allow_pickle=True)

        with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_test.npy', 'rb') as f:
            Y_test = np.load(f, allow_pickle=True)

        with open(f'{DATA_DIR}/{"_".join(DATA)}_X_winTest.npy', 'rb') as f:
            X_winTest = np.load(f, allow_pickle=True)

        with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_winTest.npy', 'rb') as f:
            Y_winTest = np.load(f, allow_pickle=True)
        roll_win_width = int(7.0 * 50)
        print('DONE\n')
        print(f'Number of test episodes = {len(X_winTest)}')
    else:
        print('\nCreating data with DataPreprocessng class...', end='')
        dp.run(save_data=SAVE_DATA, verbose=True)
        X_train = dp.X_train
        Y_train = dp.Y_train
        X_test = dp.X_test
        Y_test = dp.Y_test
        X_winTest = dp.X_winTest
        Y_winTest = dp.Y_winTest
        roll_win_width = dp.rollWinWidth
        print('DONE\n')

    # From the previous we have 0.8 train split and 0.2 test split, now we need to separate
    # the train split into train-validation splits

    # Generate train-validation split with 0.8 train and 0.2 validation
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.8)

    if os.path.exists('../saved_data/model_sizes.json'):
        with open('../saved_data/model_sizes.json', 'r') as f:
            model_n_params = json.load(f)
    else:
        model_n_params = {}

    print(model_n_params)

    with open('../saved_data/training_times.txt', 'r+') as f:
        f.truncate(0)

    preds_dict = {}
    overall_start_time = time.time()
    for model_name in MODELS_TO_RUN:
        model = get_model(
            name=model_name,
            roll_win_width=roll_win_width,
            X_sample=X_train[:64]
        )
        print(f'--> Training {model_name}...')
        model_n_params, model_training_time, preds = run_model(
            model=model,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_val,
            Y_test=Y_val,
            X_window_test=X_winTest,
            Y_window_test=Y_winTest,
            model_n_params=model_n_params,
            compute=False
        )
        preds_dict[model_name] = preds

        with open('../saved_data/training_times.txt', 'a') as f:
            f.write(f'{model_name} training time = {model_training_time} minutes\n')

        with open('../saved_data/model_sizes.json', 'w') as f:
            json.dump(model_n_params, f)

        tf.keras.backend.clear_session()

    full_training_time = (time.time() - overall_start_time) / 60.0
    print(f'Whole training time = {full_training_time} minutes')
    with open('../saved_data/training_times.txt', 'a') as f:
        f.write(f'Full training time = {full_training_time} minutes\n')

    # TODO plot ROC AUC

