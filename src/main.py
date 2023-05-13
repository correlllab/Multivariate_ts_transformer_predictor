import os, sys
sys.path.append(os.path.realpath('./'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import compute_confusion_matrix

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.dirname(os.path.dirname(__file__))

DATA = 'reactive'
DATA_DIR = f'../../data/data_manager/{DATA}'
MODELS_TO_RUN = [
    'FCN',
    'VanillaTransformer'
]


def load_keras_model(model_name: str, makespan_models: dict, verbose: bool = True):
    try:
        model = tf.keras.models.load_model(f'../saved_models/{model_name}.keras')
        makespan_models[model_name] = model
        if verbose:
            print(f'--> Loaded {model_name}')
    except OSError as e:
        print(f'{e}: model {model_name}.keras does not exist!\n')
    

def load_keras_weights(model_build: OOPTransformer, model_name: str, makespan_models: dict, verbose: bool = True):
    try:
        model_build.model.load_weights(f'../saved_models/{model_name}/').expect_partial()
        makespan_models[model_name] = model_build.model
        if verbose:
            print(f'--> Loaded {model_name}')
    except OSError as e:
        print(f'{e}: model weights {model_name} do not exist!')


if __name__ == '__main__':
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

    # Load training data
    print('\nLoading data from files...', end='')
    with open(f'{DATA_DIR}/{DATA}_X_train.npy', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{DATA}_Y_train.npy', 'rb') as f:
        Y_train = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{DATA}_X_test.npy', 'rb') as f:
        X_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{DATA}_Y_test.npy', 'rb') as f:
        Y_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{DATA}_X_winTest.npy', 'rb') as f:
        X_window_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{DATA}_Y_winTest.npy', 'rb') as f:
        Y_window_test = np.load(f, allow_pickle=True)
    roll_win_width = int(7.0 * 50)
    print('DONE\n')

    # Load models
    makespan_models = {}

    if 'FCN' in MODELS_TO_RUN:
        load_keras_model(model_name='FCN', makespan_models=makespan_models)

    if 'VanillaTransformer' in MODELS_TO_RUN:
        load_keras_model(model_name='VanillaTransformer', makespan_models=makespan_models)

    # Call function to compute confusion matrices
    fcn_model = makespan_models['FCN']
    fcn_conf_mat = compute_confusion_matrix(
        model=fcn_model,
        file_name=fcn_model.file_name,
        imgs_path=fcn_model.imgs_path,
        X_winTest=X_window_test,
        Y_winTest=Y_window_test,
        plot=True
    )

    transformer_model = makespan_models['VanillaTransformer']
    transformer_conf_mat = compute_confusion_matrix(
        model=transformer_model,
        file_name=transformer_model.file_name,
        imgs_path=transformer_model.imgs_path,
        X_winTest=X_window_test,
        Y_winTest=Y_window_test,
        plot=True
    )