import sys, os, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf
from random import choice

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer

from utilities.makespan_utils import *


MODELS_TO_RUN = [
    'FCN',
    # 'RNN',
    # 'GRU',
    # 'LSTM',
    'VanillaTransformer',
    # 'OOP_Transformer',
    # 'OOP_Transformer_small'
    ]
# MODE = 'create_data'
MODE = 'load_data'


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
        print

    if MODE == 'create_data':
        print('FETCHING DATA...')
        dp = DataPreprocessing(sampling='under', data='reactive')
        dp.shuffle = False
        dp.run(save_data=False, verbose=True)

        X_train_sampled = dp.X_train_sampled
        trunc_data = dp.truncData

        if not os.path.exists('../../data/makespan_data/'):
            os.makedirs('../../data/makespan_data/')

        print('Creating X_train_sampled...', end='')
        with open('../../data/makespan_data/X_train_sampled.npy', 'wb') as f:
            np.save(f, dp.X_train_sampled, allow_pickle=True)
        print('DONE')

        print('Creating trunc_data...', end='')
        with open('../../data/makespan_data/trunc_data.npy', 'wb') as f:
            np.save(f, np.asarray(dp.truncData, dtype=object), allow_pickle=True)
        print('DONE')
    elif MODE == 'load_data':
        print('Loading data from files...', end='')
        with open('../../data/makespan_data/X_train_sampled.npy', 'rb') as f:
            X_train_sampled = np.load(f, allow_pickle=True)

        with open('../../data/makespan_data/trunc_data.npy', 'rb') as f:
            trunc_data = np.load(f, allow_pickle=True)
        print('DONE')

    makespan_models = {}

    # If True it will run pipeline: load models, predict (if True) and generate metrics, if False it will generate metrics from saved files
    compute = False
    # If True, it will save dicts upon metric generation
    save_dicts = True
    # If True it will save generated plots
    save_plots = True
    with open('../saved_data/makespan/makespan_results.txt', 'r') as f:
        res = json.loads(f.read())
    if compute:
        print('LOADING MODELS...')
        if 'FCN' in MODELS_TO_RUN:
            load_keras_model(model_name='FCN', makespan_models=makespan_models)

        if 'RNN' in MODELS_TO_RUN:
            load_keras_model(model_name='RNN', makespan_models=makespan_models)

        if 'VanillaTransformer' in MODELS_TO_RUN:
            load_keras_model(model_name='VanillaTransformer', makespan_models=makespan_models)

        if 'OOP_Transformer' in MODELS_TO_RUN:
            oop_transformer = OOPTransformer()
            num_layers = 4
            d_model = 6
            ff_dim = 256
            num_heads = 8
            head_size = 256
            dropout_rate = 0.2
            mlp_dropout = 0.4
            mlp_units = [128, 256, 64]

            oop_transformer.build(
                X_sample=X_train_sampled[:64],
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
            oop_transformer.compile()
            load_keras_weights(model_build=oop_transformer, model_name='OOP_Transformer', makespan_models=makespan_models)

        if 'OOP_Transformer_small' in MODELS_TO_RUN:
            oop_transformer_small = OOPTransformer()
            num_layers = 4
            d_model = 6
            ff_dim = 256
            num_heads = 4
            head_size = 128
            dropout_rate = 0.2
            mlp_dropout = 0.4
            mlp_units = [128]

            oop_transformer_small.build(
                X_sample=X_train_sampled[:64],
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
            oop_transformer_small.compile()
            load_keras_weights(model_build=oop_transformer_small, model_name='OOP_Transformer_small', makespan_models=makespan_models)

        for model_name, model in makespan_models.items():
            if model_name not in res.keys():
                res[model_name] = {'metrics': {}, 'conf_mat': {}, 'times': {}, 'makespan_sim_hist': [], 'makespan_sim_avg': -1, 'makespan_sim_std': -1}
            print(f'====> For model {model_name}:')
            avg_mks, mks = run_simulation(
                model=model,
                episodes=trunc_data,
                n_simulations=100,
                verbose=True
            )
            res[model_name]['makespan_sim_hist'] = mks
            res[model_name]['makespan_sim_avg'] = avg_mks
            res[model_name]['makespan_sim_std'] = np.std(res[model_name]['makespan_sim_hist'])

    print(f'res = {res}\n')

    if save_dicts:
        with open('../saved_data/makespan/makespan_results.txt', 'w') as f:
            f.write(json.dumps(res))

    plot_simulation_makespans(res=res, models=MODELS_TO_RUN, save_plots=save_plots)