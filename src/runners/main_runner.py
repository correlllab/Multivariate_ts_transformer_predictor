import os, sys, json
sys.path.append(os.path.realpath('../'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf

from tabulate import tabulate

from run_makespan_simulation import run_reactive_simulation, run_makespan_simulation
from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN, GRU, LSTM
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import compute_confusion_matrix, plot_roc_window_data, plot_equation_simulation_makespan_barplots, make_probabilities_plots, plot_monte_carlo_simulation_barplots
from utilities.makespan_utils import get_makespan_for_model, get_mts_mtf, scan_output_for_decision, monitored_makespan, reactive_makespan, plot_simulation_makespans
from utilities.utils import CounterDict
from utilities.plot_classification_examples import plot_ft_classification_for_model

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.dirname(os.path.dirname(__file__))

DATA = ['reactive', 'training']
DATA_DIR = f'../../data/instance_data/{"_".join(DATA)}'
MODELS_TO_RUN = [
    'FCN',
    'RNN',
    'GRU',
    'LSTM',
    'OOP_Transformer_small',
    'OOP_Transformer'
]


def load_keras_model(model_name: str, makespan_models: dict, verbose: bool = True):
    try:
        model = tf.keras.models.load_model(f'../saved_models/{model_name}.keras')
        makespan_models[model_name.split('-')[0]] = model
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
        # name = name + '_' + model_type
        name = 'Transformer'

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
    elif name == 'OOP_Transformer':
        return build_oop_transformer(X_sample=X_sample, model_type='big')
    elif name == 'OOP_Transformer_small':
        return build_oop_transformer(X_sample=X_sample, model_type='small')


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

    # Load data
    print('\nLoading data from files...', end='')
    with open(f'{DATA_DIR}/{"_".join(DATA)}_X_train.npy', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_train.npy', 'rb') as f:
    #     Y_train = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{"_".join(DATA)}_X_test.npy', 'rb') as f:
    #     X_test = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_test.npy', 'rb') as f:
    #     Y_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_X_winTest.npy', 'rb') as f:
        X_window_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_Y_winTest.npy', 'rb') as f:
        Y_window_test = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_data.npy', 'rb') as f:
        data = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_trunc_data.npy', 'rb') as f:
        trunc_data = np.load(f, allow_pickle=True)
    roll_win_width = int(7.0 * 50)
    print('DONE\n')

    # Create table to be printed
    headers = ['Measure', 'Reactive', 'FCN', 'RNN', 'GRU', 'LSTM', 'Transformer', 'Transformer_big']
    data_table = [
        ['Makespan [s] (simulation)', None, None, None, None, None, None, None, None],
        ['Predicted [s] (equation)', None, None, None, None, None, None, None, None],
        ['MTS', None, None, None, None, None, None, None, None],
        ['MTF', None, None, None, None, None, None, None, None],
        ['MTP', None, None, None, None, None, None, None, None],
        ['MTN', None, None, None, None, None, None, None, None],
        ['P_TP', None, None, None, None, None, None, None, None],
        ['P_FN', None, None, None, None, None, None, None, None],
        ['P_TN', None, None, None, None, None, None, None, None],
        ['P_FP', None, None, None, None, None, None, None, None],
        ['P_NCS', None, None, None, None, None, None, None, None],
        ['P_NCF', None, None, None, None, None, None, None, None]
    ]

    # Load models
    makespan_models = {}

    if 'FCN' in MODELS_TO_RUN:
        load_keras_model(model_name='FCN', makespan_models=makespan_models)

    if 'RNN' in MODELS_TO_RUN:
        load_keras_model(model_name='RNN', makespan_models=makespan_models)

    if 'GRU' in MODELS_TO_RUN:
        load_keras_model(model_name='GRU', makespan_models=makespan_models)

    if 'LSTM' in MODELS_TO_RUN:
        load_keras_model(model_name='LSTM', makespan_models=makespan_models)

    if 'OOP_Transformer_small' in MODELS_TO_RUN:
        transformer = get_model(name='OOP_Transformer_small', roll_win_width=roll_win_width, X_sample=X_train[:64])
        load_keras_weights(model_build=transformer, model_name='OOP_Transformer_small', makespan_models=makespan_models, verbose=True)

    if 'OOP_Transformer' in MODELS_TO_RUN:
        transformer = get_model(name='OOP_Transformer', roll_win_width=roll_win_width, X_sample=X_train[:64])
        load_keras_weights(model_build=transformer, model_name='OOP_Transformer', makespan_models=makespan_models, verbose=True)

    # Call function to compute confusion matrices
    compute_conf_mats = False
    make_prob_plots = False
    for model_name in MODELS_TO_RUN:
        model = get_model(name=model_name, roll_win_width=roll_win_width, X_sample=X_train[:64])
        model.model = makespan_models[model_name]
        print(f'--> For model {model.model_name}')
        if compute_conf_mats:
            _ = compute_confusion_matrix(
                model=model.model,
                model_name=model.model_name,
                file_name=model.file_name,
                imgs_path=model.imgs_path,
                X_winTest=X_window_test,
                Y_winTest=Y_window_test,
                confidence=0.9,
                simulation=False,
                plot=True
            )
        if make_prob_plots:
            make_probabilities_plots(
                model=model.model,
                model_name=model_name,
                imgs_path=f'../saved_data/imgs/{model_name}/',
                X_winTest=X_window_test,
                Y_winTest=Y_window_test
            )

    print(f'\nTest data len = {len(test_data)}\n')

    MTS, MTF, p_success, p_failure = get_mts_mtf(data=test_data)
    r_mks = reactive_makespan(MTF=MTF, MTS=MTS, pf=p_failure, ps=p_success)
    print(f'Reactive policy equation makespan = {r_mks}')
    data_table[0][1] = None
    data_table[1][1] = r_mks
    data_table[2][1] = MTS
    data_table[3][1] = MTF

    # Run simulations
    react_avg_mks, react_mks = run_reactive_simulation(
        episodes=test_data,
        n_simulations=1000,
        verbose=True
    )
    data_table[0][1] = react_avg_mks
    print()

    sim_models = {
        'FCN': makespan_models['FCN'],
        # 'RNN': makespan_models['RNN'],
        'GRU': makespan_models['GRU'],
        # 'LSTM': makespan_models['LSTM'],
        'Transformer': makespan_models['OOP_Transformer_small'],
        # 'Transformer_big': makespan_models['OOP_Transformer']
    }
    plot_models = {
        'FCN': makespan_models['FCN'],
        # 'RNN': makespan_models['RNN'],
        'GRU': makespan_models['GRU'],
        # 'LSTM': makespan_models['LSTM'],
        'Transformer': makespan_models['OOP_Transformer_small'],
        # 'Transformer_big': makespan_models['OOP_Transformer']
    }

    # confidence_list = [0.85, 0.9, 0.95, 0.99]
    confidence_list = [1]

    # To get equation makespan:
    get_eq_makespan = False
    if get_eq_makespan:
        for confidence in confidence_list:
            print(f'For confidence {confidence}:')
            for model_name, model in sim_models.items():
                print(f'\n--> Computing for model {model_name}')
                get_makespan_for_model(
                    model_name=model_name,
                    model=model,
                    episodes=test_data,
                    confidence=confidence,
                    verbose=True
                )

    # To get simulated makespan:
    run_simulation = True
    sim_results = {k: {} for k in confidence_list}
    for confidence in confidence_list:
        if run_simulation:
            sim_results[confidence] = run_makespan_simulation(
                models_to_run=sim_models,
                data=test_data,
                n_simulations = 500,
                confidence=confidence,
                compute=False
            )

            plot_simulation_makespans(
                models=plot_models,
                confidence=confidence,
                reactive_mks=react_mks[:150],
                plot_reactive=True,
                save_plots=True
            )

    if run_simulation:
        for confidence in confidence_list:
            for i, model_name in enumerate(sim_models.keys()):
                if sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['EMS'] == 'N/A':
                    data_table[0][i+2] = 'N/A'
                else:
                    data_table[0][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['makespan_sim_avg']
                data_table[1][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['EMS']
                data_table[2][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['MTS']
                data_table[3][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['MTF']
                data_table[4][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['MTP']
                data_table[5][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['MTN']
                data_table[6][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_TP']
                data_table[7][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_FN']
                data_table[8][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_TN']
                data_table[9][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_FP']
                data_table[10][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_NCS']
                data_table[11][i+2] = sim_results[confidence][f'{model_name}_{int(confidence*100)}']['metrics']['P_NCF']

            print(f'\nConfidence = {confidence}')
            print(tabulate(data_table, headers=headers))

        with open('../saved_data/simulation_results.json', 'w') as f:
            json.dump(sim_results, f)

    # PLotting:
    # Plot 3 episode examples and their classifications by the FCN, GRU and Small Transformer
    plot_examples = False
    if plot_examples:
        indices = [40, 6, 95]
        plot_ft_classification_for_model(
            model_names=plot_models.keys(),
            models=plot_models.values(),
            episodes=[test_data[idx] for idx in indices],
            confidence=0.9
        )

    # Equation and simulation makespan bar plots:
    plot_roc = False
    plot_eq_sim_barplots = False
    plot_monte_carlo_barplots = False
    plot_sim_makespans = False
    for confidence in confidence_list:
        if plot_roc:
            plot_roc_window_data(
                models=sim_models,
                X_data=X_window_test,
                Y_data=Y_window_test,
                confidence=confidence
            )

        if plot_eq_sim_barplots:
            plot_equation_simulation_makespan_barplots(
                models=plot_models,
                confidence=confidence,
                reactive_eq=r_mks,
                reactive_sim=react_mks,
                plot_reactive=False,
                save=True
            )

        if plot_monte_carlo_barplots:
            plot_monte_carlo_simulation_barplots(
                models=plot_models,
                confidence=confidence
            )

        if plot_sim_makespans:
            plot_simulation_makespans(
                models=plot_models,
                confidence=confidence,
                reactive_mks=react_mks[:500],
                plot_reactive=True,
                save_plots=True
            )

