import os, sys, json
sys.path.append(os.path.realpath('../'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
from scipy import stats
from tabulate import tabulate

from run_makespan_simulation import run_reactive_simulation
from utilities.makespan_utils import get_mts_mtf, reactive_makespan

DATA = ['reactive', 'training']
DATA_DIR = f'../../data/instance_data/{"_".join(DATA)}'

if __name__ == '__main__':
    # Load data
    print('Loading data test from file...', end='')
    with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
            test_data = np.load(f, allow_pickle=True)
    print('DONE')

    # Reactive
    MTS, MTF, p_success, p_failure = get_mts_mtf(data=test_data)
    react_mks = reactive_makespan(MTF=MTF, MTS=MTS, pf=p_failure, ps=p_success)
    react_avg_mks, sim_mks = run_reactive_simulation(
        episodes=test_data,
        n_simulations=1000,
        verbose=True
    )
    print('\n')

    # Preemptive
    confidence_levels = [0.85, 0.9, 0.95, 0.99]
    models = ['FCN', 'GRU', 'Transformer']
    data = {}
    for confidence in confidence_levels:
        headers = [f'Confidence {confidence}', 'Reactive', 'FCN', 'GRU', 'Transformer']
        data_table = [
            ['Reactive', None, None, None, None],
            ['FCN', None, None, None, None],
            ['GRU', None, None, None, None],
            ['Transformer', None, None, None, None]
        ]
        data[confidence] = {}
        data[confidence]['Reactive'] = sim_mks[:500]
        data_dir = f'../saved_data/test_data_simulation_confidence_{int(confidence * 100)}'
        for model_name in models:
            # Load data
            with open(f'{data_dir}/{model_name}.json', 'r') as f:
                 sim_data = json.load(f)
            data[confidence][model_name] = sim_data['simulation_makespan_list']

        # Perform Kruskal-Wallis test
        kw_test_result = stats.kruskal(
             sim_mks,
             data[confidence]['FCN'],
             data[confidence]['GRU'],
             data[confidence]['Transformer'],
        )
        print(f'Confidence = {int(confidence*100)}% --> Kruskal-Wallis test result = {kw_test_result}; p-value < 0.05 = {kw_test_result[1] < 0.05}')
        

        # Perform post-hoc Wilcoxon test
        wilcoxon_test_models = ['Reactive'] + models
        for i in range(len(wilcoxon_test_models)):
            for j in range(i):
                data_table[i][j+1] = stats.wilcoxon(
                    x=data[confidence][wilcoxon_test_models[i]],
                    y=data[confidence][wilcoxon_test_models[j]],
                    correction=True
                )[1]

        print(tabulate(data_table, headers=headers))
        print('\n')
