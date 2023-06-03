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
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import compute_confusion_matrix
from utilities.makespan_utils import get_makespan_for_model, get_mts_mtf, scan_output_for_decision, reactive_makespan, monitored_makespan, reactive_makespan, plot_simulation_makespans
from utilities.utils import CounterDict
from utilities.utils import set_size

SRC_PATH = os.path.dirname(os.path.realpath(__file__))
MAIN_PATH = os.path.dirname(os.path.dirname(__file__))

DATA = ['reactive', 'training']
DATA_DIR = f'../../data/data_manager/{"_".join(DATA)}'
MODELS_TO_RUN = [
    'FCN',
    'RNN',
    'GRU',
    'LSTM',
    'VanillaTransformer',
    'OOP_Transformer_small',
    'OOP_Transformer'
]




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


    with open('../saved_data/simulation_results.json', 'r') as f:
        sim_results = json.load(f)

    with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
        test_data = np.load(f, allow_pickle=True)

    MTS, MTF, p_s, p_f = get_mts_mtf(data=test_data)

    # confidence_list = [0.85, 0.9, 0.95, 0.99]
    confidence_list = [0.9]
    models = ['FCN', 'GRU', 'Transformer']
    n_vals = 200

    # Setup
    plt.style.use('seaborn')
    # From Latex \textwidth
    fig_width = 800
    tex_fonts = {
        # Use LaTeX to write all text
        # "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "font.size": 14,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    }
    plt.rcParams.update(tex_fonts)

    # fig, axes = plt.subplots(len(confidence_list), len(models), figsize=set_size(fig_width, subplots=(len(confidence_list), len(models))))
    fig, axes = plt.subplots(len(confidence_list), len(models), figsize=(15, 5))

    for i, confidence in enumerate(confidence_list):
        for j, model in enumerate(models):
            with open(f'../saved_data/test_data_makespan_confidence_{int(confidence*100)}/{model}.json', 'r') as f:
                model_makespan = json.load(f)

            model_metrics = model_makespan['variables']

            # Prob failure
            probs_f = np.linspace(0.0, 0.8, num=n_vals, endpoint=1)
            react_t = []
            model_t = []
            for k, p_f_k in enumerate(probs_f):
                p_s_k = 1.0 - p_f_k

                model_P_tp_k = model_metrics['P_TP'] * p_s_k/p_s
                model_P_fn_k = model_metrics['P_FN'] * p_s_k/p_s
                model_P_ncs_k = model_metrics['P_NCS'] * p_s_k/p_s
                model_P_tn_k = model_metrics['P_TN'] * p_f_k/p_f
                model_P_fp_k = model_metrics['P_FP'] * p_f_k/p_f
                model_P_ncf_k = model_metrics['P_NCF'] * p_f_k/p_f

                react_t.append(reactive_makespan(MTS, MTF, p_s_k, p_f_k))
                model_t.append(monitored_makespan(
                    MTS,
                    MTF,
                    model_metrics['MTN'],
                    model_P_tp_k,
                    model_P_fn_k,
                    model_P_tn_k,
                    model_P_fp_k,
                    model_P_ncs_k,
                    model_P_ncf_k
                ))

            axes[j].plot(probs_f, react_t, label='Reactive')
            # axes[j].plot(baseline_metrics['MTN'], reactive_sim_makespan, marker='x', color='black')

            axes[j].plot(probs_f, model_t, label=model)
            # axes[j].plot(model_metrics['MTN'], model_sim_makespan, marker='o', color='gray')

            axes[j].plot(probs_f, [r - m for r, m in zip(react_t, model_t)], label=f'Time saved')

            # axes[j].set_title(f'Confidence {int(confidence * 100)}%')
            # axes[j].set_xlabel('Probability of failure')
            # axes[j].set_ylabel('Makespan [s]')
            axes[j].legend()

    # fig.suptitle(f'Confidence {int(confidence * 100)}%')
    fig.supxlabel('Probability of failure')
    fig.supylabel('Makespan [s]')
    plt.tight_layout()
    plt.savefig(f'../saved_data/imgs/simulation/Pf_all_models_and_confidence.png')
    plt.clf()


    # baseline = 'FCN'
    # model_name = 'Transformer'

    # for confidence in confidence_list:
    #     with open(f'../saved_data/test_data_makespan_confidence_{int(confidence*100)}/{baseline}.json', 'r') as f:
    #         baseline_makespan = json.load(f)

    #     with open(f'../saved_data/test_data_makespan_confidence_{int(confidence*100)}/{model_name}.json', 'r') as f:
    #         model_makespan = json.load(f)

    #     reactive_sim_makespan = 66.828
    #     reactive_eq_makespan = 66.6894

    #     baseline_key = f'{baseline}_{int(confidence*100)}'
    #     # baseline_metrics = sim_results[str(confidence)][baseline_key]['metrics']
    #     baseline_metrics = baseline_makespan['variables']
    #     baseline_sim_makespan = sim_results[str(confidence)][baseline_key]['makespan_sim_avg']

    #     model_key = f'{model_name}_{int(confidence*100)}'
    #     # model_metrics = sim_results[str(confidence)][model_key]['metrics']
    #     model_metrics = model_makespan['variables']
    #     model_sim_makespan = sim_results[str(confidence)][model_key]['makespan_sim_avg']

    #     # P_TP var independent
    #     n_vals = 200
    #     prob = np.linspace(0.0, 0.99, num=n_vals)
    #     result_react = [reactive_eq_makespan] * n_vals
    #     result_baseline = abs(monitored_makespan(
    #         MTS=baseline_metrics['MTS'],
    #         MTF=baseline_metrics['MTF'],
    #         MTN=baseline_metrics['MTN'],
    #         P_TP=prob,
    #         P_FN=baseline_metrics['P_FN'],
    #         P_TN=baseline_metrics['P_TN'],
    #         P_FP=baseline_metrics['P_FP'],
    #         P_NCF=baseline_metrics['P_NCF'],
    #         P_NCS=baseline_metrics['P_NCS']
    #     ))
    #     result_model = abs(monitored_makespan(
    #         MTS=model_metrics['MTS'],
    #         MTF=model_metrics['MTF'],
    #         MTN=model_metrics['MTN'],
    #         P_TP=prob,
    #         P_FN=model_metrics['P_FN'],
    #         P_TN=model_metrics['P_TN'],
    #         P_FP=model_metrics['P_FP'],
    #         P_NCF=model_metrics['P_NCF'],
    #         P_NCS=model_metrics['P_NCS']
    #     ))
    #     plt.plot(prob, result_react, label='Reactive')
    #     plt.plot(baseline_metrics['P_TP'], reactive_sim_makespan, marker='x', color='black')

    #     plt.plot(prob, result_baseline, label='FCN')
    #     plt.plot(baseline_metrics['P_TP'], baseline_sim_makespan, marker='o', color='black')

    #     plt.plot(prob, result_model, label='Transformer')
    #     plt.plot(model_metrics['P_TP'], model_sim_makespan, marker='o', color='gray')

    #     plt.plot(prob, result_react - result_baseline, label='Time saved FCN')
    #     plt.plot(prob, result_react - result_model, label='Time saved Transformer')

    #     plt.grid(visible=True)
    #     plt.legend()
    #     plt.savefig(f'../saved_data/imgs/simulation/sim_img_TP_confidence_{int(confidence*100)}.png')
    #     plt.clf()

    #     # P_TN var independent
    #     prob = np.linspace(0.0, 0.6, num=n_vals)
    #     result_baseline = abs(monitored_makespan(
    #         MTS=baseline_metrics['MTS'],
    #         MTF=baseline_metrics['MTF'],
    #         MTN=baseline_metrics['MTN'],
    #         P_TP=baseline_metrics['P_TP'],
    #         P_FN=baseline_metrics['P_FN'],
    #         P_TN=prob,
    #         P_FP=baseline_metrics['P_FP'],
    #         P_NCF=baseline_metrics['P_NCF'],
    #         P_NCS=baseline_metrics['P_NCS']
    #     ))
    #     result_model = abs(monitored_makespan(
    #         MTS=model_metrics['MTS'],
    #         MTF=model_metrics['MTF'],
    #         MTN=model_metrics['MTN'],
    #         P_TP=model_metrics['P_TP'],
    #         P_FN=model_metrics['P_FN'],
    #         P_TN=prob,
    #         P_FP=model_metrics['P_FP'],
    #         P_NCF=model_metrics['P_NCF'],
    #         P_NCS=model_metrics['P_NCS']
    #     ))

    #     plt.plot(prob, result_react, label='Reactive')
    #     plt.plot(baseline_metrics['P_TN'], reactive_sim_makespan, marker='x', color='black')

    #     plt.plot(prob, result_baseline, label='FCN')
    #     plt.plot(baseline_metrics['P_TN'], baseline_sim_makespan, marker='o', color='black')

    #     plt.plot(prob, result_model, label='Transformer')
    #     plt.plot(model_metrics['P_TN'], model_sim_makespan, marker='o', color='gray')

    #     plt.plot(prob, result_react - result_baseline, label='Time saved FCN')
    #     plt.plot(prob, result_react - result_model, label='Time saved Transformer')

    #     plt.grid(visible=True)
    #     plt.legend()
    #     plt.savefig(f'../saved_data/imgs/simulation/sim_img_TN_confidence_{int(confidence*100)}.png')
    #     plt.clf()

    #     # MTF var independent
    #     mtf = np.linspace(0.0, 100, num=n_vals)
    #     result_baseline = abs(monitored_makespan(
    #         MTS=baseline_metrics['MTS'],
    #         MTF=mtf,
    #         MTN=baseline_metrics['MTN'],
    #         P_TP=baseline_metrics['P_TP'],
    #         P_FN=baseline_metrics['P_FN'],
    #         P_TN=baseline_metrics['P_TN'],
    #         P_FP=baseline_metrics['P_FP'],
    #         P_NCF=baseline_metrics['P_NCF'],
    #         P_NCS=baseline_metrics['P_NCS']
    #     ))
    #     result_model = abs(monitored_makespan(
    #         MTS=model_metrics['MTS'],
    #         MTF=mtf,
    #         MTN=model_metrics['MTN'],
    #         P_TP=model_metrics['P_TP'],
    #         P_FN=model_metrics['P_FN'],
    #         P_TN=model_metrics['P_TN'],
    #         P_FP=model_metrics['P_FP'],
    #         P_NCF=model_metrics['P_NCF'],
    #         P_NCS=model_metrics['P_NCS']
    #     ))

    #     plt.plot(mtf, result_react, label='Reactive')
    #     plt.plot(baseline_metrics['MTF'], reactive_sim_makespan, marker='x', color='black')

    #     plt.plot(mtf, result_baseline, label='FCN')
    #     plt.plot(baseline_metrics['MTF'], baseline_sim_makespan, marker='o', color='black')

    #     plt.plot(mtf, result_model, label='Transformer')
    #     plt.plot(model_metrics['MTF'], model_sim_makespan, marker='o', color='gray')

    #     plt.plot(mtf, result_react - result_baseline, label='Time saved FCN')
    #     plt.plot(mtf, result_react - result_model, label='Time saved Transformer')

    #     plt.grid(visible=True)
    #     plt.legend()
    #     plt.savefig(f'../saved_data/imgs/simulation/sim_img_MTF_confidence_{int(confidence*100)}.png')
    #     plt.clf()

    #     # MTN var independent
    #     mtn = np.linspace(0.0, 100, num=n_vals)
    #     result_baseline = abs(monitored_makespan(
    #         MTS=baseline_metrics['MTS'],
    #         MTF=baseline_metrics['MTF'],
    #         MTN=mtn,
    #         P_TP=baseline_metrics['P_TP'],
    #         P_FN=baseline_metrics['P_FN'],
    #         P_TN=baseline_metrics['P_TN'],
    #         P_FP=baseline_metrics['P_FP'],
    #         P_NCF=baseline_metrics['P_NCF'],
    #         P_NCS=baseline_metrics['P_NCS']
    #     ))
    #     result_model = abs(monitored_makespan(
    #         MTS=model_metrics['MTS'],
    #         MTF=model_metrics['MTF'],
    #         MTN=mtn,
    #         P_TP=model_metrics['P_TP'],
    #         P_FN=model_metrics['P_FN'],
    #         P_TN=model_metrics['P_TN'],
    #         P_FP=model_metrics['P_FP'],
    #         P_NCF=model_metrics['P_NCF'],
    #         P_NCS=model_metrics['P_NCS']
    #     ))

    #     plt.plot(mtn, result_react, label='Reactive')
    #     plt.plot(baseline_metrics['MTN'], reactive_sim_makespan, marker='x', color='black')

    #     plt.plot(mtn, result_baseline, label='FCN')
    #     plt.plot(baseline_metrics['MTN'], baseline_sim_makespan, marker='o', color='black')

    #     plt.plot(mtn, result_model, label='Transformer')
    #     plt.plot(model_metrics['MTN'], model_sim_makespan, marker='o', color='gray')

    #     plt.plot(mtn, result_react - result_baseline, label='Time saved FCN')
    #     plt.plot(mtn, result_react - result_model, label='Time saved Transformer')

    #     plt.grid(visible=True)
    #     plt.legend()
    #     plt.savefig(f'../saved_data/imgs/simulation/sim_img_MTN_confidence_{int(confidence*100)}.png')
    #     plt.clf()

    #     # Prob failure
    #     probs_f = np.linspace(0.0, 0.8, num=n_vals, endpoint=1)
    #     react_t = []
    #     baseline_t = []
    #     model_t = []
    #     for i, p_f_i in enumerate(probs_f):
    #         p_s_i = 1.0 - p_f_i

    #         baseline_P_tp_i = baseline_metrics['P_TP'] * p_s_i/p_s
    #         baseline_P_fn_i = baseline_metrics['P_FN'] * p_s_i/p_s
    #         baseline_P_ncs_i = baseline_metrics['P_NCS'] * p_s_i/p_s
    #         baseline_P_tn_i = baseline_metrics['P_TN'] * p_f_i/p_f
    #         baseline_P_fp_i = baseline_metrics['P_FP'] * p_f_i/p_f
    #         baseline_P_ncf_i = baseline_metrics['P_NCF'] * p_f_i/p_f

    #         model_P_tp_i = model_metrics['P_TP'] * p_s_i/p_s
    #         model_P_fn_i = model_metrics['P_FN'] * p_s_i/p_s
    #         model_P_ncs_i = model_metrics['P_NCS'] * p_s_i/p_s
    #         model_P_tn_i = model_metrics['P_TN'] * p_f_i/p_f
    #         model_P_fp_i = model_metrics['P_FP'] * p_f_i/p_f
    #         model_P_ncf_i = model_metrics['P_NCF'] * p_f_i/p_f

    #         react_t.append(reactive_makespan(MTS, MTF, p_s_i, p_f_i))
    #         baseline_t.append(monitored_makespan(
    #             MTS,
    #             MTF,
    #             baseline_metrics['MTN'],
    #             baseline_P_tp_i,
    #             baseline_P_fn_i,
    #             baseline_P_tn_i,
    #             baseline_P_fp_i,
    #             baseline_P_ncs_i,
    #             baseline_P_ncf_i
    #         ))
    #         model_t.append(monitored_makespan(
    #             MTS,
    #             MTF,
    #             model_metrics['MTN'],
    #             model_P_tp_i,
    #             model_P_fn_i,
    #             model_P_tn_i,
    #             model_P_fp_i,
    #             model_P_ncs_i,
    #             model_P_ncf_i
    #         ))

    #     plt.plot(probs_f, react_t, label='Reactive')
    #     # plt.plot(baseline_metrics['MTN'], reactive_sim_makespan, marker='x', color='black')

    #     plt.plot(probs_f, baseline_t, label='FCN')
    #     # plt.plot(baseline_metrics['MTN'], baseline_sim_makespan, marker='o', color='black')

    #     plt.plot(probs_f, model_t, label='Transformer')
    #     # plt.plot(model_metrics['MTN'], model_sim_makespan, marker='o', color='gray')

    #     plt.plot(probs_f, [r - b for r, b in zip(react_t, baseline_t)], label='Time saved FCN')
    #     plt.plot(probs_f, [r - m for r, m in zip(react_t, model_t)], label='Time saved Transformer')

    #     plt.grid(visible=True)
    #     plt.legend()
    #     plt.savefig(f'../saved_data/imgs/simulation/sim_img_Pf_confidence_{int(confidence*100)}.png')
    #     plt.clf()

