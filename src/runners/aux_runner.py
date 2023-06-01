import sys, os, pickle, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf
import seaborn as sns

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN, GRU, LSTM
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import plot_acc_loss, compute_confusion_matrix, make_probabilities_plots
from utilities.makespan_utils import get_mts_mtf, monitored_makespan
from model_evaluation.plot_model_sizes import plot_model_sizes, plot_model_sizes_bar


class hist_obj:
    def __init__(self, h: dict) -> None:
        self.history = h


if __name__ == '__main__':
    # run_makespan_prediction_for_model(model_name='FCN', verbose=True)
    # model_name = 'OOP_Transformer'
    # hist = np.load(f'../saved_data/histories/{model_name}_history', allow_pickle=True)
    # hist = hist_obj(hist)
    # plot_acc_loss(history=hist, imgs_path=f'../saved_data/imgs/{model_name}/')

    # sizes = {
    #     "FCN": 63992,
    #     "RNN": 17538,
    #     "GRU": 52482,
    #     "LSTM": 69378,
    #     "Big Transformer": 285180,
    #     "Small Transformer": 69948
    # }
    # plot_model_sizes_bar(sizes=sizes, out_name='aux_model_sizes')

    # DATA = ['reactive', 'training']
    # DATA_DIR = f'../../data/data_manager/{"_".join(DATA)}'
    # with open(f'{DATA_DIR}/{"_".join(DATA)}_data_test.npy', 'rb') as f:
    #     test_data = np.load(f, allow_pickle=True)

    # MTS, MTF, _, _ = get_mts_mtf(data=test_data)

    models = [
        'FCN',
        'RNN',
        'GRU',
        'LSTM',
        'VanillaTransformer',
        'OOP_Transformer_small',
        'OOP_Transformer'
    ]
    # confidence_list = [0.85, 0.9, 0.95, 0.99]
    # for confidence in confidence_list:
    #     print(f'For confidence {confidence}:')
    #     for model_name in models:
    #         print(f'--> For model {model_name}')
    #         file_path = f'../saved_data/test_data_makespan_confidence_{int(confidence*100)}/{model_name}.json'
    #         if os.path.exists(file_path):
    #             with open(file_path, 'r') as f:
    #                 model_data = json.load(f)

    #             model_perf = model_data['perf']
    #             model_data['variables']['MTS'] = MTS
    #             model_data['variables']['MTF'] = MTF
    #             model_data['variables']['P_TP'] = model_perf['TP'] / 244
    #             model_data['variables']['P_FN'] = model_perf['FN'] / 244
    #             model_data['variables']['P_TN'] = model_perf['TN'] / 244
    #             model_data['variables']['P_FP'] = model_perf['FP'] / 244
    #             model_data['variables']['P_NCS'] = (model_perf['NCS'] if 'NCS' in model_perf.keys() else 0) / 244
    #             model_data['variables']['P_NCF'] = (model_perf['NCF'] if 'NCF' in model_perf.keys() else 0) / 244
    #             model_data['predicted_makespan'] = monitored_makespan(
    #                 MTF = model_data['variables']['MTF'],
    #                 MTN = model_data['variables']['MTN'],
    #                 MTS = model_data['variables']['MTS'],
    #                 P_TP = model_data['variables']['P_TP'],
    #                 P_FN = model_data['variables']['P_FN'],
    #                 P_TN = model_data['variables']['P_TN'],
    #                 P_FP = model_data['variables']['P_FP'],
    #                 P_NCS = model_data['variables']['P_NCS'],
    #                 P_NCF = model_data['variables']['P_NCF']
    #             )

    #             with open(file_path, 'w') as f:
    #                 json.dump(model_data, f)

    for model_name in models:
        with open(f'../saved_data/test_conf_mats/{model_name}_test_data_perf_at_0.9.json', 'r') as f:
            model_perf = json.load(f)
        print(f'\nFor model {model_name}:')
        print(f'--> Precision = {(model_perf["TP"] / (model_perf["TP"] + model_perf["FP"]))*100:.2f}')
        print(f'--> Recall = {(model_perf["TP"] / (model_perf["TP"] + model_perf["FN"]))*100:.2f}')
        print(f'--> Sensitivity = {(model_perf["TN"] / (model_perf["TN"] + model_perf["FP"]))*100:.2f}')
        print(f'--> Specificity = {(model_perf["TP"] / (model_perf["TP"] + model_perf["FN"]))*100:.2f}')
        print(f'--> Accuracy = {((model_perf["TP"] + model_perf["TN"]) / (model_perf["TP"] + model_perf["FN"] + model_perf["FP"] + model_perf["TN"]))*100:.2f}')