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
from utilities.makespan_utils import run_makespan_prediction_for_model
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

    sizes = {
        "FCN": 63992,
        "RNN": 17538,
        "GRU": 52482,
        "LSTM": 69378,
        "Big Transformer": 285180,
        "Small Transformer": 69948
    }
    plot_model_sizes_bar(sizes=sizes, out_name='aux_model_sizes')