import sys, os
sys.path.append(os.path.realpath('../utils'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from utils.utils import CounterDict
from helper_functions import scan_output_for_decision, graph_episode_output


def plot_acc_loss(history, imgs_path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['categorical_accuracy']
    val_accuracy = history.history['val_categorical_accuracy']

    fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    axes[0, 0].plot(accuracy, label='Training accuracy')
    axes[0, 0].title.set_text('Training accuracy over epochs')
    axes[0, 1].plot(np.array(loss), label='Training loss', color='orange')
    axes[0, 1].title.set_text('Training loss over epochs')



    axes[1, 0].plot(val_accuracy, label='Validation accuracy')
    axes[1, 0].title.set_text('Valiadation accuracy over epochs')
    axes[1, 1].plot(np.array(val_loss), label='Validation loss', color='orange')
    axes[1, 1].title.set_text('Valiadation loss over epochs')


    plt.savefig(imgs_path + 'acc_loss_plots.png')
    plt.clf()


def compute_confusion_matrix(model, file_name, imgs_path, X_winTest, Y_winTest, plot=False):
    perf = CounterDict()

    for epNo in range( len( X_winTest ) ):    
        print( '>', end=' ' )
        with tf.device('/GPU:0'):
            res = model.predict( X_winTest[epNo] )
            ans, aDx = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = 0.90 )
            perf.count( ans )
            
    print( '\n', file_name, '\n', perf )

    try:
        confMatx = {
            # Actual Positives
            'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            # Actual Negatives
            'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'NC' : (perf['NC'] if ('NC' in perf) else 0) / len( X_winTest ),
        }
    except ZeroDivisionError as e:
        print(f'Building VanillaTransformer confusion matrix: {e}')
        confMatx = None
        plot = False

    print( confMatx )

    if plot:
        arr = [
            [confMatx['TP'], confMatx['FP']],
            [confMatx['FN'], confMatx['TP']]
            ]
        conf_mat = sns.heatmap(arr, annot=True).get_figure()
        conf_mat.savefig(imgs_path + 'confusion_matrix.png')


def make_probabilities_plots(model, model_name, imgs_path, X_winTest, Y_winTest):
    for epNo in range( len( X_winTest ) ):
        with tf.device('/GPU:0'):
            res = model.predict( X_winTest[epNo] )
            print( Y_winTest[epNo][0], '\n' )
            out_decision = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = 0.90 )
            print(out_decision)
            graph_episode_output(
                res=res,
                index=epNo,
                ground_truth=Y_winTest[epNo][0],
                out_decision=out_decision,
                imgs_path=imgs_path,
                net=model_name,
                ts_ms = 20,
                save_fig=True
            )
            print()