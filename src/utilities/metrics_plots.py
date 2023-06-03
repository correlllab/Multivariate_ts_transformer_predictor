import sys, os, json
sys.path.append(os.path.realpath('../utilities'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
import tensorflow as tf
import seaborn as sns

from sklearn import metrics
from utilities.utils import CounterDict, set_size
from helper_functions import scan_output_for_decision, graph_episode_output


def plot_acc_loss(history, imgs_path, save_plot=True):
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

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['categorical_accuracy']
    val_accuracy = history.history['val_categorical_accuracy']

    fig, axes = plt.subplots(2, 2, figsize=set_size(fig_width, subplots=(2, 2)))
    fig.tight_layout()
    axes[0, 0].plot(accuracy, label='Training accuracy')
    axes[0, 0].title.set_text('Training accuracy over epochs')
    axes[0, 1].plot(np.array(loss), label='Training loss', color='orange')
    axes[0, 1].title.set_text('Training loss over epochs')



    axes[1, 0].plot(val_accuracy, label='Validation accuracy')
    axes[1, 0].title.set_text('Valiadation accuracy over epochs')
    axes[1, 1].plot(np.array(val_loss), label='Validation loss', color='orange')
    axes[1, 1].title.set_text('Valiadation loss over epochs')

    if save_plot:
        plt.savefig(imgs_path + 'acc_loss_plots.png')
        plt.clf()
    else:
        plt.show()


def plot_evaluation_on_test_window_data(model: tf.keras.Model, model_name: str, X_data: list, Y_data: list, confidence: float = 0.90, simulation: bool = False, verbose: bool = True):
    perf = CounterDict()
    list_of_res = []

    for epNo in range(len(X_data)):
        if verbose:
            print('>', end=' ')
        with tf.device('/GPU:0'):
            pred = model.predict(X_data[epNo])
            ans, aDx = scan_output_for_decision(pred, Y_data[epNo][0], threshold=confidence)
            list_of_res.append(get_decision(pred, Y_data[epNo][0], threshold=confidence))
            graph_episode_output(
                res=pred,
                index=epNo,
                ground_truth=Y_data[epNo][0],
                out_decision=(ans, aDx),
                imgs_path=model.imgs_path,
                net=model_name,
                ts_ms = 20,
                save_fig=True
            )
            print()

            perf.count( ans )

            if ans == 'NC':
                if all(Y_data[epNo][0] == tf.keras.utils.to_categorical(0.0, num_classes=2)):
                    perf.count('NCS')
                elif all(Y_data[epNo][0] == tf.keras.utils.to_categorical(1.0, num_classes=2)):
                    perf.count('NCF')

    print( '\n', model.file_name, '\n', perf )

    try:
        confMatx = {
            # Actual Positives
            'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
            # Actual Negatives
            'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
            'NC' : (perf['NC'] if ('NC' in perf) else 0) / len( X_data ),
            'NCS' : (perf['NCS'] if ('NCS' in perf) else 0) / len( X_data ),
            'NCF' : (perf['NCF'] if ('NCF' in perf) else 0) / len( X_data ),
        }
    except ZeroDivisionError as e:
        print(f'Building VanillaTransformer confusion matrix: {e}')
        confMatx = None
        plot = False

    print(f'For model {model_name}:\n{confMatx}')
    with open(f'../saved_data/test_conf_mats/{model_name}_test_data_perf_at_{confidence}.json', 'w') as f:
        json.dump(perf, f)
    with open(f'../saved_data/test_conf_mats/{model_name}_test_data_conf_mat_at_{confidence}.json', 'w') as f:
        json.dump(confMatx, f)

    if plot:
        arr = [
            [confMatx['TP'], confMatx['FP']],
            [confMatx['FN'], confMatx['TN']]
            ]
        sns.set(font_scale=1.5)
        conf_mat = sns.heatmap(arr, annot=True).get_figure()
        if simulation:
            sim_path = f'../saved_data/simulation_{confidence}'
            if not os.path.exists(sim_path):
                os.makedirs(sim_path)
            conf_mat.savefig(f'{sim_path}/confusion_matrix_{int(confidence*100)}.png')
        else:
            conf_mat.savefig(model.imgs_path + 'confusion_matrix_' + str(int(confidence*100)) + '.png')
        plt.close('all')



    return list_of_res, confMatx



def compute_confusion_matrix(model, model_name, file_name, imgs_path, X_winTest, Y_winTest, confidence=0.90, simulation=False, plot=False):
    perf = CounterDict()

    for epNo in range( len( X_winTest ) ):
        print( '>', end=' ' )
        with tf.device('/GPU:0'):
            res = model.predict( X_winTest[epNo] )
            # res = model.predict( X_winTest[epNo][350:, :, :] )
            ans, aDx = scan_output_for_decision( res, Y_winTest[epNo][0], threshold = confidence )
            perf.count( ans )

            if ans == 'NC':
                if all(Y_winTest[epNo][0] == tf.keras.utils.to_categorical(0.0, num_classes=2)):
                    perf.count('NCS')
                elif all(Y_winTest[epNo][0] == tf.keras.utils.to_categorical(1.0, num_classes=2)):
                    perf.count('NCF')
            
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
            'NCS' : (perf['NCS'] if ('NCS' in perf) else 0) / len( X_winTest ),
            'NCF' : (perf['NCF'] if ('NCF' in perf) else 0) / len( X_winTest ),
        }
    except ZeroDivisionError as e:
        print(f'Building VanillaTransformer confusion matrix: {e}')
        confMatx = None
        plot = False

    print(f'For model {model_name}:\n{confMatx}')
    with open(f'../saved_data/test_conf_mats/{model_name}_test_data_perf_at_{confidence}.json', 'w') as f:
        json.dump(perf, f)
    with open(f'../saved_data/test_conf_mats/{model_name}_test_data_conf_mat_at_{confidence}.json', 'w') as f:
        json.dump(confMatx, f)

    if plot:
        arr = [
            [confMatx['TP'], confMatx['FP']],
            [confMatx['FN'], confMatx['TN']]
            ]
        sns.set(font_scale=1.5)
        conf_mat = sns.heatmap(arr, annot=True).get_figure()
        if simulation:
            sim_path = f'../saved_data/simulation_{confidence}'
            if not os.path.exists(sim_path):
                os.makedirs(sim_path)
            conf_mat.savefig(f'{sim_path}/confusion_matrix_{int(confidence*100)}.png')
        else:
            conf_mat.savefig(imgs_path + 'confusion_matrix_' + str(int(confidence*100)) + '.png')
        plt.close('all')

    return confMatx


def make_probabilities_plots(model, model_name, imgs_path, X_winTest, Y_winTest):
    for epNo in range( len( X_winTest ) ):
        with tf.device('/GPU:0'):
            print(epNo, ':')
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


def get_decision( output, trueLabel, threshold = 0.90 ):
    for i, row in enumerate( output ):
        if np.amax( row ) >= threshold:
            break

    if i >= len(output) - 1:
        return 'NC'

    if np.amax(row) == row[0]:
        return 1 - row[0]

    return row[1]


def plot_roc_window_data(models: dict, X_data: list, Y_data: list, confidence: float = 0.9):
    img_path = f'../saved_data/imgs/roc_curves_confidence_{int(confidence * 100)}.png'

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
    cmap = cm.get_cmap('Spectral', len(models.keys()))
    colors = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    i = 0

    results = {}

    for model_name, model in models.items():
        print(f'--> For model {model_name}...')
        res = []
        labels = []
        for epNo in range(len(X_data)):
            with tf.device('/GPU:0'):
                pred = model.predict(X_data[epNo])
                value = get_decision(pred, Y_data[epNo][0], threshold=confidence)
                if value != 'NC':
                    res.append(value)
                    labels.append(Y_data[epNo][0][1])

        # labels = [l[0][1] for l in Y_data]
        if len(res) != 0:
            fpr, tpr, _ = metrics.roc_curve(labels, res)
            auc = metrics.roc_auc_score(labels, res)

            results[model_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc}

            label_name = model_name
            if model_name == 'Transformer':
                label_name = 'Small Transformer'
            elif model_name == 'Transformer_big':
                label_name = 'Big Transformer'

            plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc:.4f})', color=colors[i])

        i += 1

    with open(f'../saved_data/ROC/results_confidence_{int(confidence * 100)}.json', 'w') as f:
        json.dump(results, f)

    plt.axline((1, 1), slope=1, linestyle='dashed', color='black')
    plt.tight_layout()
    plt.legend()
    plt.savefig(img_path)
    plt.clf()
    plt.close('all')


def plot_roc_window_data_from_preds(models: dict, preds: dict, Y_data: list, confidence: float = 0.9, img_path: str = '../saved_data/imgs/roc_curves.png'):
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
    cmap = cm.get_cmap('Spectral', len(models.keys()))
    colors = [rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
    i = 0
    for model_name, pred in preds.items():
        labels = [l[0][1] for l in Y_data]
        fpr, tpr, _ = metrics.roc_curve(labels, pred)
        auc = metrics.roc_auc_score(labels, pred)
        label_name = model_name
        if model_name == 'Transformer':
            label_name = 'Small Transformer'
        elif model_name == 'Transformer_big':
            label_name = 'Big Transformer'
        plt.plot(fpr, tpr, label=f'{label_name} (AUC = {auc:.4f})', color=colors[i])

        i += 1

    plt.axline((1, 1), slope=1, linestyle='dashed', color='black')
    plt.tight_layout()
    plt.legend()
    plt.savefig(img_path)
    plt.clf()
    plt.close('all')


def plot_equation_simulation_makespan_barplots(models: dict, confidence: float, reactive_eq: float, reactive_sim: list, plot_reactive: bool = True, save: bool = True):
    img_path = f'../saved_data/imgs/equation_simulation_makespans_barplot_confidence_{int(confidence * 100)}.png'
    eq_file_dir = f'../saved_data/test_data_makespan_confidence_{int(confidence * 100)}'
    sim_file_dir = f'../saved_data/test_data_simulation_confidence_{int(confidence * 100)}'

    makespans = {model_name: {'eq': None, 'sim': []} for model_name in models.keys()}
    for model_name in models.keys():
        with open(f'{eq_file_dir}/{model_name}.json', 'r') as f:
            data = json.load(f)
        makespans[model_name]['eq'] = data['predicted_makespan']

        with open(f'{sim_file_dir}/{model_name}.json', 'r') as f:
            data = json.load(f)
        makespans[model_name]['sim'] = data['simulation_makespan_list']

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
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    }
    plt.rcParams.update(tex_fonts)

    fig, axes = plt.subplots(1, 1, figsize=set_size(fig_width, subplots=(1, 1)))

    width = 0.25
    if plot_reactive:
        r = np.arange(len(list(models.keys())) + 1)
    else:
        r = np.arange(len(list(models.keys())) )

    eq_heights = []
    if plot_reactive and reactive_eq:
        eq_heights.append(reactive_eq)
    for v in makespans.values():
        eq_heights.append(v['eq'])

    sim_heights = []
    sim_errors = []
    if plot_reactive and reactive_sim != []:
        sim_heights.append(np.mean(reactive_sim))
        sim_errors.append(np.std(reactive_sim))
    for v in makespans.values():
        sim_heights.append(np.mean(v['sim']))
        sim_errors.append(np.std(v['sim']))

    axes.bar(
        x=r,
        height=eq_heights,
        width=width,
        label='Equation makespans [s]',
        alpha=0.5
    )

    axes.bar(
        x=r + width,
        height=sim_heights,
        yerr=sim_errors,
        width=width,
        label='Simulation makespans [s]',
        alpha=0.5,
        capsize=10
    )

    labels = []
    if plot_reactive and reactive_eq and reactive_sim != []:
        labels.append('Reactive')
    for model_name in makespans.keys():
        if model_name == 'Transformer':
            labels.append('Small Transformer')
        else:
            labels.append(model_name)
    plt.xticks(r + width/2, labels)
    plt.tight_layout()
    plt.legend()
    plt.savefig(img_path)
    plt.clf()
    plt.close('all')


def plot_monte_carlo_simulation_barplots(models: dict, confidence: float, save: bool = True):
    img_path = f'../saved_data/imgs/monte_carlo_simulation_barplot_confidence_{int(confidence * 100)}.png'
    sim_file_dir = f'../saved_data/test_data_simulation_confidence_{int(confidence * 100)}'

    makespans = {model_name: {'sim_eq': None, 'sim': []} for model_name in models.keys()}
    for model_name in models.keys():
        with open(f'{sim_file_dir}/{model_name}.json', 'r') as f:
            data = json.load(f)
        makespans[model_name]['sim'] = data['simulation_makespan_list']
        makespans[model_name]['sim_eq'] = data['equation_predicted_makespan']

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
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    }
    plt.rcParams.update(tex_fonts)

    fig, axes = plt.subplots(1, 1, figsize=set_size(fig_width, subplots=(1, 1)))

    width = 0.25
    r = np.arange(len(list(models.keys())) )

    eq_heights = []
    for v in makespans.values():
        eq_heights.append(v['sim_eq'])

    sim_heights = []
    sim_errors = []
    for v in makespans.values():
        sim_heights.append(np.mean(v['sim']))
        sim_errors.append(np.std(v['sim']))

    axes.bar(
        x=r,
        height=eq_heights,
        width=width,
        label='Equation makespans [s]',
        alpha=0.5
    )

    axes.bar(
        x=r + width,
        height=sim_heights,
        yerr=sim_errors,
        width=width,
        label='Simulation makespans [s]',
        alpha=0.5,
        capsize=10
    )

    labels = []
    for model_name in makespans.keys():
        if model_name == 'Transformer':
            labels.append('Small Transformer')
        else:
            labels.append(model_name)
    plt.xticks(r + width/2, labels)
    plt.tight_layout()
    plt.legend()
    plt.savefig(img_path)
    plt.clf()
    plt.close('all')