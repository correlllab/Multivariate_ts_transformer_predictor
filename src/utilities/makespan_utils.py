import sys, os, json
import random
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from random import choice
from data_management.data_preprocessing import DataPreprocessing
from utilities.utils import CounterDict

# Classes --------------------------------------------------------------------------
class EpisodePerf:
    """ Container class for per-episode performance """
    def __init__( self, trueLabel = -1, answer = 'NC', runTime = float('nan'),
                  TTP = float('nan'), TTN = float('nan'), TTS = float('nan'), TTF = float('nan') ):
        self.trueLabel = trueLabel
        self.answer    = answer
        self.runTime   = runTime
        self.TTP       = TTP
        self.TTN       = TTN
        self.TTS       = TTS
        self.TTF       = TTF

# Functions -----------------------------------------------------------------------
def scan_output_for_decision( output, trueLabel, threshold = 0.90 ):
    for i, row in enumerate( output ):
        if np.amax( row ) >= threshold:
            if np.dot( row, trueLabel[i] ) >= threshold:
                ans = 'T'
            else:
                ans = 'F'
            if row[0] > row[1]:
                ans += 'P'
            else:
                ans += 'N'
            return ans, i, row
    return 'NC', output.shape[0], row


def expected_makespan( MTF, MTN, MTS, P_TP, P_FN, P_TN, P_FP ):
    """ Return the expected makespan for the given parameters """
    return -( MTF*P_FP + MTN*P_FN + MTN*P_TN + MTS*P_TP + 1) / (P_FN + P_FP + P_TN - 1)


def monitored_makespan( MTS, MTF, MTN, P_TP, P_FN, P_TN, P_FP, P_NCS, P_NCF ):
    """ Closed form makespan for monitored case, symbolic simplification from Mathematica """
    return (1 + MTF*(P_FP + P_NCF) + MTS*(P_NCS + P_TP) + MTN*(P_FN + P_TN) ) / (1 - P_FN - P_FP - P_TN)


def make_predictions(model_name: str, model: tf.keras.Model, trunc_data, verbose: bool = False):
    episode_predictions = []
    rolling_window_width = int(7.0 * 50)
    print( f"Each window is {rolling_window_width} timesteps long!" )
    for ep_index, episode in enumerate(trunc_data):
        with tf.device('/GPU:0'):
            print(f'Episode {ep_index}')
            time_steps = episode.shape[0]
            n_windows = time_steps - rolling_window_width + 1
            episode_X_list = []
            # episode_label_list = []
            true_label = None
            for i in range(n_windows):
                # TODO: make the window centered (i.e. i +- (window_width / 2)) ??
                episode_window = episode[i:i + rolling_window_width, :]
                episode_X_list.append(episode_window[:, 1:7])

            episode_X = tf.stack(episode_X_list)
            # episode_label = tf.stack(tf.keras.utilities.to_categorical(episode_label_list, num_classes=2))
            if episode_window[0, 7] == 0.0:
                true_label = 1.0
            elif episode_window[0, 7] == 1.0:
                true_label = 0.0
            prediction = model.predict(episode_X, use_multiprocessing=True)
            episode_predictions.append([prediction, true_label])

    np.save(f'../saved_data/episode_predictions/{model_name}_episode_predictions.npy', episode_predictions, allow_pickle=True)



def classify(model: tf.keras.Model, episode: np.ndarray, true_label: float, window_width: int):
    with tf.device('/GPU:0'):
        time_steps = episode.shape[0]
        n_windows = time_steps - window_width + 1
        episode_X_list = []
        for i in range(n_windows):
            episode_window = episode[i:i + window_width, :]
            episode_X_list.append(episode_window[:, 1:7])
        episode_X = tf.stack(episode_X_list)
        prediction = model.predict(episode_X, use_multiprocessing=True)
        ans, t_c, row = scan_output_for_decision(
            np.array(prediction),
            np.array([tf.keras.utils.to_categorical(true_label, num_classes=2)] * len(prediction)),
            threshold=0.9
        )

    # Return the classification answer and the time (in seconds) at which the classification happened
    return ans, t_c * (20 / 1000)


def run_simulation(model: tf.keras.Model, episodes: list, n_simulations: int = 1000, verbose: bool = False):
    rolling_window_width = int(7.0 * 50)
    total_time = 0
    mks = []
    for i in range(n_simulations):
        if verbose:    
            print(f'Simulation {i+1}/{n_simulations}:')
        win = False
        makespan = 0
        while not win:
            ep = random.choice(episodes)
            episode_time = ep.shape[0] * (20 / 1000)
            true_label = 0.0
            if ep[0, 7] == 0.0:
                true_label = 1.0
            ans, t_c = classify(model=model, episode=ep, true_label=true_label, window_width=rolling_window_width)
            if ans == 'NC':
                # If true_label == failure
                if true_label == 1.0:
                    makespan += episode_time
                # If true_label == success
                elif true_label == 0.0:
                    makespan += episode_time
                    win = True
            else:
                if ans == 'TN' or ans == 'FP':
                    makespan += t_c
                elif ans == 'TP' or ans == 'FN':
                    makespan += episode_time
                    win = True
        total_time += makespan
        mks.append(makespan)
    return total_time / n_simulations, mks



def run_makespan_prediction_for_model(model_name: str, verbose: bool = False):
    perf = CounterDict()

    N_posCls  = 0 # --------- Number of positive classifications
    N_negCls  = 0 # --------- Number of negative classifications
    N_success = 0 # --------- Number of true task successes
    N_failure = 0 # --------- Number of true task failures
    ts_s      = 20.0/1000.0 # Seconds per timestep
    MTP       = 0.0 # ------- Mean Time to Positive Classification
    MTN       = 0.0 # ------- Mean Time to Negative Classification
    MTS       = 0.0 # ------- Mean Time to Success
    MTF       = 0.0 # ------- Mean Time to Failure
    # xList     = dp.X_winTest
    # yList     = dp.Y_winTest
    # winLen    = dp.truncData[0].shape[1]

    episode_predictions = np.array(np.load(f'../saved_data/episode_predictions/{model_name}_episode_predictions.npy', allow_pickle=True))
    if verbose:
        print(f'\nEpisode predictions matrix shape = {episode_predictions.shape}\n')

    timed_episodes = []

    rolling_window_width = int(7.0 * 50)
    print( f"Each window is {rolling_window_width} timesteps long!" )
    for episode in episode_predictions:
        predictions = episode[0]
        true_label = episode[1]
        n_windows = len(predictions) - rolling_window_width + 1
        n_steps = 0
        for prediction in predictions:
            ans, ad_x, row = scan_output_for_decision(
                np.array([prediction]),
                np.array([tf.keras.utils.to_categorical(true_label, num_classes=2)]),
                threshold=0.9
            )
            n_steps += 1
            if ans != 'NC':
                break

        print(f'Window {n_steps}/{n_windows}: Prediction = [{row[0]:.2f}, {row[1]:.2f}] | True label = {true_label} | Answer = {ans}')

        # perf.count(ans)
        if ans == 'NC':
            if true_label == 0.0:
                perf.count('NCS')
            elif true_label == 1.0:
                perf.count('NCF')
        else:
            perf.count(ans)



        # Measure time performance
        # episode_len_s = n_steps * ts_s
        episode_len_s = (rolling_window_width + n_steps - 1) * ts_s
        # episode_len_s = (rolling_window_width + X_episodes.shape[0] - 1) * ts_s

        episode_obj = EpisodePerf()
        episode_obj.trueLabel = true_label
        episode_obj.answer = ans
        episode_obj.runTime = episode_len_s

        if true_label == 0.0:
            MTS += episode_len_s
            N_success += 1
            episode_obj.TTS = episode_len_s
        elif true_label == 1.0:
            MTF += episode_len_s
            N_failure += 1
            episode_obj.TTF = episode_len_s

        tAns_s = (ad_x + rolling_window_width - 1) * ts_s
        if ans in ( 'TP', 'FP' ):
            MTP += tAns_s
            episode_obj.TTP = tAns_s
            N_posCls += 1
        elif ans in ( 'TN', 'FN' ):
            MTN += tAns_s
            episode_obj.TTN = tAns_s
            N_negCls += 1

        timed_episodes.append(episode_obj)

    times = []
    for ep in timed_episodes:
        times.append(ep.runTime) 

    confMatx = {
        # Actual Positives
        'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
        'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
        # Actual Negatives
        'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
        'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
        'NC' : (perf['NCS'] + perf['NCF'] if ('NCS' in perf and 'NCF' in perf) else 0) / len(episode_predictions),
    }

    print()
    print( perf )
    print( confMatx )

    MTP /= N_posCls #- Mean Time to Positive Classification
    MTN /= N_negCls #- Mean Time to Negative Classification
    MTS /= N_success # Mean Time to Success
    MTF /= N_failure # Mean Time to Failure

    print()
    print( f"MTP: {MTP} [s]" )
    print( f"MTN: {MTN} [s]" )
    print( f"MTS: {MTS} [s]" )
    print( f"MTF: {MTF} [s]" )
    print( f"Success: {N_success}" )
    print( f"Failure: {N_failure}" )

    if verbose:
        print('MC BASELINE...')
    N   = 200000 # Number of trials
    MTS = 0.0 # Mean time to success

    for i in range(N):
        failing = 1
        TS_i    = 0.0
        while( failing ):
            
            # Draw episode
            ep = choice( timed_episodes )
            
            # If classifier decides Negative, then we only take TTN hit but commit to another attempt
            if ep.answer[1] == 'N':
                failing = True
                TS_i += ep.TTN
            # Else we always let the episode play out and task makes decision for us
            else:
                failing = not int( ep.trueLabel )
                TS_i += ep.runTime
        MTS += TS_i
        
    MTS /= N
    print( f"\n    Task Mean Time to Success: {MTS} [s]" )


    print('    Expected makespan:', end=' ')
    # EMS = expected_makespan( 
    #     MTF = MTF, 
    #     MTN = MTN, 
    #     MTS = MTS, 
    #     P_TP = confMatx['TP'] * (N_success/(N_success + N_failure)) , 
    #     P_FN = confMatx['FN'] * (N_success/(N_success + N_failure)) , 
    #     P_TN = confMatx['TN'] * (N_failure/(N_success + N_failure)) , 
    #     P_FP = confMatx['FP'] * (N_failure/(N_success + N_failure))  
    # )
    EMS = monitored_makespan(
        MTF = MTF, 
        MTN = MTN, 
        MTS = MTS, 
        P_TP = confMatx['TP'] * (N_success/(N_success + N_failure)), 
        P_FN = confMatx['FN'] * (N_success/(N_success + N_failure)), 
        P_TN = confMatx['TN'] * (N_failure/(N_success + N_failure)), 
        P_FP = confMatx['FP'] * (N_failure/(N_success + N_failure)),
        # P_NCS = perf['NCS'] / (confMatx['NC']),
        # P_NCF = perf['NCF'] / (confMatx['NC'])
        P_NCS = perf['NCS'] / len(episode_predictions),
        P_NCF = perf['NCF'] / len(episode_predictions)
    )
    print( EMS, end=' [s]' )
    metrics = {
        'EMS': [EMS],
        'MTP': [MTP],
        'MTN': [MTN],
        'MTS': [MTS],
        'MTF': [MTF],
        'P_TP': confMatx['TP'] * (N_success/(N_success + N_failure)),
        'P_FN': confMatx['FN'] * (N_success/(N_success + N_failure)),
        'P_TN': confMatx['TN'] * (N_failure/(N_success + N_failure)),
        'P_FP': confMatx['FP'] * (N_failure/(N_success + N_failure)),
        'P_NCS': perf['NCS'] / len(episode_predictions),
        'P_NCF': perf['NCF'] / len(episode_predictions)
    }
    return metrics, confMatx, times

# Plotting ----------------------------------------------------------------------------
def plot_mts_ems(res: dict, save_plots: bool = True):
    mts_time_reduction = 1 - (res['VanillaTransformer']['metrics']['MTS'][0] / res['FCN']['metrics']['MTS'][0])
    mts_textstr = ''.join(f'Mean Time to Success is decreased by {mts_time_reduction*100:.2f}%\nwith VanillaTransformer')

    ems_time_reduction = 1 - (res['VanillaTransformer']['metrics']['EMS'][0] / res['FCN']['metrics']['EMS'][0])
    ems_textstr = ''.join(f'Makespan is decreased by {ems_time_reduction*100:.2f}%\nwith VanillaTransformer')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)


    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    for model_name in res.keys():
        axes.bar([model_name], res[model_name]['metrics']['MTS'], label=model_name)

    axes.set_xlabel('Model')
    axes.set_ylabel('Mean time to Success [s]')
    axes.text(0.55, 0.75, mts_textstr, transform=axes.transAxes, alpha=0.5, bbox=props, fontsize=20)

    if save_plots:
        plt.savefig('../saved_data/imgs/makespan_prediction/mts_barplots.png')
        plt.clf()
    else:
        plt.plot()

    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    for model_name in res.keys():
        axes.bar([model_name], res[model_name]['metrics']['EMS'], alpha=0.5, label=model_name)

    # axes[1].legend()
    axes.set_xlabel('Model')
    axes.set_ylabel('Expected Makespan [s]')
    axes.text(0.55, 0.75, ems_textstr, transform=axes.transAxes, bbox=props, fontsize=20)


    if save_plots:
        plt.savefig('../saved_data/imgs/makespan_prediction/ems_barplots.png')
        plt.clf()
    else:
        plt.plot()


def plot_model_confusion_matrix(model_name: str, conf_mat: dict, save_plot: bool = True):
    arr = [
        [conf_mat['TP'], conf_mat['FP']],
        [conf_mat['FN'], conf_mat['TN']]
    ]

    sns.set(font_scale=2)
    if save_plot:
        nc_textstr = ''.join(f'{model_name} has a {conf_mat["NC"]*100:.2f}% NC rate')
        props = dict(boxstyle='round', facecolor='green', alpha=0.75)
        conf_mat_plot = sns.heatmap(arr, annot=True).get_figure()
        conf_mat_plot.text(0.43, 0.95, nc_textstr, bbox=props, fontsize=20, horizontalalignment='center', verticalalignment='top')
        conf_mat_plot.savefig(f'../saved_data/imgs/makespan_prediction/{model_name}_confusion_matrix.png')
        plt.clf()
    else:
        sns.heatmap(arr, annot=True)

    # NC percentage visualization through stacked plots
    x = ['No classification', 'Actual Positives', 'Actual negatives']
    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    if save_plot:
        nc = [conf_mat['NC'], 0, 0]
        tp = np.array([0, conf_mat['TP'], 0])
        fn = np.array([0, conf_mat['FN'], 0])
        fp = np.array([0, 0, conf_mat['FP']])
        tn = np.array([0, 0, conf_mat['TN']])
        axes.bar(x, nc, label='NC')
        axes.bar(x, tp, bottom=nc, label='TP')
        axes.bar(x, fp, bottom=nc+tp, label='FP')
        axes.bar(x, fn, bottom=nc+tp+fp, label='FN')
        axes.bar(x, tn, bottom=nc+tp+fp+fn, label='TN')
        axes.legend()
        plt.savefig(f'../saved_data/imgs/makespan_prediction/{model_name}_stacked_classification_barplots.png')
        plt.clf()
    else:
        plt.show()


def plot_runtimes(res: dict, save_plots: bool = True):
    ems_hits = np.sum([1 if vt < fcn else 0 for vt, fcn in zip(res['VanillaTransformer']['times'], res['FCN']['times'])])
    ems_performance = (ems_hits * 100) / len(res['VanillaTransformer']['times'])
    ems_textstr = ''.join(f'Runtime is lower for {ems_performance:.2f}% of the episodes with VanillaTransformer')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    axes.title.set_text('Runtimes for each model')
    runtimes = []
    for model_name in res.keys():
        runtimes.append(res[model_name]['times'])
    
    axes.hist(runtimes, alpha=0.5, label=list(res.keys()), bins=10)
    axes.legend()
    axes.set_xlabel('Runtime [s]')
    axes.set_ylabel('Count')
    axes.text(0.2, 0.75, ems_textstr, transform=axes.transAxes, bbox=props, fontsize=20)

    if save_plots:
        plt.savefig('../saved_data/imgs/makespan_prediction/runtime_histogram.png')
        plt.clf()
    else:
        plt.plot()


def plot_simulation_makespans(res: dict, models: list, save_plots: bool = True):
    textstr_dict = dict.fromkeys(models, None)
    for model_name in models:
        if model_name != 'FCN':
            hits = np.sum([1 if vt < fcn else 0 for vt, fcn in zip(res[model_name]['makespan_sim_hist'], res['FCN']['makespan_sim_hist'])])
            performance = (hits * 100) / len(res[model_name]['makespan_sim_hist'])
            textstr_dict[model_name] = ''.join(f'Makespan is lower for {performance:.2f}% of the episodes with {model_name} ({res[model_name]["makespan_sim_avg"]:.2f} \u00B1 {res[model_name]["makespan_sim_std"]:.2f})')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig, axes = plt.subplots(1, 1, figsize=(20, 8))
    axes.title.set_text(f'Simulated makespans for each model (N = {len(res[list(res.keys())[0]]["makespan_sim_hist"])})')
    runtimes = []
    for model_name in models:
        runtimes.append(res[model_name]['makespan_sim_hist'])

    axes.hist(runtimes, alpha=0.5, label=models, bins=10)
    axes.legend()
    axes.set_xlabel('Makespan [s]')
    axes.set_ylabel('Count')
    x = 0.15
    y = 0.85
    for model_name, textstr in textstr_dict.items():
        if textstr is not None:
            axes.text(x, y, textstr, transform=axes.transAxes, bbox=props, fontsize=20)
            y -= 0.08

    if save_plots:
        plt.savefig('../saved_data/imgs/makespan_prediction/makespan_simulation_histogram.png')
        plt.clf()
    else:
        plt.plot()

