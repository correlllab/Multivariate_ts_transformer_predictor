import os, sys, json
print(sys.version)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from random import choice

from data_preprocessing import DataPreprocessing
from FCN import FCN
from VanillaTransformer import Transformer
from Transformer.CustomSchedule import CustomSchedule
from utils import CounterDict


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


def scan_output_for_decision( output, trueLabel, threshold = 0.95 ):
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
            return ans, i
    return 'NC', output.shape[0]


def expected_makespan( MTF, MTN, MTS, P_TP, P_FN, P_TN, P_FP ):
    """ Return the expected makespan for the given parameters """
    return -( MTF*P_FP + MTN*P_FN + MTN*P_TN + MTS*P_TP + 1) / (P_FN + P_FP + P_TN - 1)


def run_makespan_prediction_for_model(model: tensorflow.keras.Model, dp: DataPreprocessing, verbose=False):
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
    xList     = dp.X_winTest
    yList     = dp.Y_winTest
    winLen    = xList[0].shape[1]

    # print( xList[7].shape )

    if verbose:
        print( f"Each window is {winLen} timesteps long!" )

    timedEpisodes = []


    for epNo in range( len( xList ) ):    
        
        print( '>', end=' ' )
        
        # Bookkeepping #
        x_i   = xList[epNo]
        y_i   = yList[epNo]
        epObj = EpisodePerf()
        
        with tensorflow.device('/GPU:0'):
            
            # Measure Accuracy Performance #
            res      = model.predict( x_i, verbose=1 )
            if verbose:
                print(res.shape, y_i.shape)
            ans, aDx = scan_output_for_decision( res, y_i, threshold = 0.90 )
            perf.count( ans )
            
            # Measure Time Performance #
            epLen_s   = (winLen + x_i.shape[0] - 1)*ts_s
            trueLabel = y_i[0]
            
            epObj.trueLabel = trueLabel
            epObj.answer    = ans
            epObj.runTime   = epLen_s
            
            if trueLabel[0] == 1.0:
                MTS += epLen_s
                N_success += 1
                epObj.TTS = epLen_s
            elif trueLabel[0] == 0.0:
                MTF += epLen_s
                N_failure += 1
                epObj.TTF = epLen_s
            
            tAns_s = (aDx + winLen - 1)*ts_s
            # print( aDx, winLen, tAns_s )
            if ans in ( 'TP', 'FP' ):
                MTP += tAns_s
                epObj.TTP = tAns_s
                N_posCls += 1
            elif ans in ( 'TN', 'FN' ):
                MTN += tAns_s
                epObj.TTN = tAns_s
                N_negCls += 1
                
        timedEpisodes.append( epObj )
            
    confMatx = {
        # Actual Positives
        'TP' : (perf['TP'] if ('TP' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
        'FN' : (perf['FN'] if ('FN' in perf) else 0) / ((perf['TP'] if ('TP' in perf) else 0) + (perf['FN'] if ('FN' in perf) else 0)),
        # Actual Negatives
        'TN' : (perf['TN'] if ('TN' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
        'FP' : (perf['FP'] if ('FP' in perf) else 0) / ((perf['TN'] if ('TN' in perf) else 0) + (perf['FP'] if ('FP' in perf) else 0)),
        'NC' : (perf['NC'] if ('NC' in perf) else 0) / len( xList ),
    }

    if verbose:
        print()
        print( perf )
        print( confMatx )


    if N_posCls == 0:
        MTP = 'inf'
    else:
        MTP /= N_posCls #- Mean Time to Positive Classification

    if N_negCls == 0:
        MTN = 'inf'
    else:
        MTN /= N_negCls #- Mean Time to Negative Classification

    if N_success == 0:
        MTS = 'inf'
    else:
        MTS /= N_success # Mean Time to Success

    if N_failure == 0:
        MTF = 'inf'
    else:
        MTF /= N_failure # Mean Time to Failure

    if verbose:
        print()
        print( f"MTP: {MTP} [s]" )
        print( f"MTN: {MTN} [s]" )
        print( f"MTS: {MTS} [s]" )
        print( f"MTF: {MTF} [s]" )
        print( f"Success: {N_success}" )
        print( f"Failure: {N_failure}" )


    if verbose:
        print('MC BASELINE...')
    # TODO: decide which of the two is the right one and ask why N = 200000
    # N   = 200000 # Number of trials
    # MTS = 0.0 # Mean time to success

    # for i in range(N):
    #     failing = 1
    #     TS_i    = 0.0
    #     while( failing ):
    #         ep = choice( timedEpisodes )
    #         TS_i += ep.runTime
    #         failing = not int( ep.trueLabel )
    #     MTS += TS_i
        
    # MTS /= N
    # print( f"Task Mean Time to Success: {MTS} [s]" )

    N   = 200000 # Number of trials
    MTS = 0.0 # Mean time to success

    for i in range(N):
        failing = 1
        TS_i    = 0.0
        while( failing ):
            
            # Draw episode
            ep = choice( timedEpisodes )
            
            # If classifier decides Negative, then we only take TTN hit but commit to another attempt
            if ep.answer[1] == 'N':
                failing = True
                TS_i += ep.TTN
            # Else we always let the episode play out and task makes decision for us
            else:
                failing = not int( ep.trueLabel[0] )
                TS_i += ep.runTime
        MTS += TS_i
        
    MTS /= N
    print( f"\n    Task Mean Time to Success: {MTS} [s]" )


    print('    Expected makespan:', end=' ')
    EMS = expected_makespan( 
        MTF = MTF, 
        MTN = MTN, 
        MTS = MTS, 
        P_TP = confMatx['TP'] * (N_success/(N_success + N_failure)) , 
        P_FN = confMatx['FN'] * (N_success/(N_success + N_failure)) , 
        P_TN = confMatx['TN'] * (N_failure/(N_success + N_failure)) , 
        P_FP = confMatx['FP'] * (N_failure/(N_success + N_failure))  
    )
    print( EMS, end=' [s]' )
    return MTS, EMS


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
        print

    print('FETCHING DATA...')
    dp = DataPreprocessing(sampling='none')
    dp.run(verbose=True)

    makespan_models = {}

    compute = True
    save_dicts = True
    res = {}
    total = 50
    if compute:
        print('LOADING MODELS...')
        try:
            fcn = tensorflow.keras.models.load_model('models/FCN.keras')
            makespan_models['FCN'] = fcn
        except OSError as e:
            print(f'{e}: model does not exist!\n')

        try:
            vanilla_transformer = Transformer()
            vanilla_transformer.build_model(
                dp.X_train_sampled.shape[1:],
                head_size=256,
                num_heads=4,
                ff_dim=4,
                num_transformer_blocks=4,
                mlp_units=[128],
                mlp_dropout=0.4,
                dropout=0.25
            )
            vanilla_transformer.model.compile(
                loss="binary_focal_crossentropy",
                optimizer=tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                metrics=["categorical_accuracy"]
            )
            loss, accuracy_d = vanilla_transformer.model.evaluate(dp.X_test, dp.Y_test, verbose=2)
            print("Untrained model: accuracy = {:5.2f}%; loss = {:5.2f}".format(100 * accuracy_d, loss))
            vanilla_transformer.model.load_weights('./models/tmp/checkpoints/')
            loss, accuracy_d = vanilla_transformer.model.evaluate(dp.X_test, dp.Y_test, verbose=2)
            print("Restored model: accuracy = {:5.2f}%; loss = {:5.2f}".format(100 * accuracy_d, loss))
            vanilla_transformer.model.save('models/Transformer.keras')
            vanilla_transformer =  tensorflow.keras.models.load_model('models/Transformer.keras')
            makespan_models['VanillaTransformer'] = vanilla_transformer
        except OSError as e:
            print(f'{e}: model does not exist!\n')


        for i in range(total):
            dp = DataPreprocessing(sampling='none')
            dp.run(verbose=False)
            print(f'\nIteration {i}/{total} [{(i*100)/total}%]')
            for model_name, model in makespan_models.items():
                if model_name not in res.keys():
                    res[model_name] = {'MTS': [], 'EMS': []}
                print(f'====> For model {model_name}:')
                mts, ems = run_makespan_prediction_for_model(model=model, dp=dp, verbose=False)
                res[model_name]['MTS'].append(mts)
                res[model_name]['EMS'].append(ems)
                print()
        
        if save_dicts:
            with open('makespan_results.txt', 'w') as f:
                f.write(json.dumps(res))
    else:
        with open('makespan_results.txt', 'r') as f:
            res = json.loads(f.read())

    print(f'res = {res}\n')

    metrics = {k: {} for k in list(res.keys())}
    for model_name in res.keys():
        metrics[model_name]['MTS_mean'] = np.mean(res[model_name]['MTS'])
        metrics[model_name]['MTS_std'] = np.std(res[model_name]['MTS'])

        metrics[model_name]['EMS_mean'] = np.mean(res[model_name]['EMS'])
        metrics[model_name]['EMS_std'] = np.std(res[model_name]['EMS'])

    if save_dicts:
        with open('makespan_metrics.txt', 'w') as f:
                f.write(json.dumps(res))

    print(f'metrics = {metrics}\n')

    mts_hits = np.sum([1 if vt < fcn else 0 for vt, fcn in zip(res['VanillaTransformer']['MTS'], res['FCN']['MTS'])])
    mts_performance = (mts_hits * 100) / total
    mts_textstr = ''.join(f'Predicted Mean Time to Success is lower {mts_performance}% of the time with VanillaTransformer')

    ems_hits = np.sum([1 if vt < fcn else 0 for vt, fcn in zip(res['VanillaTransformer']['EMS'], res['FCN']['EMS'])])
    ems_performance = (ems_hits * 100) / total
    ems_textstr = ''.join(f'Predicted Makespan is lower {ems_performance}% of the time with VanillaTransformer')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].title.set_text('Task Mean Time To Success for each model')
    axes[1].title.set_text('Expected Makespan for each model')
    for model_name in res.keys():
        axes[0].plot(res[model_name]['MTS'], label=model_name)
        axes[1].plot(res[model_name]['EMS'], label=model_name)

    axes[0].legend()
    axes[0].set_xlabel('Prediction index')
    axes[0].set_ylabel('Mean time to Success [s]')
    axes[0].text(0.02, 0.75, mts_textstr, transform=axes[0].transAxes, bbox=props)
    axes[1].legend()
    axes[1].set_xlabel('Prediction index')
    axes[1].set_ylabel('Makespan prediction [s]')
    axes[1].text(1.25, 0.75, ems_textstr, transform=axes[0].transAxes, bbox=props)

    plt.savefig('imgs/makespan_prediction/mts_ems_lineplots.png')
    plt.clf()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].title.set_text('Task Mean Time To Success for each model')
    axes[1].title.set_text('Expected Makespan for each model')
    mts = []
    ems = []
    for model_name in res.keys():
        mts.append(res[model_name]['MTS'])
        ems.append(res[model_name]['EMS'])

    axes[0].hist(mts, alpha=0.5, label=list(res.keys()), bins=10)
    axes[1].hist(ems, alpha=0.5, label=list(res.keys()), bins=10)

    axes[0].legend()
    axes[0].set_xlabel('Mean time to success prediction [s]')
    axes[0].set_ylabel('Count')
    axes[0].text(0.2, 0.75, mts_textstr, transform=axes[0].transAxes, bbox=props)
    axes[1].legend()
    axes[1].set_xlabel('Makespan prediction [s]')
    axes[1].set_ylabel('Count')
    axes[1].text(1.5, 0.75, ems_textstr, transform=axes[0].transAxes, bbox=props)

    plt.savefig('imgs/makespan_prediction/mts_ems_histograms.png')
    plt.clf()
