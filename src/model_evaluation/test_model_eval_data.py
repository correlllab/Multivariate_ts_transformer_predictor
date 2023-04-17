import json
import os
import sys

sys.path.append(os.path.realpath('../'))
sys.path.append(os.path.realpath('../src'))
print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from model_builds.OOPTransformer import OOPTransformer
from utilities.helper_functions import scan_output_for_decision


MODE = 'load_data'    # modes: ['create_data', 'load_data']


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

    X_data, Y_data = None, None

    if MODE == 'create_data':
        dp = DataPreprocessing(sampling='none', data='preemptive')
        dp.run(verbose=True)

        X_data = np.concatenate((dp.X_train_sampled, dp.X_test), axis=0)
        Y_data = np.concatenate((dp.Y_train_sampled, dp.Y_test), axis=0)

        print('Creating X_data...', end='')
        with open('../../data/eval_data/X_data.npy', 'wb') as f:
            np.save(f, X_data)
        print('DONE')

        print('Creating Y_data...', end='')
        with open('../../data/eval_data/Y_data.npy', 'wb') as f:
            np.save(f, Y_data)
        print('DONE')
    elif MODE == 'load_data':
        with open('../../data/eval_data/X_data.npy', 'rb') as f:
            X_data = np.load(f)

        with open('../../data/eval_data/Y_data.npy', 'rb') as f:
            Y_data = np.load(f)

        positive = 0
        negative = 0
        for el in Y_data:
            if el[0] == 1:
                positive += 1
            elif el[0] == 0:
                negative +=1
        print(f'\nNumber of windows = {X_data.shape[0]}; Positive pctg = {(positive * 100) / X_data.shape[0]}%; Negative pctg = {(negative * 100) / X_data.shape[0]}%\n')

        # model_names = ['FCN']
        # model_names = ['RNN']
        # model_names = ['VanillaTransformer']
        model_names = ['OOP_Transformer_small']
        # model_names = ['OOP_Transformer']

        results = dict.fromkeys(
            model_names,
            {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'NC': 0}
        )

        # print(X_data.shape, Y_data.shape)

        for model_name in model_names:
            # Load model
            if model_name == 'OOP_Transformer':
                transformer = OOPTransformer()
                num_layers = 4
                d_model = 6
                ff_dim = 256
                num_heads = 8
                head_size = 256
                dropout_rate = 0.25
                mlp_dropout = 0.4
                mlp_units = [128, 256, 64]
                transformer.build(
                    X_sample=X_data[:32],
                    num_layers=num_layers,
                    d_model=d_model,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout_rate=dropout_rate,
                    mlp_dropout=mlp_dropout,
                    mlp_units=mlp_units,
                    save_model=True,
                    verbose=True
                )
                transformer.compile()
                model = transformer.model
                model.load_weights(f'../saved_models/{model_name}/').expect_partial()
            elif model_name == 'OOP_Transformer_small':
                transformer = OOPTransformer(model_name='OOP_Transformer_small')
                num_layers = 4
                d_model = 6
                ff_dim = 256
                num_heads = 4
                head_size = 128
                dropout_rate = 0.25
                mlp_dropout = 0.4
                mlp_units = [128]

                transformer.build(
                    X_sample=X_data[:32],
                    num_layers=num_layers,
                    d_model=d_model,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout_rate=dropout_rate,
                    mlp_dropout=mlp_dropout,
                    mlp_units=mlp_units,
                    save_model=True,
                    verbose=True
                )
                transformer.compile()
                model = transformer.model
                model.load_weights(f'../saved_models/{model_name}/').expect_partial()
            else:
                model = tf.keras.models.load_model(f'../saved_models/{model_name}.keras')

            print(f'\nEvaluating model {model_name}...')
            # print(f'{model_name}: {model(tf.expand_dims(X_data[0], axis=0))}')

            predictions = model.predict(X_data)

            for pred, label in zip(predictions, Y_data):
                result, _ = scan_output_for_decision(output=np.asarray([pred]), trueLabel=label, threshold=0.9)
                results[model_name][result] += 1

            print('DONE\n')

        with open(f'../saved_data/evaluation_results/{model_name}_eval.json', 'w') as f:
            json.dump(results, f)

        confusion_matrix = {
            # Actual Positives
            'TP': (results[model_name]['TP'] if ('TP' in results[model_name]) else 0) / ((results[model_name]['TP'] if ('TP' in results[model_name]) else 0) + (results[model_name]['FN'] if ('FN' in results[model_name]) else 0)),
            'FN': (results[model_name]['FN'] if ('FN' in results[model_name]) else 0) / ((results[model_name]['TP'] if ('TP' in results[model_name]) else 0) + (results[model_name]['FN'] if ('FN' in results[model_name]) else 0)),
            # Actual Negatives
            'TN': (results[model_name]['TN'] if ('TN' in results[model_name]) else 0) / ((results[model_name]['TN'] if ('TN' in results[model_name]) else 0) + (results[model_name]['FP'] if ('FP' in results[model_name]) else 0)),
            'FP': (results[model_name]['FP'] if ('FP' in results[model_name]) else 0) / ((results[model_name]['TN'] if ('TN' in results[model_name]) else 0) + (results[model_name]['FP'] if ('FP' in results[model_name]) else 0)),
            'NC': (results[model_name]['NC'] if ('NC' in results[model_name]) else 0) / X_data.shape[0],
        }

        with open(f'../saved_data/evaluation_results/{model_name}_conf_matrix.json', 'w') as f:
            json.dump(confusion_matrix, f)
    else:
        print('ERROR: wrong MODE definition')
