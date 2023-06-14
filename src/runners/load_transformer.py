import os, sys
sys.path.append(os.path.realpath('../'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
import numpy as np

from model_builds.OOPTransformer import OOPTransformer


DATA = ['reactive', 'training']
DATA_DIR = f'../../data/instance_data/{"_".join(DATA)}'


def load_keras_weights(model_build: OOPTransformer, model_name: str, makespan_models: dict, verbose: bool = True):
    try:
        model_build.model.load_weights(f'../saved_models/{model_name}/').expect_partial()
        makespan_models[model_name] = model_build.model
        if verbose:
            print(f'--> Loaded {model_name}')
    except OSError as e:
        print(f'{e}: model weights {model_name} do not exist!')


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

    return transformer_net


def get_model(name: str, roll_win_width: int = 0, X_sample = None):
    if name == 'OOP_Transformer':
        return build_oop_transformer(X_sample=X_sample, model_type='big')
    elif name == 'OOP_Transformer_small':
        return build_oop_transformer(X_sample=X_sample, model_type='small')




if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print( f"Found {len(gpus)} GPUs!" )
    for i in range( len( gpus ) ):
        try:
            tf.config.experimental.set_memory_growth(device=gpus[i], enable=True)
            tf.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*3 )
            print( f"\t{tf.config.experimental.get_device_details( device=gpus[i] )}" )
        except RuntimeError as e:
            print( '\n', e, '\n' )

    devices = tf.config.list_physical_devices()
    print( "Tensorflow sees the following devices:" )
    for dev in devices:
        print( f"\t{dev}" )

    # Load data
    print('\nLoading data from files...', end='')
    with open(f'{DATA_DIR}/{"_".join(DATA)}_X_train.npy', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
    print('DONE')

    roll_win_width = int(7.0 * 50)

    models = {}

    # Load Small Transformer model
    transformer = get_model(name='OOP_Transformer_small', roll_win_width=roll_win_width, X_sample=X_train[:64])
    load_keras_weights(model_build=transformer, model_name='OOP_Transformer_small', makespan_models=models, verbose=True)