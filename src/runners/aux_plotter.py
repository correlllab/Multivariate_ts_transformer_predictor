import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utilities.metrics_plots import plot_acc_loss, compute_confusion_matrix, make_probabilities_plots


MODEL_NAME = 'OOP_Transformer_small'
MODEL_WEIGHTS_PATH = f'../saved_models/{MODEL_NAME}/'
HISTORY_PATH = f'../saved_data/histories/{MODEL_NAME}_history'
IMAGES_PATH = f'../saved_data/imgs/{MODEL_NAME}/'


class hist_class:
    def __init__(self, h) -> None:
        self.history = h


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

    dp = DataPreprocessing(sampling='none', data='reactive')
    dp.run()

    transformer = OOPTransformer(model_name=MODEL_NAME)
    num_layers = 4
    d_model = 6
    ff_dim = 256
    num_heads = 4
    head_size = 128
    dropout_rate = 0.25
    mlp_units = [128]
    transformer.build(
        X_sample=dp.X_train_sampled[:32],
        num_layers=num_layers,
        d_model=d_model,
        ff_dim=ff_dim,
        num_heads=num_heads,
        head_size=head_size,
        dropout_rate=dropout_rate,
        mlp_units=mlp_units,
        save_model=True,
        verbose=True
    )

    transformer.compile()
    transformer.model.load_weights(MODEL_WEIGHTS_PATH).expect_partial()

    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    hist_obj = hist_class(history)

    plot_acc_loss(history=hist_obj, imgs_path=IMAGES_PATH)
    compute_confusion_matrix(
        model=transformer.model,
        file_name=MODEL_WEIGHTS_PATH,
        imgs_path=IMAGES_PATH,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=transformer.model,
        model_name=transformer.model_name,
        imgs_path=IMAGES_PATH,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )
