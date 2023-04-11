import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from Transformer.Transformer import Transformer
from utils.metrics_plots import plot_acc_loss, compute_confusion_matrix, make_probabilities_plots


MODEL_WEIGHTS_PATH = '../saved_models/OOP_transformer/'
HISTORY_PATH = '../saved_data/histories/OOP_transformer_history'
IMAGES_PATH = '../saved_data/imgs/oop_transformer/'


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

    num_layers = 8
    d_model = 6
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    mlp_units = [128, 256, 64]
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=dff,
        mlp_units=mlp_units,
        input_space_size=6,
        target_space_size=2,
        training=True,
        dropout_rate=dropout_rate,
        pos_encoding=True
    )
    # learning_rate = CustomSchedule()
    opt = tf.keras.optimizers.legacy.Adam(1e-4, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)
    transformer.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    transformer.load_weights(MODEL_WEIGHTS_PATH).expect_partial()

    with open(HISTORY_PATH, 'rb') as f:
        history = pickle.load(f)

    hist_obj = hist_class(history)

    plot_acc_loss(history=hist_obj, imgs_path=IMAGES_PATH)
    compute_confusion_matrix(
        model=transformer,
        file_name=MODEL_WEIGHTS_PATH,
        imgs_path=IMAGES_PATH,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=transformer,
        model_name=transformer.model_name,
        imgs_path=IMAGES_PATH,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )
