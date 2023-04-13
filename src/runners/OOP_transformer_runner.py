import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

from random import choice
import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from model_builds.OOPTransformer import OOPTransformer

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


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
        print

    dp = DataPreprocessing(sampling='none', data='reactive')
    dp.run(save_data=False, verbose=True)

    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    transformer_net = OOPTransformer()

    num_layers = 8
    d_model = 6
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    mlp_units = [128, 256, 64]

    transformer_net.build(
        X_sample=dp.X_train_sampled[:32],
        num_layers=num_layers,
        d_model=d_model,
        dff=dff,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        mlp_units=mlp_units,
        save_model=True
    )

    transformer_net.compile()

    X_train = dp.X_train_sampled
    Y_train = dp.Y_train_sampled
    X_test = dp.X_test
    Y_test = dp.Y_test
    epochs = 200
    batch_size = 32
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            start_from_epoch=epochs * 0.2
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-4,
            verbose=2
        )
    ]
    transformer_net.fit(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        callbacks=callbacks,
        epochs=epochs,
        batch_size=batch_size,
        save_model=True
    )

    transformer_net.model.save_weights(filepath=transformer_net.file_name)

    with open(transformer_net.histories_path, 'wb') as file_pi:
        pickle.dump(transformer_net.history.history, file_pi)

    print(transformer_net.model.summary())



