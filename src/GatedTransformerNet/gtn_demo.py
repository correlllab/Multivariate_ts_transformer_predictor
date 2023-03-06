import os, sys
sys.path.insert(1, os.path.realpath('..'))
print( sys.version )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow_lattice as tfl

from GatedTransformer import GatedTransformer
from gtnCustomSchedule import GTN_CustomSchedule
from fcnn_vs_transformer.data_preprocessing import DataPreprocessing

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

    dp = DataPreprocessing()
    dp.run(verbose=True)

    epochs = 100
    batch_size = 16
    d_model = 512
    d_hidden = 2048
    d_feature = 6
    d_timestep = 350
    q = 8
    v = 8
    h = 4
    num_heads = 4
    learning_rate = GTN_CustomSchedule(d_model)
    # learning_rate = 1e-4
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                                epsilon=1e-9)


    print(f'CREATING GTN...')
    gtn = GatedTransformer(
        d_model=d_model,
        d_hidden=d_hidden,
        d_feature=d_feature,
        d_timestep=d_timestep,
        q=q,
        v=v,
        h=h,
        num_heads=num_heads,
        class_num=2
    )
    print(f'DONE')

    print('COMPILING GTN...', end='')
    gtn.compile(
        loss='binary_focal_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy'],
        experimental_run_tf_function=False
    )
    print('DONE')

    history = gtn.fit(
        x=dp.X_train_under,
        y=dp.Y_train_under,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )
    
    # print(f'CALLING GTN()')
    # output = gtn(dp.X_train_under)
    # print(output.shape)