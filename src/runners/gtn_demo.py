import sys, os, pickle, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# import tensorflow_lattice as tfl

from GatedTransformerNet.GatedTransformer import GatedTransformer
from GatedTransformerNet.gtnCustomSchedule import GTN_CustomSchedule
from data_management.data_preprocessing import DataPreprocessing

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
    dp.run(verbose=True)

    d_model = 6
    d_hidden = 512
    d_feature = 6
    d_timestep = 350
    num_heads = 4
    num_layers = 4
    mlp_units = [128, 256, 64]
    learning_rate = GTN_CustomSchedule(d_model)
    # learning_rate = 1e-4
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    print(f'CREATING GTN...', end='')
    gtn = GatedTransformer(
        d_model=d_model,
        d_hidden=d_hidden,
        d_feature=d_feature,
        d_timestep=d_timestep,
        num_layers=num_layers,
        num_heads=num_heads,
        class_num=2,
        mlp_units=mlp_units
    )
    print(f'DONE')

    print('COMPILING GTN...', end='')
    gtn.compile(
        loss='binary_focal_crossentropy',
        optimizer=optimizer,
        metrics=['categorical_accuracy']
    )
    print('DONE')


    X_train = dp.X_train_sampled
    Y_train = dp.Y_train_sampled
    X_test = dp.X_test
    Y_test = dp.Y_test
    epochs = 200
    batch_size = 16
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            start_from_epoch=epochs*0.2
        )
    ]
    history = gtn.fit(
        x=X_train,
        y=Y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data=(X_test, Y_test),
        steps_per_epoch=len(X_train) // batch_size,
        validation_steps=len(X_test) // batch_size
    )
    
    # print(f'CALLING GTN()')
    # output = gtn(dp.X_train_under)
    # print(output.shape)