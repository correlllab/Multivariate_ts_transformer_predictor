import os, sys
sys.path.insert(1, os.path.realpath('../Transformer'))
print(sys.version)
print(sys.path)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from random import choice
import tensorflow as tf

from data_preprocessing import DataPreprocessing
from FCN import FCN
# from VanillaTransformer import Transformer
from Transformer.CustomSchedule import CustomSchedule
from utils import CounterDict
from Transformer.Transformer import Transformer
from Transformer.CustomSchedule import CustomSchedule

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


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

    dp = DataPreprocessing(sampling='none')
    dp.run(save_data=False, verbose=True)

    num_layers = 8
    d_model = 6
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    vanilla_transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        ff_dim=dff,
        input_space_size=6,
        target_space_size=2,
        dropout_rate=dropout_rate,
        pos_encoding=True
    )
    output = vanilla_transformer(dp.X_train_sampled[:100])
    print(output.shape)
    attn_scores = vanilla_transformer.encoder.enc_layers[-1].last_attn_scores
    print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    print(vanilla_transformer.summary())

    vanilla_transformer.save_weights(filepath='./models/OOP_transformer/')

    # vanilla_transformer = Transformer(
    #     num_layers=num_layers,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     ff_dim=dff,
    #     input_space_size=6,
    #     target_space_size=2,
    #     dropout_rate=dropout_rate,
    #     pos_encoding=True
    # )
    learning_rate = CustomSchedule()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)
    # vanilla_transformer.compile(
    #     loss="binary_focal_crossentropy",
    #     optimizer=opt,
    #     metrics=["categorical_accuracy"]
    # )
    vanilla_transformer.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=opt,
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

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
            start_from_epoch=epochs*0.2
        )
    ]

    vanilla_transformer.fit(
        x=X_train,
        y=Y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        validation_data = (X_test, Y_test),
        steps_per_epoch = len(X_train) // batch_size,
        validation_steps = len(X_test) // batch_size
    )

    vanilla_transformer.save_weights(filepath='./models/OOP_transformer/')

    print(vanilla_transformer.summary())



