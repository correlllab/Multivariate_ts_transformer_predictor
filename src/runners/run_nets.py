import sys, os, pickle, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import numpy as np
import tensorflow as tf

from data_management.data_preprocessing import DataPreprocessing
from model_builds.FCN import FCN
from model_builds.RNN import RNN
from model_builds.VanillaTransformer import VanillaTransformer
from model_builds.OOPTransformer import OOPTransformer
from utils.metrics_plots import plot_acc_loss, compute_confusion_matrix, make_probabilities_plots


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

    load_data_from_file = False
    save_data = False

    dp = DataPreprocessing(sampling='none', data='reactive')
    if load_data_from_file:
        # with open('X_pos_encoded_data.npy', 'r') as f:
        #     X_pos_encoded_data = np.load(f, allow_pickle=True)

        # with open('train_sampled_data.npy', 'r') as f:
        #     train_sampled_data = np.load(f, allow_pickle=True)

        # with open('test_data.npy', 'r') as f:
        #     test_data = np.load(f, allow_pickle=True)

        with open('preprocessing_data/X_data.npy', 'rb') as f:
            X_data = np.load(f)

        with open('preprocessing_data/Y_data.npy', 'rb') as f:
            Y_data = np.load(f)

        with open('preprocessing_data/win_data.npy', 'rb') as f:
            win_data = np.load(f, allow_pickle=True)

        x_train_sampled = X_data[2]
        y_train_sampled = Y_data[0]
        x_test = X_data[3]
        y_test = Y_data[1]
        x_train_enc = X_data[0]
        x_test_enc = X_data[1]
        x_win_test = win_data[0]
        y_win_test = win_data[1]
    else:
        dp.run(save_data=save_data, verbose=True)

    model_n_params = {}

    # FCN --------------------------------------------------------
    fcn_net = FCN(rolling_window_width=dp.rollWinWidth)
    fcn_net.build()
    fcn_net.fit(
        X_train=dp.X_train_sampled,
        Y_train=dp.Y_train_sampled,
        X_test=dp.X_test,
        Y_test=dp.Y_test,
        trainWindows=dp.trainWindows,
        epochs=200,
        save_model=True
    )
    model_n_params['FCN'] = int(np.sum([np.prod(v.get_shape().as_list()) for v in fcn_net.model.trainable_variables]))
    plot_acc_loss(history=fcn_net.history, imgs_path=fcn_net.imgs_path)
    compute_confusion_matrix(
        model=fcn_net.model,
        file_name=fcn_net.file_name,
        imgs_path=fcn_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=fcn_net.model,
        model_name=fcn_net.model_name,
        imgs_path=fcn_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )

    tf.keras.backend.clear_session()
    # RNN --------------------------------------------------------
    rnn_net = RNN()
    rnn_net.fit(
        X_train=dp.X_train_sampled,
        Y_train=dp.Y_train_sampled,
        X_test=dp.X_test,
        Y_test=dp.Y_test,
        batch_size=64,
        epochs=200,
        save_model=True
    )
    model_n_params['RNN'] = int(np.sum([np.prod(v.get_shape().as_list()) for v in rnn_net.model.trainable_variables]))
    plot_acc_loss(history=rnn_net.history, imgs_path=rnn_net.imgs_path)
    compute_confusion_matrix(
        model=rnn_net.model,
        file_name=rnn_net.file_name,
        imgs_path=rnn_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=rnn_net.model,
        model_name=rnn_net.model_name,
        imgs_path=rnn_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )

    tf.keras.backend.clear_session()
    # VanillaTransformer --------------------------------------------------------
    vanilla_transformer_net = VanillaTransformer()
    vanilla_transformer_net.fit(
        X_train=dp.X_train_sampled,
        Y_train=dp.Y_train_sampled,
        X_test=dp.X_test,
        Y_test=dp.Y_test,
        trainWindows=dp.trainWindows,
        epochs=200,
        save_model=True
    )
    model_n_params['VanillaTransformer'] = int(np.sum([np.prod(v.get_shape().as_list()) for v in vanilla_transformer_net.model.trainable_variables]))
    plot_acc_loss(history=vanilla_transformer_net.history, imgs_path=vanilla_transformer_net.imgs_path)
    compute_confusion_matrix(
        model=vanilla_transformer_net.model,
        file_name=vanilla_transformer_net.file_name,
        imgs_path=vanilla_transformer_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=vanilla_transformer_net.model,
        model_name=vanilla_transformer_net.model_name,
        imgs_path=vanilla_transformer_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )

    print(f'\nAttention scores:\n{vanilla_transformer_net.last_attn_scores}\n')

    tf.keras.backend.clear_session()
    # OOP Transformer --------------------------------------------------------
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
            start_from_epoch=epochs*0.2
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

    model_n_params['OOP_transformer'] = int(np.sum([np.prod(v.get_shape().as_list()) for v in transformer_net.model.trainable_variables]))

    transformer_net.model.save_weights(filepath='../saved_models/OOP_transformer/')

    with open('../saved_data/histories/OOP_transformer_history', 'wb') as file_pi:
        pickle.dump(transformer_net.history.history, file_pi)

    print(transformer_net.model.summary())

    plot_acc_loss(history=transformer_net.history, imgs_path=transformer_net.imgs_path)
    compute_confusion_matrix(
        model=transformer_net.model,
        file_name=transformer_net.file_name,
        imgs_path=transformer_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest,
        plot=True
    )
    make_probabilities_plots(
        model=transformer_net.model,
        model_name=transformer_net.model_name,
        imgs_path=transformer_net.imgs_path,
        X_winTest=dp.X_winTest,
        Y_winTest=dp.Y_winTest
    )


    with open('../saved_data/model_sizes.json', 'w') as f:
        json.dump(model_n_params, f)

    # num_layers = 8
    # d_model = 6
    # dff = 512
    # num_heads = 8
    # dropout_rate = 0.1
    # mlp_units = [128, 256, 64]
    # transformer_net = Transformer(
    #     num_layers=num_layers,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     ff_dim=dff,
    #     mlp_units=mlp_units,
    #     input_space_size=6,
    #     target_space_size=2,
    #     training=True,
    #     dropout_rate=dropout_rate,
    #     pos_encoding=True
    # )

    # output = transformer_net(dp.X_train_sampled[:32])
    # print(output.shape)
    # attn_scores = transformer_net.encoder.enc_layers[-1].last_attn_scores
    # print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    # print(transformer_net.summary())

    # learning_rate = CustomSchedule()
    # opt = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                     epsilon=1e-9)
    
    # transformer_net.compile(
    #     loss=tf.keras.losses.CategoricalCrossentropy(),
    #     optimizer=opt,
    #     metrics=[tf.keras.metrics.CategoricalAccuracy()]
    # )

    # X_train = dp.X_train_sampled
    # Y_train = dp.Y_train_sampled
    # X_test = dp.X_test
    # Y_test = dp.Y_test
    # epochs = 200
    # batch_size = 32
    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=10,
    #         restore_best_weights=True,
    #         start_from_epoch=epochs*0.2
    #     )
    # ]

    # history = transformer_net.fit(
    #     x=X_train,
    #     y=Y_train,
    #     validation_split=0.1,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     callbacks=callbacks,
    #     validation_data = (X_test, Y_test),
    #     steps_per_epoch = len(X_train) // batch_size,
    #     validation_steps = len(X_test) // batch_size
    # )

    # model_n_params['OOP_transformer'] = int(np.sum([np.prod(v.get_shape().as_list()) for v in transformer_net.trainable_variables]))

    # transformer_net.save_weights(filepath='../saved_models/OOP_transformer/')

    # with open('../saved_data/histories/OOP_transformer_history', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    # print(transformer_net.summary())


    # with open('../saved_data/model_sizes.json', 'w') as f:
    #     json.dump(model_n_params, f)

