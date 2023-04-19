import sys, os, pickle, json
import numpy as np
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

from data_management.data_preprocessing import DataPreprocessing

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy


DATA = 'reactive'
DATA_DIR = f'../../data/data_manager/{DATA}'


# Define the Transformer model
class TransformerModel(tf.keras.Model):
    def __init__(self, num_outputs, num_layers, d_model, num_heads, dff, dropout_rate):
        super(TransformerModel, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
        self.positional_encoding = self.positional_encoding((350, d_model))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        # self.encoder_layers = [self.encoder_layer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.flatten = tf.keras.layers.Flatten()
        self.final_layer = tf.keras.layers.Dense(num_outputs, activation='softmax')

    def call(self, inputs, training):
        x = self.embedding(inputs)  # (batch_size, seq_length, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding
        x = self.dropout(x, training=training)
        x = tf.expand_dims(x, axis=0)
        for i in range(self.num_layers):
            print(x.shape)
            # x = self.encoder_layers[i](x, training)
            x = self.encoder_layer(x, self.d_model, self.num_heads, self.dff, self.dropout_rate)
        x = self.flatten(x)
        x = self.final_layer(x)
        return x

    def positional_encoding(self, s):
        aux = np.zeros(s)
        mat = np.arange(s[0], dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, s[-1], 2, dtype=np.float32) / s[-1])
        aux[:, 0::2] = np.sin(mat)
        aux[:, 1::2] = np.cos(mat)
        pe = tf.convert_to_tensor(aux, dtype=tf.float32)

        # return tf.keras.layers.Add()([x, pe])
        return pe

    def encoder_layer(self, inputs, d_model, num_heads, dff, dropout_rate):
        # inputs = tf.keras.Input(shape=(360, d_model))
        # inputs = tf.squeeze(inputs, axis=0)
        print(inputs.shape)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
        attention = tf.keras.layers.Dropout(rate=dropout_rate)(attention)
        attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        ffn_output = ffn(attention)
        ffn_output = tf.keras.layers.Dropout(rate=dropout_rate)(ffn_output)
        ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + ffn_output)
        return ffn_output
        # return tf.keras.Model(inputs=inputs, outputs=ffn_output)

# Define the training loop
def train(model, optimizer, loss_fn, train_data, train_labels, val_data, val_labels, num_epochs):
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
    for epoch in range(num_epochs):
        for x_train, y_train in zip(train_data, train_labels):
            with tf.GradientTape() as tape:
                logits = model(x_train, training=True)
                loss = loss_fn(y_train, logits[0])
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(y_train, logits)
        for x_val, y_val in zip(val_data, val_labels):
            logits = model(x_val, training=False)
            loss = loss_fn(y_val, logits[0])
            val_loss(loss)
            val_accuracy(y_val, logits)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, val_loss.result(), val_accuracy.result() * 100))
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()


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

    # print('\nLoading data from files...', end='')
    # with open(f'{DATA_DIR}/{DATA}_X_train_sampled.npy', 'rb') as f:
    #     X_train_sampled = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{DATA}_Y_train_sampled.npy', 'rb') as f:
    #     Y_train_sampled = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{DATA}_X_test.npy', 'rb') as f:
    #     X_test = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{DATA}_Y_test.npy', 'rb') as f:
    #     Y_test = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{DATA}_X_winTest.npy', 'rb') as f:
    #     X_winTest = np.load(f, allow_pickle=True)

    # with open(f'{DATA_DIR}/{DATA}_Y_winTest.npy', 'rb') as f:
    #     Y_winTest = np.load(f, allow_pickle=True)
    # roll_win_width = int(7.0 * 50)
    # print('DONE\n')

    dp = DataPreprocessing(sampling='none', data='reactive')
    dp.run(save_data=False, verbose=True)

    train_data = dp.X_train_sampled
    train_labels = dp.Y_train_sampled
    val_data = dp.X_test
    val_labels = dp.Y_test

    model = TransformerModel(num_outputs=2, num_layers=2, d_model=6, num_heads=2, dff=128, dropout_rate=0.2)

    # Compile the model
    optimizer = Adam(learning_rate=1e-3)
    loss_fn = CategoricalCrossentropy()
    accuracy_fn = CategoricalAccuracy()

    # Train the model using the train() function
    train(model, optimizer, loss_fn, train_data, train_labels, val_data, val_labels, num_epochs=100)
