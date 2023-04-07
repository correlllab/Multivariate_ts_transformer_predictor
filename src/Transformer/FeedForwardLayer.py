import tensorflow as tf
# from YamlLoader import YamlLoader
# from MultiHeadAttention import MultiHeadAttention

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# FeedForward net for both encoder and decoder
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            # tf.keras.layers.Conv1D(filters=ff_dim,
            #                        kernel_size=1,
            #                        activation='relu'),
            # tf.keras.layers.Dropout(dropout_rate),
            # tf.keras.layers.Conv1D(filters=d_model,
            #                        kernel_size=1)
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model, activation='relu')
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        # print(f'X shape = {x.shape}; self.seq(x) shape = {self.seq(x).shape}')
        seq_out = self.seq(x)
        x = self.add([x, self.layer_norm(seq_out)])
        # x = self.layer_norm(x)
        return x