#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/feedforward.py
import tensorflow as tf
# import tensorflow_lattice as tfl


class GTN_FeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 512,
                 dropout_rate: int = 0.1):
        super().__init__()

        # self.linear1 = tfl.layers.Linear(num_input_dims=d_model, units=d_hidden)
        # self.linear2 = tfl.layers.Linear(num_input_dims=d_hidden, units=d_model)
        # self.linear1 = tf.keras.layers.Dense(d_hidden, input_shape=(d_model,), activation='relu')
        # self.linear2 = tf.keras.layers.Dense(d_model, input_shape=(d_hidden,), activation='relu')
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(d_hidden, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model, activation='relu')
        ])
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


    def call(self, x: tf.Tensor):
        seq_out = self.seq(x)
        x = self.add([x, self.layer_norm(seq_out)])

        return x
