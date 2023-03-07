#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/feedforward.py
import tensorflow as tf
import tensorflow_lattice as tfl


class GTN_FeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 2048,
                 trainable: bool = True,
                 name: str = None,
                 dtype: str = None,
                 dynamic: bool = False,
                 **kwargs):
        super(GTN_FeedForward, self).__init__(trainable, name, dtype, dynamic, **kwargs)

        # self.linear1 = tfl.layers.Linear(num_input_dims=d_model, units=d_hidden)
        # self.linear2 = tfl.layers.Linear(num_input_dims=d_hidden, units=d_model)
        self.linear1 = tf.keras.layers.Dense(d_hidden, input_shape=(d_model,), activation='relu')
        self.linear2 = tf.keras.layers.Dense(d_model, input_shape=(d_hidden,), activation='relu')
        self.relu = tf.keras.layers.ReLU()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


    def call(self,
                x: tf.Tensor):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.layer_norm(self.add([x, residual]))
        return x