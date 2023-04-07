#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/module/for_MTS/embedding.py
import math
import tensorflow as tf
# import tensorflow_lattice as tfl
import numpy as np


class GTN_Embedding(tf.keras.layers.Layer):
    def __init__(self,
                 d_feature: int,
                 d_timestep: int,
                 d_model: int,
                 wise: str = 'timestep' or 'feature',
                 trainable: bool = True,
                 name: str = None,
                 dtype: str = None,
                 dynamic: bool = False,
                 **kwargs):
        super(GTN_Embedding, self).__init__(trainable, name, dtype, dynamic, **kwargs)

        assert wise == 'timestep' or wise == 'feature', 'ERROR: embedding wise parameter'
        self.wise = wise

        # TODO: why is the embedding al reves del wise??
        if self.wise == 'timestep':
            # self.embedding = tfl.layers.Linear(num_input_dims=d_feature, units=d_model)
            self.embedding = tf.keras.layers.Dense(d_model, input_shape=(d_feature,), activation='relu')
        elif self.wise == 'feature':
            # self.embedding = tfl.layers.Linear(num_input_dims=d_timestep, units=d_model)
            self.embedding = tf.keras.layers.Dense(d_model, input_shape=(d_timestep,), activation='relu')


    def call(self,
                x: tf.Tensor):
        if self.wise == 'feature':
            x = self.embedding(x)
        elif self.wise == 'timestep':
            x = self.embedding(tf.transpose(x))
            # x = self.embedding(x)
            x = position_encode(x)

        return tf.convert_to_tensor(x)



def position_encode(x):
    # pe = tf.ones_like(x)
    # # position = tf.expand_dims(tf.convert_to_tensor(tf.range(0., x.shape[1]), dtype='float32'), axis=-1)
    # position = tf.expand_dims(tf.range(0., x.shape[1]), axis=-1)
    # # temp = tf.convert_to_tensor(tf.range(0., x.shape[-1], delta=2.), dtype='float32')
    # temp = tf.range(0., x.shape[-1], delta=2.)
    # temp = tf.multiply(temp, -(tf.divide(math.log(10000), x.shape[-1])))
    # temp = tf.expand_dims(tf.math.exp(temp), axis=0)
    # temp = tf.linalg.matmul(position, temp)  # shape:[input, d_model/2]
    # pe = tf.Variable(pe)
    # pe[:, 0::2].assign(tf.math.sin(temp))
    # pe[:, 1::2].assign(tf.math.cos(temp))
    # pe = tf.convert_to_tensor(pe)

    aux = np.zeros(x.shape)
    mat = np.arange(x.shape[1], dtype=np.float32).reshape(
        -1,1)/np.power(10000, np.arange(
        0, x.shape[-1], 2, dtype=np.float32) / x.shape[-1])
    aux[:, :, 0::2] = np.sin(mat)
    aux[:, :, 1::2] = np.cos(mat)
    pe = tf.convert_to_tensor(aux, dtype=tf.float32)

    return tf.keras.layers.Add()([x, pe])