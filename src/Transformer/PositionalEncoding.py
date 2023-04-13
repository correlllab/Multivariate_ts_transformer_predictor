import numpy as np
import tensorflow as tf
# from YamlLoader import YamlLoader

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# def positional_encoding(length, depth):
#     depth = depth/2

#     positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

#     angle_rates = 1 / (10000**depths)         # (1, depth)
#     angle_rads = positions * angle_rates      # (pos, depth)

#     pos_encoding = np.concatenate(
#         # TODO: np.sin np.cos seem to not be in np 1.18.5
#         [np.sin(angle_rads), np.cos(angle_rads)],
#         axis=-1
#     ) 

#     return tf.cast(pos_encoding, dtype=tf.float32)

# def positional_encoding(x):
#     pe = tf.ones_like(x, dtype=tf.float32)
#     position = tf.expand_dims(tf.range(0., x.shape[0]), axis=-1)
#     temp = tf.range(0., x.shape[-1], delta=2.)
#     temp = tf.multiply(temp, -(tf.divide(math.log(10000), x.shape[-1])))
#     temp = tf.expand_dims(tf.math.exp(temp), axis=0)
#     temp = tf.linalg.matmul(position, temp)  # shape:[input, d_model/2]
#     pe = tf.Variable(pe, dtype=tf.float32)
#     pe[:, 0::2].assign(tf.math.sin(temp))
#     pe[:, 1::2].assign(tf.math.cos(temp))
#     pe = tf.convert_to_tensor(pe, dtype=tf.float32)
    
#     return x + pe


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # self.d_model = d_model
        # self.embedding = tf.keras.layers.Embedding(input_dim=space_size, output_dim=d_model, mask_zero=True)
        # self.pos_encoding = positional_encoding()


    def positional_encoding(self, s):
        # x = np.zeros(s)
        # pe = tf.ones_like(x, dtype=tf.float32)
        # position = tf.expand_dims(tf.range(0., x.shape[0]), axis=-1)
        # temp = tf.range(0., x.shape[-1], delta=2.)
        # temp = tf.multiply(temp, -(tf.divide(math.log(10000), x.shape[-1])))
        # temp = tf.expand_dims(tf.math.exp(temp), axis=0)
        # temp = tf.linalg.matmul(position, temp)  # shape:[input, d_model/2]
        # pe = tf.Variable(pe, dtype=tf.float32, validate_shape=False)
        # pe[:, 0::2].assign(tf.math.sin(temp))
        # pe[:, 1::2].assign(tf.math.cos(temp))
        # pe = tf.convert_to_tensor(pe, dtype=tf.float32)

        # return pe

        aux = np.zeros(s)
        mat = np.arange(s[0], dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, s[-1], 2, dtype=np.float32) / s[-1])
        aux[:, 0::2] = np.sin(mat)
        aux[:, 1::2] = np.cos(mat)
        pe = tf.convert_to_tensor(aux, dtype=tf.float16)

        # return tf.keras.layers.Add()([x, pe])
        return pe


    # def compute_mask(self, *args, **kwargs):
    #     return self.embedding.compute_mask(*args, **kwargs)


    def call(self, x):
        # print(f'In PositionalEmbedding call, shape = {x.shape}')
        # length = tf.shape(x)[1]
        # x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding(x.shape[-2:])
        return x