#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/multiHeadAttention.py
import tensorflow as tf
import tensorflow_lattice as tfl


class GTN_MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 trainable: bool = True,
                 name: str = None,
                 dtype: str = None,
                 dynamic: bool = False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        # self.W_Q = tfl.layers.Linear(num_input_dims=d_model, units=q * h)
        # self.W_K = tfl.layers.Linear(num_input_dims=d_model, units=q * h)
        # self.W_V = tfl.layers.Linear(num_input_dims=d_model, units=v * h)
        # self.W_O = tfl.layers.Linear(num_input_dims=v * h, units=d_model)

        self.h = h

        self.W_Q = tf.keras.layers.Dense(q * h, input_shape=(d_model, ), activation=None)
        self.W_K = tf.keras.layers.Dense(q * h, input_shape=(d_model, ), activation=None)
        self.W_V = tf.keras.layers.Dense(v * h, input_shape=(d_model, ), activation=None)
        self.W_O = tf.keras.layers.Dense(d_model, input_shape=(v * h,), activation=None)

        self.inf = tf.math.add(tf.math.pow(-2., 32), 1.)


    def call(self,
                x: tf.Tensor,
                stage: str = 'train' or 'test'):
        # Q = tf.concat(self.W_Q(x).split(self.h, axis=-1), axis=0)
        # K = tf.concat(self.W_K(x).split(self.h, axis=-1), axis=0)
        # V = tf.concat(self.W_V(x).split(self.h, axis=-1), axis=0)

        Q = tf.concat(tf.split(value=self.W_Q(x), num_or_size_splits=self.h, axis=-1), axis=0)
        K = tf.concat(tf.split(value=self.W_K(x), num_or_size_splits=self.h, axis=-1), axis=0)
        V = tf.concat(tf.split(value=self.W_V(x), num_or_size_splits=self.h, axis=-1), axis=0)

        score = tf.linalg.matmul(Q, tf.transpose(K, perm=[0,2,1]))
        # score /= Q.shape[1]
        heatmap_score = score

        if stage == 'train':
            mask = tf.ones_like(score[0])
            mask = tf.linalg.band_part(mask, -1, 0)
            score = tf.where(mask > 0, score, (tf.multiply(tf.ones_like(mask), self.inf)))

        score = tf.nn.softmax(score, axis=-1)
        # weight_V = tf.concat(tf.linalg.matmul(score, V).split(self.h, axis=0), axis=-1)
        weight_V = tf.concat(tf.split(value=tf.linalg.matmul(score, V), num_or_size_splits=self.h, axis=0), axis=-1)
        out = self.W_O(weight_V)

        return out, heatmap_score