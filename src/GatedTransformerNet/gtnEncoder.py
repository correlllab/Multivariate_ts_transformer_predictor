#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/encoder.py
import tensorflow as tf
import tensorflow_lattice as tfl

from gtnMultiHeadAttention import GTN_MultiHeadAttention
from gtnFeedForward import GTN_FeedForward


class GTN_Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 q: int,
                 v: int,
                 h: int,
                 d_model: int,
                 d_hidden: int,
                 dropout: float = 0.1,
                 trainable: bool = True,
                 name: str = None,
                 dtype: str = None,
                 dynamic: bool = False,
                 **kwargs):
        super(GTN_Encoder, self).__init__(trainable, name, dtype, dynamic, **kwargs)

        self.mha = GTN_MultiHeadAttention(d_model=d_model, q=q, v=v, h=h)
        self.feedforward = GTN_FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


    def call(self,
             x: tf.Tensor,
             stage: str = 'train' or 'test'):
        residual = x
        x, heatmap_score = self.mha(x, stage)
        x = self.dropout(x)
        x = self.layer_norm(self.add([x, residual]))
        x = self.feedforward(x)
        return x, heatmap_score