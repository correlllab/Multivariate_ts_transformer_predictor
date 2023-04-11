import sys, os
sys.path.append(os.path.realpath('../Transformer'))

#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/encoder.py
import tensorflow as tf
# import tensorflow_lattice as tfl

from gtnMultiHeadAttention import GTN_MultiHeadAttention
from gtnFeedForward import GTN_FeedForward

from Transformer.AttentionLayers import GlobalSelfAttention, CausalSelfAttention


class GTN_EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 ff_dim,
                 dropout_rate=0.1,
                 enc_type='timestep' or 'feature'
    ):
        super().__init__()

        if enc_type == 'timestep':
            self.attention = CausalSelfAttention(
                num_heads=num_heads,
                key_dim=d_model,
                dropout=dropout_rate
            )
        else:
            self.attention = GlobalSelfAttention(
                num_heads=num_heads,
                key_dim=d_model,
                dropout=dropout_rate
            )

        self.ffn = GTN_FeedForward(d_model=d_model, d_hidden=ff_dim)


    def call(self, x: tf.Tensor):
        x = self.attention(x)
        x = self.ffn(x)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.attention.last_attn_scores


class GTN_Encoder(tf.keras.layers.Layer):
    def __init__(self,
                *,
                num_layers,
                d_model,
                num_heads,
                ff_dim,
                dropout_rate=0.1,
                enc_type='timestep' or 'feature'
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [
            GTN_EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                enc_type=enc_type
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.last_attn_scores = None


    def call(self, x: tf.Tensor):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        self.last_attn_scores = self.enc_layers[-1].last_attn_scores

        return x



# class GTN_Encoder(tf.keras.layers.Layer):
#     def __init__(self,
#                  q: int,
#                  v: int,
#                  h: int,
#                  d_model: int,
#                  d_hidden: int,
#                  dropout: float = 0.1,
#                  trainable: bool = True,
#                  name: str = None,
#                  dtype: str = None,
#                  dynamic: bool = False):
#         super().__init__()

#         self.mha = GTN_MultiHeadAttention(d_model=d_model, q=q, v=v, h=h)
#         self.feedforward = GTN_FeedForward(d_model=d_model, d_hidden=d_hidden)
#         self.dropout = tf.keras.layers.Dropout(rate=dropout)
#         self.layer_norm = tf.keras.layers.LayerNormalization()
#         self.add = tf.keras.layers.Add()


#     def call(self,
#              x: tf.Tensor,
#              stage: str = 'train' or 'test'):
#         residual = x
#         x, heatmap_score = self.mha(x, stage)
#         x = self.dropout(x)
#         x = self.layer_norm(self.add([x, residual]))
#         x = self.feedforward(x)
#         return x, heatmap_score