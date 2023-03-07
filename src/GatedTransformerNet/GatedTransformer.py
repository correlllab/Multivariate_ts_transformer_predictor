#https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/5b75d17e94bc96c62a5504afc5e8331918dc1cc5/Gated_Transfomer_Network/module/for_MTS/transformer.py
import tensorflow as tf
import tensorflow_lattice as tfl

from gtnEncoder import GTN_Encoder
from gtnEmbedding import GTN_Embedding


class GatedTransformer(tf.keras.Model):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 d_feature: int,
                 d_timestep: int,
                 q: int,
                 v: int,
                 h: int,
                 num_heads: int,
                 class_num: int,
                 dropout: float = 0.2,
                 *args,
                 **kwargs):
        super(GatedTransformer, self).__init__(*args, **kwargs)

        self.num_heads = num_heads

        self.timestep_embedding = GTN_Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='timestep')
        self.feature_embedding = GTN_Embedding(d_feature=d_feature, d_timestep=d_timestep, d_model=d_model, wise='feature')

        self.timestep_encoder = GTN_Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, dropout=dropout)
        self.feature_encoder = GTN_Encoder(d_model=d_model, d_hidden=d_hidden, q=q, v=v, h=h, dropout=dropout)

        # self.gate = tfl.layers.Linear(num_input_dims=d_timestep * d_model + d_feature * d_model, units=2)
        self.gate = tf.keras.layers.Dense(2, input_shape=(d_timestep * d_model + d_feature * d_model,), activation=None)

        # self.linear_out = tfl.layers.Linear(num_input_dims=d_timestep * d_model + d_feature * d_model,
        #                                     units=class_num)
        self.linear_out = tf.keras.layers.Dense(class_num, input_shape=(d_timestep * d_model + d_feature * d_model,), activation=None)


    def call(self,
                x: tf.Tensor,
                stage: str = 'train' or 'test'):
        x = tf.keras.Input(shape=x.shape[1:], dtype=tf.float32, batch_size=16)
        x_timestep = self.timestep_embedding(x)
        x_feature = self.feature_embedding(x)

        for _ in range(self.num_heads):
            x_timestep, timestep_heatmap = self.timestep_encoder(x_timestep, stage=stage)
            x_timestep, feature_heatmap = self.timestep_encoder(x_feature, stage=stage)


        x_timestep = tf.reshape(x_timestep, shape=(x_timestep.shape[0], -1))
        x_feature = tf.reshape(x_feature, shape=(x_feature.shape[0], -1))


        gate = tf.nn.softmax(self.gate(tf.concat([x_timestep, x_feature], axis=-1)), axis=-1)
        gate_out = tf.concat([tf.multiply(x_timestep, gate[:, 0:1]), tf.multiply(x_feature, gate[:, 1:2])], axis=-1)
        out = self.linear_out(gate_out)
        return out