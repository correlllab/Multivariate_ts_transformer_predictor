# https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/module/for_TS/embedding.py
import math
import tensorflow as tf
import tensorflow_lattice as tfl


class Embedding(tf.Module):
    def __init__(self,
                 d_feature: int,
                 d_timestep: int,
                 d_model: int,
                 wise: str = 'timestep' or 'feature',
                 dropout: float = 0.2,
                 name: str = None):
        super(Embedding, self).__init__(name)

        self.wise = wise
        assert wise == 'timestep' or wise == 'feature', 'Embedding wise error!'
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='causal')
        self.conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='causal')
        self.conv3 = tf.keras.layers.Conv1D(filters=d_feature, kernel_size=3, padding='causal')

        if wise == 'timestep':
            self.linear = tfl.layers.Linear(num_input_dims=16, units=d_model)
        elif wise == 'feature':
            self.linear = tfl.layers.Linear(num_input_dims=128, units=d_model)

        self.relu = tf.keras.layers.ReLU(dropout)


    def __call__(self, x):
        x = tf.expand_dims(x, axis=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        expand_tensor = x

        if self.wise == 'timestep':
            x = self.linear(tf.transpose(x))
            x = position_encode(x)
        elif self.wise == 'feature':
            x = self.linear(x)

        return self.relu(x), expand_tensor


def position_encode(x):
    pe = tf.ones_like(x, dtype=tf.float32)
    position = tf.expand_dims(tf.range(0., x.shape[0]), axis=-1)
    temp = tf.range(0., x.shape[-1], delta=2.)
    temp = tf.multiply(temp, -(tf.divide(math.log(10000), x.shape[-1])))
    temp = tf.expand_dims(tf.math.exp(temp), axis=0)
    temp = tf.linalg.matmul(position, temp)  # shape:[input, d_model/2]
    pe = tf.Variable(pe, dtype=tf.float32)
    pe[:, 0::2].assign(tf.math.sin(temp))
    pe[:, 1::2].assign(tf.math.cos(temp))
    pe = tf.convert_to_tensor(pe, dtype=tf.float32)
    
    return x + pe

# def position_encode(x):

#     pe = tf.ones_like(x[0])
#     position = tf.range(0, x.shape[1]).unsqueeze(-1)
#     temp = tf.Tensor(range(0, x.shape[-1], 2))
#     temp = temp * -(math.log(10000) / x.shape[-1])
#     temp = tf.math.exp(temp).unsqueeze(0)
#     temp = tf.linalg.matmul(position.float(), temp)  # shape:[input, d_model/2]
#     pe[:, 0::2] = tf.math.sin(temp)
#     pe[:, 1::2] = tf.math.cos(temp)

#     return x + pe