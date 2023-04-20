import os, sys
sys.path.append(os.path.realpath('../Transformer'))

import tensorflow as tf

# from YamlLoader import YamlLoader
# from MultiHeadAttention import MultiHeadAttention
from Encoder import Encoder
from PositionalEncoding import PositionalEncoding
# from Decoder import Decoder

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# Full transformer
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, head_size, ff_dim, mlp_units, input_space_size,
                 target_space_size, training, dropout_rate=0.1, mlp_dropout=0.4):
        super().__init__()

        # Params
        self.model_name = 'OOP_Transformer'
        self.training = training

        # Layers
        self.embedding = tf.keras.layers.Dense(d_model, activation='relu')
        self.positional_encoding = PositionalEncoding()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               head_size=head_size,
                               ff_dim=ff_dim,
                               space_size=input_space_size,
                               dropout_rate=dropout_rate,
                               mlp_dropout=mlp_dropout)
        
        self.mlp = tf.keras.Sequential()
        for dim in mlp_units:
            self.mlp.add(tf.keras.layers.Dense(dim, activation='relu'))
            self.mlp.add(tf.keras.layers.Dropout(mlp_dropout))

        self.global_average_pooling = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')
        self.final_layer = tf.keras.layers.Dense(target_space_size, activation='softmax', dtype='float32', kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))

    def call(self, inputs):
        # Input embedding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        # Positional encoding
        x = self.positional_encoding(x)  # Shape `(batch_size, seq_len, d_model)`.

        x = self.dropout(x)
        x = self.encoder(x, self.training)  # (batch_size, context_len, d_model)

        # Global average pooling for temporal data
        x = self.global_average_pooling(x)

        # MLP net
        # x = self.mlp(x)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_space_size)

        # Return the final output and the attention weights.
        return logits
