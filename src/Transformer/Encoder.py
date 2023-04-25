import tensorflow as tf
# from YamlLoader import YamlLoader
# from MultiHeadAttention import MultiHeadAttention
from AttentionLayers import GlobalSelfAttention
from FeedForwardLayer import FeedForward
from PositionalEncoding import PositionalEncoding

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, head_size, ff_dim, dropout_rate=0.2, mlp_dropout=0.4):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            dropout_rate=dropout_rate,
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, ff_dim, dropout_rate, mlp_dropout)

    def call(self, x, training):
        x = self.self_attention(x, training)
        x = self.ffn(x)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.self_attention.last_attn_scores

        return x


# Full encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, head_size,
                ff_dim, dropout_rate=0.1, mlp_dropout=0.4):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_dim=d_model, output_dim=1, input_length=350)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                head_size=head_size,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                mlp_dropout=mlp_dropout)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.last_attn_scores = None

    def call(self, x, training):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)

        # print(f'==> In Encoder call, last x.shape = {x.shape}')

        self.last_attn_scores = self.enc_layers[-1].last_attn_scores

        return x  # Shape `(batch_size, seq_len, d_model)`.