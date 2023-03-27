import tensorflow as tf
# from YamlLoader import YamlLoader
# from MultiHeadAttention import MultiHeadAttention

# from https://www.tensorflow.org/text/tutorials/transformer#define_the_components
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: mha layer not in tf 2.3
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


# MultiHeadAttention layer in the decoder (joints otput of encoder and output of first MHA layer of decoder)
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Add & Norm layer with residual connections
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# MultiHeadAttention layer in the encoder
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        # print(f'In encoder call(), input shape = {x.shape}')
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Add & Norm layer with residual connections
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


# Masked MultiHeadAttention layer in the decoder (the mask prevents from predictions looking
# into the future, only considers past observations)
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)

        # Add & Norm layer with residual connection
        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x