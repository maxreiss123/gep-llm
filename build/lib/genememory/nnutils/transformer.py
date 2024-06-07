from abc import ABC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import genememory.nnutils.encoder as enc
from rotary_embedding_tensorflow import apply_rotary_emb, RotaryEmbedding

# https://keras.io/examples/generative/text_generation_with_miniature_gpt/
# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
# https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993 - lbfgs optimixation
# https://machinelearningmastery.com/the-transformer-positional-encoding-layer-in-keras-part-2/ set pos embed sinusodial

loss_tracker = keras.metrics.Mean(name="loss")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.weight = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True,
            name="weight"
        )
        self.dim = dim
        self.eps = eps

    def _norm(self, x):
        mean_square = tf.math.reduce_mean(tf.math.square(x), axis=-1, keepdims=True)
        return x * tf.math.rsqrt(mean_square + self.eps)

    def call(self, inputs):
        normalized_input = self._norm(tf.cast(inputs, dtype=tf.float32))
        return self.weight * tf.cast(normalized_input, dtype=inputs.dtype)


class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SwiGLU, self).__init__(**kwargs)
        self.beta = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)

    def call(self, inputs):
        x, g = tf.split(inputs, 2, axis=-1)
        swish_gate = tf.sigmoid(tf.multiply(self.beta, g))
        return g * swish_gate * x


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1, seq_len=256):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_dim // num_heads)
        self.att2 = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_dim // num_heads)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.layernorm1 = RMSNorm(embed_dim)
        self.layernorm2 = RMSNorm(embed_dim)
        self.layernorm3 = RMSNorm(embed_dim)
        self.pos_emb = RotaryEmbedding(embed_dim // num_heads, learned_freq=False)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(embed_dim * 4, activation='linear'),
            SwiGLU(),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, inputs_, training=False):
        inputs, meta = inputs_
        seq_len = tf.shape(inputs)[1]
        freqs = self.pos_emb(tf.range(seq_len))[None, ...]
        q = k = self.layernorm1(inputs)

        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attention_output = self.dropout1(self.att(q, k, attention_mask=causal_mask),
                                         training=training)
        attention_meta = self.dropout1(self.att2(attention_output,
                                                 meta, meta))
        out1 = self.layernorm2(inputs + attention_output)
        out2 = self.layernorm3(out1 + attention_meta)
        ffn_output = self.dropout2(self.ffn(out2), training=training)
        return ffn_output + out2, meta

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'att2': self.att2,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'layernorm3': self.layernorm3,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'pos_emb': self.pos_emb
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class InvariantPointEncoder(layers.Layer):
    def __init__(self, embed_dimension):
        super(InvariantPointEncoder, self).__init__()
        self.t_net_layer = enc.create_point_encoder(embed_dim=embed_dimension)

    @tf.function
    def call(self, meta, training=False):
        x = self.t_net_layer(meta, training)
        return x

    def get_config(self):
        config = super(InvariantPointEncoder, self).get_config()
        config.update({
            't_net_layer': self.t_net_layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, mask_zero=True)
        self.padding = tf.ones((maxlen, embed_dim))
        self.max_len = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.case_embed = self.add_weight(
            shape=(self.embed_dim, self.embed_dim,),
            initializer="random_normal",
            trainable=True,
            name="case_embed")
        self.unit_embed = self.add_weight(
            shape=(7, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="case_embed")

    @tf.function
    def call(self, sequence, meta_info, ph_dimensions, training=False):
        features = sequence
        seq_len = tf.shape(sequence)[1]
        position_indices = tf.range(seq_len)
        pos_encoding = self.pos_emb(position_indices)
        unit = tf.einsum('kij,jl->kil', tf.cast(ph_dimensions, dtype=tf.float32
                                                ), self.unit_embed)
        meta = tf.matmul(tf.expand_dims(meta_info, 1) * self.padding, self.case_embed)
        seq = self.token_emb(features) + pos_encoding
        return seq + unit, meta

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'token_emb': self.token_emb,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'pos_embed': self.pos_emb
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(max_len=60, vocab_size=20000, embed_dim=256, num_heads=2, num_blocks=4,
                 size_of_cols=7):
    features = layers.Input(shape=(None,), dtype=tf.int8)
    meta = layers.Input(shape=(None, size_of_cols), dtype=tf.float32)
    dimension_sequences = layers.Input(shape=(None, 7), dtype=tf.int8)

    encoder = InvariantPointEncoder(embed_dimension=embed_dim)
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    blocks_m = [TransformerBlock(embed_dim, num_heads, 0.1, max_len) for _ in range(num_blocks)]

    x = encoder(meta)
    x = embedding_layer(features, x, dimension_sequences)
    for block in blocks_m:
        x = block(x)
    outputs = layers.Dense(vocab_size)(x[0])
    model = keras.Model(inputs=[features, meta, dimension_sequences], outputs=[outputs, x])
    optimizer = keras.optimizers.Adam(epsilon=1e-9, beta_1=0.9, beta_2=0.95, clipvalue=1)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  weighted_metrics=['accuracy'],
                  loss=[loss_fn, None])
    return model
