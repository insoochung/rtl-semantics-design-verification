import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(dff, activation="relu"),
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
    self.mha2 = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    # (batch_size, target_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(
        query=x, value=x, attention_mask=look_ahead_mask,
        return_attention_scores=True)
    attn1 = self.dropout1(attn1)
    out1 = self.layernorm1(attn1 + x)

    # (batch_size, target_seq_len, d_model)
    attn2, attn_weights_block2 = self.mha2(
        query=out1, value=enc_output, attention_mask=padding_mask,
        return_attention_scores=True)
    attn2 = self.dropout2(attn2)

    # (batch_size, target_seq_len, d_model)
    out2 = self.layernorm2(attn2 + out1)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output)
    # (batch_size, target_seq_len, d_model)
    out3 = self.layernorm3(ffn_output + out2)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None,
           return_attention_weights=False):
    attention_weights = {}

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x)

    for i, dec_layer in enumerate(self.dec_layers):
      x, block1, block2 = dec_layer(
          x, enc_output, look_ahead_mask, padding_mask)

    attention_weights[f"decoder_layer{i+1}_block1"] = block1
    attention_weights[f"decoder_layer{i+1}_block2"] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    if return_attention_weights:
      return x, attention_weights
    return x
