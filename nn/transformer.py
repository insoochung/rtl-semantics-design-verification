import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(dff, activation="relu"),
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class CrossModalLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, dff, rate=0.1):
    super(CrossModalLayer, self).__init__()
    self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
    # (batch_size, target_seq_len, d_model)
    attn, attn_weights = self.mha(
        query=x, value=enc_output, attention_mask=padding_mask,
        return_attention_scores=True)
    attn = self.dropout1(attn)

    # (batch_size, target_seq_len, d_model)
    out2 = self.layernorm1(attn + x)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    # (batch_size, target_seq_len, d_model)
    out3 = self.layernorm2(ffn_output + out2)

    return out3, attn_weights


class CrossModalModule(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, rate=0.1):
    super(CrossModalModule, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.dec_layers = [
        CrossModalLayer(
            d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, look_ahead_mask=None, padding_mask=None,
           return_attention_weights=False):
    attention_weights = {}

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = self.dropout(x)

    for i, dec_layer in enumerate(self.dec_layers):
      x, attn_res = dec_layer(
          x, enc_output, look_ahead_mask, padding_mask)

    attention_weights[f"decoder_layer{i+1}"] = attn_res

    # x.shape == (batch_size, target_seq_len, d_model)
    if return_attention_weights:
      return x, attention_weights
    return x
