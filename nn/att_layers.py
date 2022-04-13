import os
import tensorflow as tf

# from transformers.models.longformer.modeling_tf_longformer import (
#     LongformerConfig, TFLongformerModel)

from transformers import AutoConfig
from transformers import TFAutoModel


def tile_token_for_batch(tok, batch_size):
  assert len(tok.shape) == 2, (
      f"Token shape must be [1, n_hidden], instead got {tok.shape}")
  tok = tf.expand_dims(tok, axis=0)
  tok = tf.tile(tok, [batch_size, 1, 1])
  return tok


def get_transformer(model_dir, model_id="allenai/longformer-base-4096",
                    n_hidden=768, n_layers=12, num_attention_heads=12,
                    attention_window=256, max_pos=4098, from_scratch=False):

  if from_scratch:
    print("Please use nn/init_longformer.py "
          "to get a starting point for a transformer.")
    raise NotImplementedError

  ignored_values = {'max_position_embeddings': max_pos,
                    'hidden_size': n_hidden,
                    'intermediate_size': n_hidden * 4,
                    'num_attention_heads': num_attention_heads,
                    'num_hidden_layers': n_layers, }
  print(f"These values are ignored {ignored_values}")
  print("Please use make sure that you are reading from correct pretrained "
        "model.")

  if os.path.isdir(model_dir):
    # If model_dir is a directory, load the model from there.
    print(f"Loading Longformer from {model_dir}...")
    config = AutoConfig.from_pretrained(model_dir)
    config.gradient_checkpointing = True
    config.attention_window = attention_window
    model = TFAutoModel.from_pretrained(model_dir, config=config)
  else:
    # If no model exists, download the model
    print(f"Downloading Longformer: {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    config.gradient_checkpointing = True
    config.attention_window = attention_window
    model = TFAutoModel.from_pretrained(model_id, config=config)
  print(f"Config used: {config}")
  model.save_pretrained(model_dir)

  return model


class AttentionModule(tf.keras.layers.Layer):
  def __init__(self, n_hidden, n_layers, max_n_nodes=4096,
               pretrain_dir="pretrain/longformer-base-4096", params=None):
    super().__init__()
    self.max_n_nodes = max_n_nodes
    self.att_block = get_transformer(
        pretrain_dir, model_id=params["huggingface_model_id"],
        max_pos=max_n_nodes + 2, n_hidden=n_hidden,
        n_layers=n_layers, attention_window=params["attention_window"],
        from_scratch=params["init_att_from_scratch"],
        num_attention_heads=params["num_attention_heads"])

    self.att_block.set_output_embeddings(lambda x: x)

    self.n_hidden = self.att_block.config.hidden_size
    self.one = tf.ones(shape=(1))
    self.zero = tf.zeros(shape=(1))
    var_init = tf.keras.initializers.glorot_uniform()
    self.cls = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True,
        name="cls_node")
    self.sep = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True,
        name="sep_node")
    self.mask = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True,
        name="mask_node")

  def call(self, inputs):
    _x_list = []
    batch_size = tf.shape(inputs[0])[0]
    mask_dtype = tf.int32
    sep = tile_token_for_batch(self.sep, batch_size)
    cls = tile_token_for_batch(self.cls, batch_size)
    sep = tf.cast(sep, dtype=inputs[0].dtype)
    cls = tf.cast(cls, dtype=inputs[0].dtype)
    ones_batch = tf.ones(shape=(batch_size, 1), dtype=mask_dtype)

    global_att_mask = []
    for x in inputs:
      _x_list.append(x)
      _x_list.append(sep)
      global_att_mask.append(tf.zeros(shape=tf.shape(x)[:2], dtype=mask_dtype))
      global_att_mask.append(ones_batch)

    x = tf.concat([cls] + _x_list, axis=1)
    global_att_mask = tf.concat([ones_batch] + global_att_mask, axis=1)
    query_len = tf.shape(inputs[-1])[1] + 1
    token_type_ids = tf.concat([
        tf.zeros(shape=(batch_size, tf.shape(x)[1] - query_len),
                 dtype=mask_dtype),
        tf.ones(shape=(batch_size, query_len), dtype=mask_dtype)],
        axis=1)
    x = self.att_block(input_ids=None, inputs_embeds=x,
                       token_type_ids=token_type_ids,
                       global_attention_mask=global_att_mask)
    x = tf.expand_dims(x.pooler_output, axis=1)
    return x
