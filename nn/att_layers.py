import logging
import os
import math
import copy
import random
import numpy as np
import tensorflow as tf

from transformers.models.longformer.modeling_tf_longformer import (
    TFLongformerSelfAttention, TFLongformerForMaskedLM, LongformerConfig,
    TFLongformerModel)


def clone_dense(orig_dense):
  config = orig_dense.get_config()
  config["kernel_initializer"] = tf.constant_initializer(
      orig_dense.kernel.numpy())
  config["bias_initializer"] = tf.constant_initializer(
      orig_dense.bias.numpy())
  cloned_dense = type(orig_dense).from_config(config)
  return cloned_dense


def tile_token_for_batch(tok, batch_size):
  assert len(tok.shape) == 2, (
      f"Token shape must be [1, n_hidden], instead got {tok.shape}")
  tok = tf.expand_dims(tok, axis=0)
  tok = tf.tile(tok, [batch_size, 1, 1])
  return tok


def get_longformer(model_dir, model_id="allenai/longformer-base-4096",
                   attention_window=128, max_pos=4098, from_scratch=False):

  config = LongformerConfig.from_pretrained(
      model_id, max_position_embeddings=max_pos,
      attention_window=attention_window,
      gradient_checkpointing=True)

  if from_scratch:
    # If from_scratch is True, create a new model from scratch.
    print("Creating Longformer from scratch...")
    model = TFLongformerModel(config=config)
  elif os.path.isdir(model_dir):
    # If model_dir is a directory, load the model from there.
    print(f"Loading Longformer from {model_dir}...")
    model = TFLongformerModel.from_pretrained(model_dir,
                                              config=config)
  else:
    # If no model exists, download the model
    print(f"Downloading Longformer: {model_id}")
    model = TFLongformerModel.from_pretrained(model_id,
                                              config=config)
  model.save_pretrained(model_dir)

  return model


# class TFRobertaLongForMaskedLM(TFRobertaForMaskedLM):
#   def __init__(self, config):
#     super().__init__(config)
#     for i, layer in enumerate(self.roberta.encoder.layer):
#       layer.self_attention = TFLongformerSelfAttention(config, layer_id=i)

#   # def call(
#   #         self,
#   #         input_ids=None,
#   #         attention_mask=None,
#   #         token_type_ids=None,
#   #         position_ids=None,
#   #         head_mask=None,
#   #         inputs_embeds=None,
#   #         encoder_hidden_states=None,
#   #         encoder_attention_mask=None,
#   #         past_key_values=None,
#   #         use_cache=None,
#   #         output_attentions=None,
#   #         output_hidden_states=None,
#   #         return_dict=None,
#   #         training=False):
#   #   assert inputs_embeds is not None, "inputs_embeds should be provided"


class AttentionModule(tf.keras.layers.Layer):
  def __init__(self, n_hidden, max_n_nodes=4096,
               pretrain_path="pretrain/longformer-base-4096", dtype=tf.float32, **kwargs):
    super().__init__()
    self.max_n_nodes = max_n_nodes
    self.att_block = get_longformer(pretrain_path)
    self.att_block.set_output_embeddings(lambda x: x)
    self.n_hidden = self.att_block.config.hidden_size
    self.one = tf.ones(shape=(1), dtype=dtype)
    self.zero = tf.zeros(shape=(1), dtype=dtype)
    var_init = tf.keras.initializers.glorot_uniform()
    self.cls = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True, dtype=dtype,
        name="cls_node")
    self.sep = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True, dtype=dtype,
        name="sep_node")
    self.mask = tf.Variable(
        var_init(shape=(1, self.n_hidden)), trainable=True, dtype=dtype,
        name="mask_node")

  def call(self, inputs):
    _x_list = []
    batch_size = tf.shape(inputs[0])[0]
    mask_dtype = tf.int32
    sep = tile_token_for_batch(self.sep, batch_size)
    cls = tile_token_for_batch(self.cls, batch_size)
    ones_batch = tf.ones(shape=(batch_size, 1), dtype=mask_dtype)

    global_att_mask = []
    for x in inputs:
      _x_list.append(x)
      _x_list.append(sep)
      global_att_mask.append(tf.zeros(shape=tf.shape(x)[:2], dtype=mask_dtype))
      global_att_mask.append(ones_batch)

    x = tf.concat([cls] + _x_list, axis=1)
    global_att_mask = tf.concat([ones_batch] + global_att_mask, axis=1)
    token_type_ids = tf.zeros(shape=tf.shape(x)[:2], dtype=mask_dtype)
    # print(f"before: {x.shape}")
    x = self.att_block(input_ids=None, inputs_embeds=x,
                       token_type_ids=token_type_ids,
                       global_attention_mask=global_att_mask)
    x = tf.expand_dims(x.pooler_output, axis=1)
    return x
