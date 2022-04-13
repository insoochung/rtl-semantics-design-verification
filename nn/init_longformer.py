import os
import sys
import argparse

import h5py
import tensorflow as tf
import numpy as np

from transformers import AutoConfig
from transformers import TFAutoModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn.train import set_model_flags

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  set_model_flags(parser, set_required=False)
  params = vars(parser.parse_args())
  pretrain_dir = params["pretrain_dir"]
  model_id = params["huggingface_model_id"]
  attention_window = params["attention_window"]
  max_pos = params["max_n_nodes"] + 2
  num_attention_heads = params["num_attention_heads"]
  n_att_hidden = params["n_att_hidden"]
  n_att_layers = params["n_att_layers"]

  # Initialize and save model
  config = AutoConfig.from_pretrained(model_id)
  config.gradient_checkpointing = True
  config.attention_window = attention_window
  config.max_position_embeddings = max_pos
  config.hidden_size = n_att_hidden
  config.intermediate_size = n_att_hidden * 4
  config.num_attention_heads = num_attention_heads
  config.num_hidden_layers = n_att_layers
  init_model = TFAutoModel.from_config(config)
  # Run fake input to build the init_model
  init_model(tf.keras.Input(shape=[None], dtype=tf.int32))
  embed_name = init_model.longformer.embeddings.position_embeddings.name
  print(
      f"Embedding before: {init_model.longformer.embeddings.position_embeddings}")
  init_model.save_pretrained(pretrain_dir)
  tf.keras.backend.clear_session()

  # Load parent to get the embeddings
  parent = TFAutoModel.from_pretrained(model_id)
  # Run fake input to build the parent model
  parent(tf.keras.Input(shape=[None], dtype=tf.int32))
  parent_pos_emb = parent.longformer.embeddings.position_embeddings
  parent_pos_emb = parent_pos_emb[:max_pos].numpy()
  tf.keras.backend.clear_session()

  pool_step_size = parent_pos_emb.shape[-1] / n_att_hidden
  assert pool_step_size - int(pool_step_size) == 0, (
      f"Attention model's hidden size should be divisible by the parent's "
      f"hidden size. Got {n_att_hidden} and {parent_pos_emb.shape[-1]}.")
  pool_step_size = int(pool_step_size)
  new_embed = np.zeros((max_pos, n_att_hidden))
  for i in range(max_pos):
    for j in range(n_att_hidden):
      new_embed[i][j] = parent_pos_emb[i][j * pool_step_size]

  print(f"Embedding is being replaced with: {new_embed}")
  with h5py.File(os.path.join(pretrain_dir, "tf_model.h5"), "r+") as f:
    key = f"longformer/{embed_name}"
    f[key][:] = new_embed

  init_model = TFAutoModel.from_pretrained(pretrain_dir)
  print(
      f"Embedding after: {init_model.longformer.embeddings.position_embeddings}")
