import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import kerastuner as kt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.cdfg_datagen import GraphHandler
from nn.datagen import load_dataset, split_dataset
from nn.train import (get_d2v_model, compile_model_for_training,
                      set_model_flags)

DEFAULT_SEARCH_RANGE = {  # Common hparams
    "n_hidden": [32, 64, 128, 256, 512],
    "n_mlp_hidden": [32, 64, 128, 256, 512],
    "n_lstm_hidden": [32, 64, 128, 256, 512],
    "n_gnn_layers": [2, 4, 8],
    "n_lstm_layers": [1, 2],
    "n_mlp_layers": [2, 4, 8],
    "lr": [0.0005, 0.001, 0.002],
    "decay_rate": [0.80, 0.90, 0.95],
    "warmup_steps": [1, 500, 1000, 2000, 4000],
    "dropout": [0.1, 0.2],
    "num_attention_heads": [1, 2, 4, 8],
    "n_att_hidden": [32, 64, 128, 256, 512],
    "n_att_layers": [1, 2, 4, 8],

}


def get_params(hp, params: dict):
  search_params = {}
  for key in params["search_keys"]:
    search_params[key] = hp.Choice(key, DEFAULT_SEARCH_RANGE[key])

  search_params.update(params)
  return search_params


def build_model(hp, params: dict):
  s_params = get_params(hp, params)
  model = get_d2v_model(s_params)
  compile_model_for_training(model, s_params)
  return model


def search(params):
  graph_handler = GraphHandler(output_dir=params["graph_dir"])
  graphs = graph_handler.get_dataset()
  params["graphs"] = graphs
  tuner = kt.Hyperband(
      lambda hp: build_model(hp, params),
      objective=params["objective"],
      max_epochs=params["max_epochs"],
      hyperband_iterations=params["hyperband_iterations"],
      directory=params["search_dir"],
      project_name=params["project_name"],
      overwrite=False)

  dataset = load_dataset(params["tf_data_dir"])
  dataset, splits = split_dataset(dataset, params["split_ratio"])
  print(f"Splits: {splits}")
  for k, ds in dataset.items():
    if ds:
      dataset[k] = ds.batch(params["batch_size"])
  tuner.search(dataset["train"],
               validation_data=dataset["valid"],
               epochs=params["max_epochs"],
               callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
  print(f"Best HParams: {tuner.get_best_hyperparameters(1)[0].get_config()}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  set_model_flags(parser)
  parser.add_argument("-sd", "--search_dir", type=str, required=True,
                      help="Directory to save the checkpoints from tuner.")
  parser.add_argument("--project_name", type=str, default="d2v",
                      help="Name of the project.")
  parser.add_argument("--objective", type=str, default="val_binary_accuracy",
                      help="Metric to optimize for.")
  parser.add_argument("--max_epochs", type=int, default=30,
                      help="Max epochs per search iteration.")
  parser.add_argument("--hyperband_iterations", type=int, default=5)
  parser.add_argument("--search_keys", type=str, default="",
                      help="Comma separated list of hparams keys to search "
                           "for.")
  args = parser.parse_args()
  print(f"Received arguments: {args}")
  params = vars(args)
  params["search_keys"] = params["search_keys"].split(",")
  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)

  search(params)
