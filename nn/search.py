import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import kerastuner as kt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn.datagen import load_dataset, split_dataset
from nn.train import get_d2v_model, compile_model_for_training

DEFAULT_SEARCH_RANGE = { # Common hparams
  "n_hidden": [32, 64, 128, 256, 512],
  "n_mlp_hidden": [32, 64, 128, 256, 512],
  "n_lstm_hidden": [32, 64, 128, 256, 512],
  "n_gnn_layers": [2, 4, 8],
  "n_lstm_layers": [2, 4, 8],
  "lr": [0.0005, 0.001, 0.002],
  "decay_rate": [0.80, 0.90, 0.95],
  "warmup_steps": [1, 500, 1000, 2000, 4000],
  "dropout": [0.1, 0.2]
}
def get_params(hp, override_params: dict):
  aggregate = override_params["aggregate"]
  use_attention = override_params["use_attention"]
  params = {}
  for key, search_range in DEFAULT_SEARCH_RANGE.items():
    if key in override_params:
      continue # Overrided params will not be searched for
    if "lstm" in key and aggregate != "lstm":
      continue # Do not add LSTM hparams to the search if not required.
    params[key] = hp.Choice(key, search_range)

  params.update(override_params)
  return params


def build_model(hp, override_params: dict):
  params = get_params(hp, override_params)
  model = get_d2v_model(**params)
  compile_model_for_training(model, **params)
  return model

def search(graph_dir, tf_data_dir, search_dir, project_name, split_ratio,
           override_params={}, objective="val_binary_accuracy", max_epochs=15,
           hyperband_iterations=4, batch_size=256, **kwargs):
  tuner = kt.Hyperband(
    lambda hp: build_model(hp, override_params),
    objective=objective,
    max_epochs=max_epochs,
    hyperband_iterations=hyperband_iterations,
    directory=search_dir,
    project_name=project_name,
    overwrite=False)

  dataset = load_dataset(tf_data_dir)
  dataset, splits = split_dataset(dataset, ".6,.4,0.")
  print(f"Splits: {splits}")
  for k, ds in dataset.items():
    if ds:
      dataset[k] = ds.batch(batch_size)
  tuner.search(dataset["train"],
              validation_data=dataset["valid"],
              epochs=max_epochs,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
  print(f"Best HParams: {tuner.get_best_hyperparameters(1)[0]}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-gd", "--graph_dir", type=str, required=True,
                      help="Directory of the graph dataset.")
  parser.add_argument("-fd", "--tf_data_dir", type=str, required=True,
                      help="Directory of the finalized TF dataset.")
  parser.add_argument("--split_ratio", type=str, default="0.6,0.4,0.0",
                      help="Ratio of the train, valid and test split in "
                           "comma-separated string format.")
  parser.add_argument("-sd", "--search_dir", type=str, required=True,
                      help="Directory to save the checkpoints from tuner.")
  parser.add_argument("--project_name", type=str, default="d2v",
                      help="Name of the project.")
  parser.add_argument("--objective", type=str, default="val_binary_accuracy",
                      help="Metric to optimize for.")
  parser.add_argument("--seed", type=int, default=123, help="Random seed.")
  parser.add_argument("--max_epochs", type=int, default=15,
                      help="Max epochs per search iteration.")
  parser.add_argument("--hyperband_iterations", type=int, default=5)
  parser.add_argument("--batch_size", type=int, default=256,
                      help="Batch size.")
  parser.add_argument("--aggregate", type=str, default="mean",
                      help="How the CDFG reader will aggregate coverpoint "
                           "embedding")
  parser.add_argument("--use_attention", action="store_true", default=False,
                      help="Whether to use attention in the design reader.")

  hparams_keys = list(DEFAULT_SEARCH_RANGE)
  for key in hparams_keys:
    parser.add_argument(f"--{key}", default=None)
  args = parser.parse_args()

  print(f"Received arguments: {args}")
  np.random.seed(args.seed)
  tf.random.set_seed(args.seed)
  args_dict = vars(args)
  override_params = {}
  for key in hparams_keys + ["aggregate", "use_attention", "graph_dir"]:
    if args_dict[key] is not None:
      override_params[key] = args_dict[key]
  args_dict["override_params"] = override_params
  search(**args_dict)