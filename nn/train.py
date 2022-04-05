import os
import sys
import argparse
import json
import math

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase
from nn.datagen import load_dataset, split_dataset
from nn.utils import get_lr_schedule


def get_d2v_model(graph_dir, n_hidden, n_gnn_layers, n_mlp_hidden, dropout=0.1,
                  aggregate="mean", use_attention=False, n_lstm_hidden=None,
                  n_lstm_layers=None, **kwargs):
  graph_handler = GraphHandler(output_dir=graph_dir)
  graphs = graph_handler.get_dataset()
  model = Design2VecBase(graphs, n_hidden=n_hidden, n_gnn_layers=n_gnn_layers,
                         n_mlp_hidden=n_mlp_hidden,
                         dropout=dropout,
                         cov_point_aggregate=aggregate,
                         use_attention=use_attention,
                         n_lstm_hidden=n_lstm_hidden,
                         n_lstm_layers=n_lstm_layers)
  return model


def compile_model_for_training(
        model, lr, lr_scheme=None, decay_rate=0.90, decay_steps=500,
        warmup_steps=1000, optimizer=tf.keras.optimizers.Adam, **kwargs):
  lr_schedule = get_lr_schedule(
      lr, lr_scheme, decay_rate, decay_steps, warmup_steps)
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy", "AUC"],
                optimizer=optimizer(learning_rate=lr_schedule))


def train(model, dataset, ckpt_dir, ckpt_name="best.ckpt",
          epochs=10, early_stopping=True):
  # Given model should be compiled before being passed.
  # Setup callbacks
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  callbacks = [tf.keras.callbacks.TensorBoard(os.path.join(ckpt_dir, "logs"))]

  if dataset["valid"]:
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.5, patience=3, verbose=True)
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_format="tf", monitor="val_loss",
            save_best_only=True, verbose=True, save_weights_only=True))
  else:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_format="tf", verbose=True, save_weights_only=True))

  if early_stopping:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=True))

  return model.fit(dataset["train"], epochs=epochs,
                   validation_data=dataset["valid"], callbacks=callbacks)


def evaluate(model, test_dataset, ckpt_dir, ckpt_name="best.ckpt"):
  model.load_weights(os.path.join(ckpt_dir, ckpt_name))
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy", "AUC"])
  return model.evaluate(test_dataset, return_dict=True)


def run(graph_dir, tf_data_dir, ckpt_dir=None, ckpt_name="best.ckpt",
        n_hidden=32, n_gnn_layers=4, n_mlp_hidden=32, n_lstm_hidden=32,
        n_lstm_layers=2, dropout=0.1, lr=None, lr_scheme=None, batch_size=32,
        epochs=50, split_ratio=(0.7, 0.15, 0.15), aggregate="mean",
        use_attention=False, early_stopping=True):

  model_config = {
      "graph_dir": graph_dir, "tf_data_dir": tf_data_dir, "ckpt_dir": ckpt_dir,
      "ckpt_name": ckpt_name, "n_hidden": n_hidden, "n_gnn_layers": n_gnn_layers,
      "n_mlp_hidden": n_mlp_hidden or n_hidden, "epochs": epochs,
      "n_lstm_hidden": n_lstm_hidden or n_hidden, "dropout": dropout,
      "split_ratio": split_ratio, "aggregate": aggregate,
      "use_attention": use_attention, "early_stopping": early_stopping,
      "n_lstm_layers": n_lstm_layers,
  }
  print(f"Model config: {model_config}")

  # Load dataset
  dataset = load_dataset(tf_data_dir)
  # Split dataset
  dataset, splits = split_dataset(dataset, split_ratio)
  for k, ds in dataset.items():
    if ds:
      dataset[k] = ds.batch(batch_size)

  # Train model
  model = get_d2v_model(**model_config)
  compile_model_for_training(model, lr, lr_scheme)
  history = train(model, dataset, ckpt_dir, ckpt_name, epochs, early_stopping)

  # Evaluate model
  if dataset["test"]:
    test_model = get_d2v_model(**model_config)
    result = evaluate(test_model, dataset["test"], ckpt_dir, ckpt_name)
    print(f"Test result: {result}")

  meta = {"splits": splits, "model_config": model_config,
          "train": {"history": history.history, "params": history.params},
          "result": result}

  return meta


def run_with_seed(seed, *args, append_seed_to_ckpt_dir=False, **kwargs):
  np.random.seed(seed)
  tf.random.set_seed(seed)
  if append_seed_to_ckpt_dir:
    seed_str = f"seed-{seed}"
    kwargs["ckpt_dir"] = os.path.join(kwargs["ckpt_dir"], seed_str)
  m = run(*args, **kwargs)
  m["seed"] = seed
  print(f"Train info: {m}")
  with open(os.path.join(kwargs["ckpt_dir"], "meta.json"), "w") as f:
    f.write(json.dumps(m, indent=2, sort_keys=True,
                       default=lambda o: "<not serializable>"))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Data related flags
  parser.add_argument("-gd", "--graph_dir", type=str, required=True,
                      help="Directory of the graph dataset.")
  parser.add_argument("-fd", "--tf_data_dir", type=str, required=True,
                      help="Directory of the finalized TF dataset.")
  parser.add_argument("--split_ratio", type=str, default="0.5,0.25,0.25",
                      help="Ratio of the train, valid and test split in "
                           "comma-separated string format.")

  # NN related flags
  parser.add_argument("-cd", "--ckpt_dir", type=str, required=True,
                      help="Directory to save the checkpoints.")
  parser.add_argument("--ckpt_name", type=str, default="model.ckpt",
                      help="Name of the checkpoint.")
  parser.add_argument("--n_hidden", type=int, default=32,
                      help="Number of hidden units.")
  parser.add_argument("--n_gnn_layers", type=int, default=4,
                      help="Number of GNN layers.")
  parser.add_argument("--n_lstm_layers", type=int, default=1,
                      help="Number of LSTM layers.")
  parser.add_argument("--n_mlp_hidden", type=int, default=None,
                      help="Number of hidden units in the MLP.")
  parser.add_argument("--n_lstm_hidden", type=int, default=None,
                      help="Size of hidden dimension in the LSTM.")
  parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate.")
  parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
  parser.add_argument("--lr_scheme", type=str, default="linear_decay",
                      help="Learning rate scheme.")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--seed", type=int, required=False, default=0,
                      help="Seed to set.")
  parser.add_argument("--append_seed_to_ckpt_dir", action="store_true",
                      default=False, help="Append the seed path/dir vars.")
  parser.add_argument("--aggregate", type=str, default="mean",
                      help="How the CDFG reader will aggregate coverpoint "
                           "embedding")
  parser.add_argument("--use_attention", action="store_true", default=False,
                      help="Whether to use attention in the design reader.")
  parser.add_argument("--no_early_stopping", dest="early_stopping",
                      action="store_false", default=True,
                      help="Whether to use attention in the design reader.")

  args = parser.parse_args()
  print(f"Received arguments: {args}")
  run_with_seed(**vars(args))
