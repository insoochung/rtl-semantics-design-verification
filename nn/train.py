import os
import sys
import argparse
import json

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase
from nn.datagen import load_dataset, split_dataset, get_fake_inputs
from nn.utils import get_lr_schedule, warmstart_pretrained_weights


def get_d2v_model(params):
  model = Design2VecBase(params)
  return model


def get_optimizer(params):
  lr_schedule = get_lr_schedule(
      params["lr"], params["lr_scheme"], params["decay_rate"],
      params["decay_steps"], params["warmup_steps"])
  optimizer = tf.keras.optimizers.get(params["optimizer"]).__class__
  optimizer = optimizer(learning_rate=lr_schedule)
  return optimizer


def compile_model_for_training(model, params):
  optimizer = get_optimizer(params)
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy", "AUC"],
                optimizer=optimizer)


def train(model, dataset, params):
  # Setup callbacks
  ckpt_dir = params["ckpt_dir"]
  ckpt_name = params["ckpt_name"]
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  callbacks = [tf.keras.callbacks.TensorBoard(os.path.join(ckpt_dir, "logs"))]

  if dataset["valid"]:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_format="tf", monitor="val_loss",
            save_best_only=True, verbose=True))
  else:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_format="tf", verbose=True))

  if params["early_stopping"]:
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=True))

  return model.fit(dataset["train"], epochs=params["epochs"],
                   validation_data=dataset["valid"], callbacks=callbacks)


def evaluate(model, test_dataset, params):
  ckpt_dir = params["ckpt_dir"]
  ckpt_name = params["ckpt_name"]
  model(get_fake_inputs(params["tf_data_dir"]))
  model.load_weights(os.path.join(ckpt_dir, ckpt_name))
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy", "AUC"])
  return model.evaluate(test_dataset, return_dict=True)


def maybe_init_or_load_model(model, params):
  ckpt_dir = params["ckpt_dir"]
  ckpt_name = params["ckpt_name"]
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  if os.path.exists(ckpt_path):
    model.load(ckpt_path)
  elif params["warmstart_dir"]:
    fake_inputs = get_fake_inputs(params["tf_data_dir"])
    model(fake_inputs)  # Init model to be able to load weights
    warmstart_path = os.path.join(
        params["warmstart_dir"], params["ckpt_name"], "cdfg_reader")
    assert os.path.exists(os.path.exists(os.path.dirname(warmstart_path))), (
        f"{warmstart_path} not found")
    warmstart_pretrained_weights(model, warmstart_path)

  return model


def run_training(params):
  tf_data_dir = params["tf_data_dir"]
  split_ratio = params["split_ratio"]
  batch_size = params["batch_size"]
  # Load dataset
  dataset = load_dataset(tf_data_dir)
  # Split dataset
  dataset, splits = split_dataset(dataset, split_ratio)
  for k, ds in dataset.items():
    if ds:
      dataset[k] = ds.batch(batch_size)

  # Train model
  graph_handler = GraphHandler(output_dir=params["graph_dir"])
  graphs = graph_handler.get_dataset()
  params["graphs"] = graphs
  model = get_d2v_model(params)
  compile_model_for_training(model, params)
  model = maybe_init_or_load_model(model, params)
  history = train(model, dataset, params)

  # Evaluate model
  if dataset["test"]:
    test_model = get_d2v_model(params)
    result = evaluate(test_model, dataset["test"], params)
    print(f"Test result: {result}")

  meta = {"splits": splits, "model_config": params,
          "train": {"history": history.history, "params": history.params},
          "result": result}

  return meta


def run_with_seed(params, run_fn=run_training):
  seed = params["seed"]
  np.random.seed(seed)
  tf.random.set_seed(seed)
  update_params_default(params)
  m = run_fn(params)
  m["seed"] = seed
  print(f"Train info: {m}")
  with open(os.path.join(params["ckpt_dir"], "meta.json"), "w") as f:
    f.write(json.dumps(m, indent=2, sort_keys=True,
                       default=lambda o: "<not serializable>"))


def update_params_default(params):
  seed = params["seed"]
  if params["append_seed_to_ckpt_dir"]:
    seed_str = f"seed-{seed}"
    params["ckpt_dir"] = os.path.join(params["ckpt_dir"], seed_str)
  params["n_mlp_hidden"] = params["n_mlp_hidden"] or params["n_hidden"]
  params["n_lstm_hidden"] = params["n_lstm_hidden"] or params["n_hidden"]
  params["pretrain_dir"] = params["pretrain_dir"] or os.path.join(
      params["ckpt_dir"], "pretrain")
  return params


def set_model_flags(parser, set_required=False):
  # Data related flags
  required = set_required
  parser.add_argument("-gd", "--graph_dir", type=str, required=required,
                      help="Directory of the graph dataset.")
  parser.add_argument("-fd", "--tf_data_dir", type=str, required=required,
                      help="Directory of the finalized TF dataset.")
  parser.add_argument("--split_ratio", type=str, default="0.5,0.25,0.25",
                      help="Ratio of the train, valid and test split in "
                           "comma-separated string format.")

  # NN related flags
  parser.add_argument("-cd", "--ckpt_dir", type=str, required=required,
                      help="Directory to save the checkpoints.")
  parser.add_argument("-wd", "--warmstart_dir", type=str, default=None,
                      help="Directory to fetch the initial weights.")
  parser.add_argument("-pd", "--pretrain_dir", type=str, default=None,
                      help="Directory to save the pretraind models.")
  parser.add_argument("--ckpt_name", type=str, default="model.ckpt",
                      help="Name of the checkpoint.")
  parser.add_argument("--n_hidden", type=int, default=32,
                      help="Number of hidden units.")
  parser.add_argument("--n_labels", type=int, default=1,
                      help="Number of labels in final task target.")
  parser.add_argument("--n_gnn_layers", type=int, default=8,
                      help="Number of GNN layers.")
  parser.add_argument("--n_lstm_layers", type=int, default=1,
                      help="Number of LSTM layers.")
  parser.add_argument("--n_mlp_hidden", type=int, default=512,
                      help="Number of hidden units in the MLP.")
  parser.add_argument("--n_mlp_layers", type=int, default=2,
                      help="Number of hidden layers in the MLP.")
  parser.add_argument("--n_att_hidden", type=int, default=768,
                      help="Number of hidden units in the Longformer.")
  parser.add_argument("--n_att_layers", type=int, default=16,
                      help="Number of hidden layers in the Longformer.")
  parser.add_argument("--n_lstm_hidden", type=int, default=64,
                      help="Size of hidden dimension in the LSTM.")
  parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate.")
  parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
  parser.add_argument("--lr_scheme", type=str, default="warmup_then_decay",
                      help="Learning rate scheme.")
  parser.add_argument("--decay_rate", type=float, default=0.9,
                      help="Learning rate decay rate.")
  parser.add_argument("--decay_steps", type=int, default=4000,
                      help="Learning rate decay steps.")
  parser.add_argument("--warmup_steps", type=int, default=8000,
                      help="Learning rate warm up steps.")
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--seed", type=int, default=123,
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
  parser.add_argument("--optimizer", type=str, default="adam",
                      help="Which optimizer to use")
  parser.add_argument("--max_n_nodes", type=int, default=4096,
                      help="Max number of nodes that CDFG reader should "
                           "process.")
  parser.add_argument("--use_custom_attention_hparams",
                      action="store_true", default=False,
                      help="Whether to use custom hparams for attention "
                           "module, if False, use pretrained model from "
                           "huggingface hub.")
  parser.add_argument("--attention_window", type=int, default=256,
                      help="Window size for Longformer attention.")
  parser.add_argument("--num_attention_heads", type=int, default=12,
                      help="Number of attention heads in Longformer layers.")
  parser.add_argument("--huggingface_model_id", type=str,
                      default="allenai/longformer-base-4096",
                      help="Id of the pretrained Longformer model.")
  parser.add_argument("--use_att_encoder",
                      action="store_true", default=False)
  parser.add_argument("--no_att_encoder", dest="use_att_encoder",
                      action="store_false", default=False)
  parser.add_argument("--use_att_decoder",
                      action="store_true", default=True)
  parser.add_argument("--no_att_decoder", dest="use_att_decoder",
                      action="store_false", default=True)
  parser.add_argument("--freeze_att_encoder",
                      action="store_true", default=False)
  parser.add_argument("--design_feedback_type",
                      type=str, default="cdfg_reader")
  parser.add_argument("--n_max_coverpoints", type=int, default=2000)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  set_model_flags(parser)
  args = parser.parse_args()
  print(f"Received arguments: {args}")
  run_with_seed(vars(args))
