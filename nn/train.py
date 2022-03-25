import os
import sys
import copy
import argparse
import pickle
import glob
import json

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constructor import DesignGraph, Module
from cdfg.graph import Node, BranchNode, EndNode
from data.utils import NodeVocab, BranchVocab, TestParameterCoverageHandler
from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase


def split_indices(indices, split_ratio):
  """Split the dataset into training and validation sets."""
  if not isinstance(split_ratio, tuple):
    assert isinstance(split_ratio, str)
    split_ratio = split_ratio.replace("(", "").replace(")", "").split(",")
    split_ratio = tuple(map(float, split_ratio))

  indices = copy.deepcopy(indices)
  np.random.shuffle(indices)
  assert abs(sum(split_ratio) - 1.0) < 1e-7  # Make sure ratio sum to 1
  train_valid_boundary = int(len(indices) * split_ratio[0])
  valid_test_boundary = int(len(indices) * (split_ratio[0] + split_ratio[1]))
  train = indices[:train_valid_boundary]
  valid = indices[train_valid_boundary:valid_test_boundary]
  test = indices[valid_test_boundary:]
  # Ensure no overlap between train, valid and test
  for pair in ((train, valid), (valid, test), (test, train)):
    for idx in pair[0]:
      assert idx not in pair[1]
  return train, valid, test


def split_dataset(dataset, split_ratio):
  train, valid, test = split_indices(list(dataset.keys()), split_ratio)
  train_ds, valid_ds, test_ds = None, None, None
  if train:
    train_ds = tf.data.experimental.sample_from_datasets([
      dataset[i] for i in train])
  if valid:
    valid_ds = tf.data.experimental.sample_from_datasets([
      dataset[i] for i in valid])
  if test:
    test_ds = tf.data.experimental.sample_from_datasets([
      dataset[i] for i in test])
  splits = {"train": train, "valid": valid, "test": test}
  dataset = {"train": train_ds, "valid": valid_ds, "test": test_ds}
  print(f"Dataset split into: {splits}")
  return dataset, splits


def convert_to_tf_dataset(dataset, label_key="is_hits"):
  """Convert the dict of numpy dataset to tf.data.Dataset."""
  def ds_gen(dataset):
    for i in range(dataset["coverpoint"].shape[0]):
      yield ({k: dataset[k][i] for k in dataset.keys()},  # inputs
             dataset[label_key][i])  # label

  signature = {}
  for key in dataset:
    nd = len(dataset[key].shape)
    signature[key] = tf.TensorSpec(
        shape=dataset[key].shape[1:] if nd > 1 else (),
        dtype=dataset[key].dtype)
    if key == label_key:
      label_signature = signature[key]
  signature = (signature, label_signature)  # inputs, label
  return tf.data.Dataset.from_generator(
      lambda: ds_gen(dataset), output_signature=signature)


def combine_data_per_cp(cp_idx, dataset, cp_idx_to_midx, cp_idx_to_mask):
  tp_vectors = []
  is_hits = []
  masks = []
  graphs = []
  coverpoints = []
  for row in range(dataset[cp_idx]["tp_vectors"].shape[0]):
    gidx = cp_idx_to_midx[cp_idx]
    tp_vectors.append(dataset[cp_idx]["tp_vectors"][row].astype(np.float32))
    is_hits.append(dataset[cp_idx]["is_hits"][row].astype(np.float32))
    masks.append(cp_idx_to_mask[cp_idx].astype(np.bool))
    graphs.append(gidx)
    coverpoints.append(cp_idx)
  tp_vectors = np.vstack(tp_vectors)
  is_hits = np.vstack(is_hits)
  masks = np.vstack(masks)
  graphs = np.array(graphs)
  coverpoints = np.array(coverpoints)
  # Indices permutated to shuffle the dataset in unison
  p = np.random.permutation(len(coverpoints))
  ret = {
      "test_parameters": tp_vectors[p],
      "is_hits": is_hits[p],
      "coverpoint_mask": masks[p],
      "graph": graphs[p],
      "coverpoint": coverpoints[p],
  }
  print(f"Dataset for coverpoint '{cp_idx}' shapes: "
        f"{[f'{k}: {v.shape}' for k, v in ret.items()]}")
  return ret


def combine_data(graph_dir, tp_cov_dir,
                 hit_lower_bound=0.1, hit_upper_bound=0.9):
  print("Aggregating sporadic information to finalize dataset...")
  # Load dataset
  graph_handler = GraphHandler(output_dir=graph_dir)

  # Constant: graphs are given as a spektral dataset
  graphs = graph_handler.get_dataset()

  # Variable: test parameter coverage dataset and target (is_hit)
  bvocab = BranchVocab(  # Utility objects
      os.path.join(tp_cov_dir, "vocab.branches.yaml"))
  tp_cov = TestParameterCoverageHandler(
      filepath=os.path.join(tp_cov_dir, "dataset.tp_cov.npy"))
  cov_dataset = tp_cov.arrange_dataset_by_coverpoint()

  # Filter with respect to hit rate
  cov_dataset_filt = {}
  for k, v in cov_dataset.items():
    if hit_lower_bound <= v["hit_rate"] <= hit_upper_bound:
      cov_dataset_filt[k] = v
  cov_dataset = cov_dataset_filt
  print(f"# of coverpoints within hit rate range between "
        f"{hit_lower_bound} and {hit_upper_bound}: {len(cov_dataset)}")

  # Map coverpoint idx to graph idx
  cp_idx_to_midx = {}
  cp_idx_to_mask = {}
  for cp_idx in cov_dataset.keys():
    midx = bvocab.get_module_index(cp_idx)
    cp_idx_to_midx[cp_idx] = midx
    cp_idx_to_mask[cp_idx] = bvocab.get_mask(cp_idx, midx,
                                             mask_length=graphs[0].n_nodes)

  # Split dataset into train, valid and test
  cp_indices = list(cov_dataset.keys())
  dataset = {}
  for cp_index in cp_indices:
    dataset[cp_index] = combine_data_per_cp(cp_index, cov_dataset,
                                            cp_idx_to_midx, cp_idx_to_mask)
    dataset[cp_index] = convert_to_tf_dataset(dataset[cp_index])
  print("Dataset finalized.")
  return dataset


def load_dataset(tf_data_dir):
  print(f"Loading data from {tf_data_dir}...")
  with open(os.path.join(tf_data_dir, "shared.element_spec"), "rb") as f:
    es = pickle.load(f)
  dataset = {}
  for tfrecord_path in glob.glob(os.path.join(tf_data_dir, "*.tfrecord")):
    cp_num = int(os.path.basename(tfrecord_path).split(".")[1])
    dataset[cp_num] = tf.data.experimental.load(
        tfrecord_path, es, compression="GZIP")
  print("Data loaded.")
  return dataset


def save_dataset(dataset, tf_data_dir):
  print(f"Saving data to {tf_data_dir}...")
  os.makedirs(tf_data_dir, exist_ok=True)
  for k, ds in dataset.items():
    tf.data.experimental.save(
        ds, os.path.join(tf_data_dir, f"coverpoint.{k:05d}.tfrecord"),
        compression="GZIP")
  with open(os.path.join(tf_data_dir, "shared.element_spec"), "wb") as f:
    pickle.dump(ds.element_spec, f)  # Redundant for TF>=2.5
  print("Data saved.")


def get_d2v_model(graph_dir, n_hidden, n_gcn_layers, n_mlp_hidden, dropout):
  graph_handler = GraphHandler(output_dir=graph_dir)
  graphs = graph_handler.get_dataset()
  model = Design2VecBase(graphs, n_hidden=n_hidden, n_gcn_layers=n_gcn_layers,
                         n_mlp_hidden=n_mlp_hidden, dropout=dropout)
  return model


def train(model, dataset, ckpt_dir, ckpt_name="best.ckpt",
          epochs=10, learning_rate=None):
  if learning_rate:
    print(f"For now, learing rate (given: {learning_rate}) is ignored, "
          f"and the ReduceLROnPlateau scheme is used.")
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy"],
                optimizer="adam")
  # Setup callbacks
  ckpt_path = os.path.join(ckpt_dir, ckpt_name)
  callbacks = [tf.keras.callbacks.TensorBoard(os.path.join(ckpt_dir, "logs"))]
  if dataset["valid"]:
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                    patience=3, verbose=True)
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_format="tf", monitor="val_loss", save_best_only=True,
        verbose=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, verbose=True)
    callbacks += [reduce_lr, model_ckpt, early_stopping]
  else:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_format="tf", verbose=True))
  return model.fit(dataset["train"], epochs=epochs,
                   validation_data=dataset["valid"], callbacks=callbacks)


def evaluate(model, test_dataset, ckpt_dir, ckpt_name="best.ckpt"):
  model.load_weights(os.path.join(ckpt_dir, ckpt_name))
  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy"])
  return model.evaluate(test_dataset, return_dict=True)


def run(graph_dir, tf_data_dir, ckpt_dir=None, ckpt_name="best.ckpt",
        tp_cov_dir=None, n_hidden=32, n_gcn_layers=4, n_mlp_hidden=32,
        dropout=0.1, learning_rate=None, batch_size=32, epochs=50,
        split_ratio=(0.7, 0.15, 0.15), generate_data=False,
        hit_lower_bound=0.1, hit_upper_bound=0.9):
  if generate_data:
    dataset = combine_data(graph_dir, tp_cov_dir,
                           hit_lower_bound, hit_upper_bound)
    save_dataset(dataset, tf_data_dir)
  model_config = {"graph_dir": graph_dir, "n_hidden": n_hidden,
                  "n_gcn_layers": n_gcn_layers, "n_mlp_hidden": n_mlp_hidden,
                  "dropout": dropout}

  print(f"Model config: {model_config}")
  # Load dataset
  dataset = load_dataset(tf_data_dir)
  # Split dataset
  dataset, splits = split_dataset(dataset, split_ratio)
  for k, ds in dataset.items():
    if ds: dataset[k] = ds.batch(batch_size)

  # Train model
  model = get_d2v_model(**model_config)
  history = train(model, dataset, ckpt_dir, ckpt_name, epochs, learning_rate)
  meta = {"splits": splits, "model_config": model_config,
          "train": {"history": history.history, "params": history.params}}

  # Evaluate model
  if dataset["test"]:
    test_model = get_d2v_model(**model_config)
    result = evaluate(test_model, dataset["test"], ckpt_dir, ckpt_name)
    print(f"Test result: {result}")
    meta["result"] = result

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
  parser.add_argument("-td", "--tp_cov_dir", type=str,
                      help="Directory of the test parameter coverage dataset.")
  parser.add_argument("-fd", "--tf_data_dir", type=str, required=True,
                      help="Directory of the finalized TF dataset.")
  parser.add_argument("--generate_data", action="store_true", default=False,
                      help="Generate TF dataset from the given graph and "
                           "test parameter coverage.")
  parser.add_argument("--hit_lower_bound", type=float, default=0.1,
                      help="Lower bound of the hit rate.")
  parser.add_argument("--hit_upper_bound", type=float, default=0.9,
                      help="Upper bound of the hit rate.")
  parser.add_argument("--split_ratio", type=str, default="0.5,0.25,0.25",
                      help="Ratio of the train, valid and test split in "
                           "comma-separated string format.")

  # NN related flags
  parser.add_argument("-cd", "--ckpt_dir", type=str,
                      help="Directory to save the checkpoints.")
  parser.add_argument("--ckpt_name", type=str, default="model.ckpt",
                      help="Name of the checkpoint.")
  parser.add_argument("--n_hidden", type=int, default=32,
                      help="Number of hidden units.")
  parser.add_argument("--n_gcn_layers", type=int, default=4,
                      help="Number of GCN layers.")
  parser.add_argument("--n_mlp_hidden", type=int, default=32,
                      help="Number of hidden units in the MLP.")
  parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate.")
  parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate.")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train.")
  parser.add_argument("--seed", type=int, required=False, default=0,
                      help="Seed to set.")
  parser.add_argument("--append_seed_to_ckpt_dir", action="store_true",
                      default=False, help="Append the seed path/dir vars.")

  args = parser.parse_args()
  print(f"Received arguments: {args}")
  run_with_seed(**vars(args))
