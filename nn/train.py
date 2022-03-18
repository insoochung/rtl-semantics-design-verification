import os
import sys
import copy
import argparse
import pickle

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
  return train, valid, test


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


def _finalize_dataset(cp_indices, dataset, cp_idx_to_midx, cp_idx_to_mask,
                      type="train"):
  tp_vectors = []
  is_hits = []
  masks = []
  graphs = []
  coverpoints = []
  for idx in cp_indices:
    for row in range(dataset[idx]["tp_vectors"].shape[0]):
      gidx = cp_idx_to_midx[idx]
      tp_vectors.append(dataset[idx]["tp_vectors"][row].astype(np.float32))
      is_hits.append(dataset[idx]["is_hits"][row].astype(np.float32))
      masks.append(cp_idx_to_mask[idx].astype(np.bool))
      graphs.append(gidx)
      coverpoints.append(idx)
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
  print(f"Dataset '{type}' shapes: ")
  for k, v in ret.items():
    print(f"- {k}: {v.shape}")
  return ret


def finalize_dataset(graph_dir, tp_cov_dir,
                     hit_lower_bound=0.1, hit_upper_bound=0.9,
                     split_ratio=(0.7, 0.15, 0.15)):
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
  cp_splits = {}
  cp_splits["train"], cp_splits["valid"], cp_splits["test"] = \
      split_indices(cp_indices, split_ratio)

  dataset = {}
  for key in (["train", "valid", "test"]):
    dataset[key] = _finalize_dataset(cp_splits[key], cov_dataset,
                                     cp_idx_to_midx, cp_idx_to_mask,
                                     type=key)
    dataset[key] = convert_to_tf_dataset(dataset[key])
  print("Dataset finalized.")
  return dataset


def load_dataset(tf_data_dir):
  print(f"Loading data from {tf_data_dir}...")
  dataset = {}
  for k in ["train", "valid", "test"]:
    with open(os.path.join(tf_data_dir, f"{k}.element_spec"), "rb") as f:
      es = pickle.load(f)
    dataset[k] = tf.data.experimental.load(
        os.path.join(tf_data_dir, f"{k}.tfrecord"), es, compression="GZIP")
  print("Data loaded.")
  return dataset


def save_dataset(dataset, tf_data_dir):
  print(f"Saving data to {tf_data_dir}...")
  os.makedirs(tf_data_dir, exist_ok=True)
  for k, ds in dataset.items():
    tf.data.experimental.save(
        ds, os.path.join(tf_data_dir, f"{k}.tfrecord"), compression="GZIP")
    with open(os.path.join(tf_data_dir, f"{k}.element_spec"), "wb") as f:
      pickle.dump(ds.element_spec, f)  # Redundant for TF>=2.5
  print("Data saved.")


def get_d2v_model(graph_dir, n_hidden, n_gcn_layers, n_mlp_hidden, dropout):
  graph_handler = GraphHandler(output_dir=graph_dir)
  graphs = graph_handler.get_dataset()
  model = Design2VecBase(graphs, n_hidden=n_hidden, n_gcn_layers=n_gcn_layers,
                         n_mlp_hidden=n_mlp_hidden, dropout=dropout)
  return model


def train(graph_dir, tf_data_dir, model_dir, tp_cov_dir=None,
          n_hidden=32, n_gcn_layers=4, n_mlp_hidden=32, dropout=0.1,
          learning_rate=0.001, batch_size=32, split_ratio=(0.7, 0.15, 0.15),
          hit_lower_bound=0.1, hit_upper_bound=0.9, generate_data=False,
          shuffle_train=True):

  if generate_data:
    dataset = finalize_dataset(graph_dir, tp_cov_dir, hit_lower_bound,
                               hit_upper_bound, split_ratio)
    save_dataset(dataset, tf_data_dir)

  dataset = load_dataset(tf_data_dir)
  for k, ds in dataset.items():  # Add batch size to dataset
    dataset[k] = ds.batch(batch_size)
  if shuffle_train:  # Add shuffle to dataset
    dataset["train"] = dataset["train"].shuffle(buffer_size=1024)

  model = get_d2v_model(graph_dir, n_hidden,
                        n_gcn_layers, n_mlp_hidden, dropout)

  model.compile(loss="binary_crossentropy", metrics=["binary_accuracy"],
                optimizer="adam")
  model.fit(dataset["train"], epochs=10, validation_data=dataset["valid"])


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
  parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                      help="Ratio of the train, valid and test split in "
                           "comma-separated string format.")

  # NN related flags
  parser.add_argument("-md", "--model_dir", type=str, required=True,
                      help="Directory of the model.")
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
  parser.add_argument("--batch_size", type=int, default=64,
                      help="Batch size.")
  parser.add_argument("--shuffle_train", action="store_true", default=False,
                      help="Shuffle TF dataset")

  args = parser.parse_args()
  train(**vars(args))
