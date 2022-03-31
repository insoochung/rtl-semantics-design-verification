import os
import sys
import argparse
import pickle
import glob
import copy

import numpy as np
import tensorflow as tf
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constructor import DesignGraph, Module
from cdfg.graph import Node, BranchNode, EndNode
from data.utils import NodeVocab, BranchVocab, TestParameterCoverageHandler
from data.cdfg_datagen import GraphHandler


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


def combine_data_per_cp(cp_idx, dataset, cp_idx_to_midx, cp_idx_to_mask,
                        cp_idx_to_sent_vec):
  tp_vectors = []
  is_hits = []
  masks = []
  cp_sent_vecs = []
  graphs = []
  coverpoints = []
  for row in range(dataset[cp_idx]["tp_vectors"].shape[0]):
    gidx = cp_idx_to_midx[cp_idx]
    tp_vectors.append(dataset[cp_idx]["tp_vectors"][row].astype(np.float32))
    is_hits.append(dataset[cp_idx]["is_hits"][row].astype(np.float32))
    masks.append(cp_idx_to_mask[cp_idx].astype(np.bool))
    cp_sent_vecs.append(cp_idx_to_sent_vec[cp_idx].astype(np.float32))
    graphs.append(gidx)
    coverpoints.append(cp_idx)
  tp_vectors = np.vstack(tp_vectors)
  is_hits = np.vstack(is_hits)
  masks = np.vstack(masks)
  cp_sent_vecs = np.vstack(cp_sent_vecs)
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
      "cp_sent_vecs": cp_sent_vecs[p],  # Sentence vector for each coverpoint
  }
  print(f"Dataset for coverpoint '{cp_idx}' shapes: "
        f"{[f'{k}: {v.shape}' for k, v in ret.items()]}")
  return ret


def combine_data(graph_dir, tp_cov_dir,
                 hit_lower_bound=0.1, hit_upper_bound=0.9):
  print("Aggregating sporadic information to finalize dataset...")
  # Load dataset
  design_graph_filepath = os.path.join(graph_dir, "design_graph.pkl")
  assert os.path.exists(design_graph_filepath), (
      f"'{design_graph_filepath}' not found")
  graph_handler = GraphHandler(design_graph_filepath, output_dir=graph_dir)

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
  cp_idx_to_sent_vec = {}
  print(f"Processing coverpoints...")
  for cp_idx in tqdm.tqdm(cov_dataset.keys()):
    midx = bvocab.get_module_index(cp_idx)
    cp_idx_to_midx[cp_idx] = midx
    cp_idx_to_mask[cp_idx] = bvocab.get_mask(cp_idx, midx,
                                             mask_length=graphs[0].n_nodes)
    cp_idx_to_sent_vec[cp_idx] = bvocab.get_sentence_vector(
        graph_handler.design_graph.nodes, cp_idx, midx)

  # Split dataset into train, valid and test
  cp_indices = list(cov_dataset.keys())
  dataset = {}
  for cp_index in cp_indices:
    dataset[cp_index] = combine_data_per_cp(cp_index, cov_dataset,
                                            cp_idx_to_midx, cp_idx_to_mask,
                                            cp_idx_to_sent_vec)
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


def generate_data(graph_dir, tp_cov_dir, tf_data_dir,
                  hit_lower_bound=0.1, hit_upper_bound=0.9,):
  dataset = combine_data(graph_dir, tp_cov_dir,
                         hit_lower_bound, hit_upper_bound)
  save_dataset(dataset, tf_data_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-gd", "--graph_dir", type=str, required=True,
                      help="Directory of the graph dataset.")
  parser.add_argument("-td", "--tp_cov_dir", type=str,
                      help="Directory of the test parameter coverage dataset.")
  parser.add_argument("-fd", "--tf_data_dir", type=str, required=True,
                      help="Directory of the finalized TF dataset.")
  parser.add_argument("--hit_lower_bound", type=float, default=0.1,
                      help="Lower bound of the hit rate.")
  parser.add_argument("--hit_upper_bound", type=float, default=0.9,
                      help="Upper bound of the hit rate.")

  args = parser.parse_args()
  print(f"Received arguments: {args}")
  generate_data(**vars(args))
