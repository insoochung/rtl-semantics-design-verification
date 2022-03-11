import os
import sys
import copy

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
  indices = copy.deepcopy(indices)
  np.random.shuffle(indices)
  assert abs(sum(split_ratio) - 1.0) < 1e-7  # Make sure ratio sum to 1
  train_valid_boundary = int(len(indices) * split_ratio[0])
  valid_test_boundary = int(len(indices) * (split_ratio[0] + split_ratio[1]))
  train = indices[:train_valid_boundary]
  valid = indices[train_valid_boundary:valid_test_boundary]
  test = indices[valid_test_boundary:]
  return train, valid, test


def convert_to_tf_dataset(dataset):
  """Convert the dict of numpy dataset to tf.data.Dataset."""
  def ds_gen(dataset):
    for i in range(dataset["coverpoint"].shape[0]):
      yield {k: dataset[k][i] for k in dataset.keys()}

  output_signature = {}
  for key in dataset:
    nd = len(dataset[key].shape)
    output_signature[key] = tf.TensorSpec(
        shape=dataset[key].shape[1:] if nd > 1 else (),
        dtype=dataset[key].dtype)

  return tf.data.Dataset.from_generator(
      lambda: ds_gen(dataset), output_signature=output_signature)


def finalize_dataset(cp_indices, dataset, cp_idx_to_midx, cp_idx_to_mask,
                     graph_dataset, type="train"):
  tp_vectors = []
  is_hits = []
  masks = []
  # graph_x = []
  # graph_a = []
  graphs = []
  coverpoints = []
  for idx in cp_indices:
    for row in range(dataset[idx]["tp_vectors"].shape[0]):
      gidx = cp_idx_to_midx[idx]
      tp_vectors.append(dataset[idx]["tp_vectors"][row].astype(np.float32))
      is_hits.append(dataset[idx]["is_hits"][row].astype(np.float32))
      masks.append(cp_idx_to_mask[idx].astype(np.bool))
      # graph_x.append(graph_dataset[gidx].x.astype(np.float32))
      # graph_a.append(graph_dataset[gidx].a.astype(np.float32))
      graphs.append(gidx)
      coverpoints.append(idx)
  tp_vectors = np.vstack(tp_vectors)
  is_hits = np.vstack(is_hits)
  masks = np.vstack(masks)
  # graph_x = np.vstack(graph_x)
  # graph_a = np.vstack(graph_a)
  graphs = np.array(graphs)
  coverpoints = np.array(coverpoints)
  # Indices permutated to shuffle the dataset in unison
  p = np.random.permutation(len(coverpoints))
  ret = {
      "test_parameters": tp_vectors[p],
      "is_hits": is_hits[p],
      "coverpoint_mask": masks[p],
      # "graph_x": graph_x[p],
      # "graph_a": graph_a[p],
      "graph": graphs[p],
      "coverpoint": coverpoints[p],
  }
  print(f"Dataset '{type}' shapes: ")
  for k, v in ret.items():
    print(f"- {k}: {v.shape}")
  return ret


def main():
  # TODO: Change hard-coded variable to passible flags
  graph_dir = os.path.expanduser("~/generated/d2v_data/graph")
  test_parameter_coverage_dir = os.path.expanduser(
      "~/generated/d2v_data/tp_cov")
  # These could be a config json or a yaml somewhere
  hit_lower_bound = 0.10
  hit_high_bound = 0.90
  n_hidden = 32
  n_gcn_layers = 4
  n_mlp_hidden = 32
  dropout = 0.1
  learning_rate = 0.001
  batch_size = 16

  # Load datasets
  graph_handler = GraphHandler(output_dir=graph_dir)

  # Constant: graphs are given as a spektral dataset
  graphs = graph_handler.get_dataset()

  # Variable: test parameter coverage dataset and target (is_hit)
  bvocab = BranchVocab(  # Utility objects
      os.path.join(test_parameter_coverage_dir, "vocab.branches.yaml"))
  tp_cov = TestParameterCoverageHandler(
      filepath=os.path.join(test_parameter_coverage_dir, "dataset.tp_cov.npy"))
  cov_dataset = tp_cov.arrange_dataset_by_coverpoint()

  # Filter with respect to hit rate
  cov_dataset_filt = {}
  for k, v in cov_dataset.items():
    if hit_lower_bound <= v["hit_rate"] <= hit_high_bound:
      cov_dataset_filt[k] = v
  cov_dataset = cov_dataset_filt
  print(f"# of coverpoints within hit rate range between "
        f"{hit_lower_bound} and {hit_high_bound}: {len(cov_dataset)}")

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
      split_indices(cp_indices, [0.7, 0.15, 0.15])

  dataset = {}
  for key in (["train", "valid", "test"]):
    dataset[key] = finalize_dataset(cp_splits[key], cov_dataset,
                                    cp_idx_to_midx, cp_idx_to_mask,
                                    graphs, type=key)
    dataset[key] = convert_to_tf_dataset(dataset[key])
    dataset[key] = dataset[key].batch(batch_size)

  model = Design2VecBase(graphs, n_hidden=n_hidden, n_gcn_layers=n_gcn_layers,
                         n_mlp_hidden=n_mlp_hidden, dropout=dropout)
  for batch in dataset["train"]:
    # TODO: Finish training loop
    y = model(batch)
    assert 0, y


if __name__ == "__main__":
  main()
