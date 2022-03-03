import os
import sys
import argparse

import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from spektral.data import Dataset
from spektral.data.graph import Graph

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constructor import DesignGraph, Module
from data.utils import NodeVocab, load_pkl


class GraphDataset(Dataset):
  def __init__(self, graphs, **kwargs):
    self.graphs = graphs
    super().__init__(**kwargs)

  def download(self):
    return

  def read(self):
    outputs = []
    for x, a in zip(self.graphs["x"], self.graphs["a"]):
      outputs.append(Graph(x=x, a=a))
    return outputs


class GraphHandler:
  def __init__(self, design_graph_filepath: str = "", output_dir: str = ""):
    self.design_graph_filepath = design_graph_filepath
    self.output_dir = output_dir or os.path.dirname(design_graph_filepath)
    self.vocab_filepath = os.path.join(self.output_dir, "vocab.graph.yaml")
    self.dataset_path = os.path.join(self.output_dir, "dataset.graphs.npy")
    self.s2v_model_name = "microsoft/codebert-base-mlm"
    self.design_graph = None
    self.vocab = None
    self.graphs = None
    self.adjs = None
    if design_graph_filepath and os.path.exists(design_graph_filepath):
      self.design_graph = load_pkl(design_graph_filepath)

  def load_or_generate_vocab(self):
    if self.vocab:
      return
    os.makedirs(self.output_dir, exist_ok=True)
    self.vocab = NodeVocab(self.vocab_filepath)
    if self.vocab.is_loaded():
      return
    # Gather information for vectorization
    type_info = set()
    block_depth_info = {"min": 1e9, "max": 0}
    seq_vec_info = {"model_name": self.s2v_model_name, "len": None}
    for n in self.design_graph.nodes:
      type_info.add(n.type)
      block_depth_info["min"] = min(block_depth_info["min"],
                                    n.block_depth)
      block_depth_info["max"] = max(block_depth_info["max"],
                                    n.block_depth)
    self.vocab.add_node_info("type", "choice", type_info)
    self.vocab.add_node_info("block_depth", "int", block_depth_info)
    self.vocab.add_node_info("seq_vec", "vec", seq_vec_info)
    # TODO: Add edge related information

    # Add meta information
    self.vocab.add_meta("module_info",
                        {"num_modules": len(self.design_graph.modules),
                         "module_names": [m.module_name for m in self.design_graph.modules],
                         "module_start_index": self.design_graph.module_start_index})
    self.vocab.save_to_file()

  def vectorize_design_graph(self):
    print("Converting design graph to numpy matrices...")
    self.load_or_generate_vocab()  # Load vocabulary in case it is not loaded
    # Calcuate to pad each graph to max length
    num_nodes_max = 0
    for module in self.design_graph.modules:
      num_nodes_max = max(num_nodes_max, len(module.nodes))

    # Get graph per module
    graphs = []
    adjs = []
    for module in self.design_graph.modules:
      print("Processing module {}".format(module.module_name))
      nodes = module.nodes
      vecs = []
      adj = np.zeros((num_nodes_max, num_nodes_max), dtype=np.float32)
      for n in tqdm.tqdm(nodes):
        vecs.append(self.vocab.vectorize(n))
        from_idx = module.node_to_index[n]
        for nn, _ in n.next_nodes:
          # Spektral assumes j->i connection when a[i, j] == 1.
          to_idx = module.node_to_index[nn]
          adj[to_idx][from_idx] = 1
          # TODO: Add edge related information
      # Pad to max length
      graph = np.concatenate([np.expand_dims(v, axis=0) for v in vecs], axis=0)
      pad = np.zeros((num_nodes_max - len(nodes), graph.shape[1]))
      graph = np.concatenate([graph, pad], axis=0)
      graphs.append(np.expand_dims(graph, axis=0))
      adjs.append(np.expand_dims(adj, axis=0))

    self.graphs = np.concatenate(graphs, axis=0)
    self.adjs = np.concatenate(adjs, axis=0)

    print(f"Done converting design graph to numpy matrices.")
    print(f"Graphs shape: {self.graphs.shape} "
          f"(batch, num_nodes, num_features)")
    print(f"Adjacency matrix shape: {self.adjs.shape}"
          f"(batch, num_nodes, num_nodes)")

    return self.graphs, self.adjs

  def load_from_file(self):
    print("Loading design graph from file...")
    assert os.path.exists(self.dataset_path), (
        f"{self.dataset_path} does not exist.")
    self.load_or_generate_vocab()
    data = dict(np.load(self.dataset_path, allow_pickle=True).item())
    self.graphs = data["x"]
    self.adjs = data["a"]
    print(f"Done loading design graph from '{self.dataset_path}'.")

  def get_dataset(self):
    assert self.graphs is not None and self.adjs is not None
    return GraphDataset(graphs={"x": self.graphs, "a": self.adjs})

  def save_to_file(self):
    print("Saving design graph to file...")
    if self.graphs is None or self.adjs is None:
      self.design_graph_to_numpy()
    os.makedirs(self.output_dir, exist_ok=True)

    np.save(self.dataset_path, {"x": self.graphs, "a": self.adjs})
    self.vocab.save_to_file()
    print(f"Done saving design graph to '{self.dataset_path}'.")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-dg", "--design_graph_filepath", type=str, required=True,
                      help="Path to the pickeld design graph file")
  parser.add_argument("-od", "--output_dir", type=str,
                      help="Where to save the output files")
  args = parser.parse_args()
  handler = GraphHandler(args.design_graph_filepath, args.output_dir)
  handler.vectorize_design_graph()
  handler.save_to_file()


if __name__ == "__main__":
  main()
