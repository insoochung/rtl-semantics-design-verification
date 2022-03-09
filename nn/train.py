import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.utils import NodeVocab, BranchVocab, TestParameterCoverageHandler
from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase


def main():
  # TODO: Change hard-coded variable to passible flags
  graph_dir = "generated/cdfgs"
  test_parameter_coverage_path = "out/dataset.tp_cov.npy"
  # These could be a config json or a yaml somewhere
  hit_lower_bound = 0.05
  hit_high_bound = 0.95
  n_hidden = 32
  n_gcn_layer = 4
  n_mlp_hidden = 32
  dropout = 0.1

  model = Design2VecBase(
      n_hidden=32, n_gcn_layers=4, n_mlp_hidden=32, dropout=0.1)
  # Load datasets
  graph_handler = GraphHandler(output_dir=graph_dir)
  graph_handler.load_from_file()

  # Constant dataset
  dataset = graph_handler.get_dataset()  # Graph is given as spektral dataset
  # Varible dataset and target
  tp_cov = TestParameterCoverageHandler(filepath=test_parameter_coverage_path)
  tp_cov.load_from_file()
  cp_to_dp = tp_cov.arrange_dataset_by_coverpoint()
  # TODO: Filter coverpoints depending on the hit_lower_bound and hit_high_bound
  # TODO: Print number of coverpoints after filtering

  # TODO: Add training loop


if __name__ == "__main__":
  main()
