import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.utils import NodeVocab, BranchVocab, TestParameterCoverageHandler
from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase


def main():
  # TODO: Change hard-coded variable to passible flags
  model = Design2VecBase(
      n_hidden=32, n_gcn_layers=4, n_mlp_hidden=32, dropout=0.1)
  # Load datasets
  graph_handler = GraphHandler(output_dir="generated/cdfgs")
  graph_handler.load_from_file()
  dataset = graph_handler.get_dataset()
  tp_cov = TestParameterCoverageHandler(
      filepath="generated/backup/tp_cov/dataset.npy")
  tp_cov.load_from_file()
  cp_to_dp = tp_cov.arrange_dataset_by_coverpoint()
  # TODO: Add training loop

if __name__ == "__main__":
  main()
