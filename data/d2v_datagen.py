import os
import sys
import argparse
import yaml
import pickle
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constructor import (
    DesignGraph, Module, construct_design_graph)
from cdfg.graph import Node, BranchNode, EndNode
from cdfg.constants import Condition
from coverage.extract_from_urg import extract as extract_from_urg
from data.utils import (BranchVocab, TestParameterVocab, CoveredTestList,
                        DatasetSaver)


def _load_yaml(filepath: str):
  """Loads a YAML file from the given filepath."""
  with open(filepath, "r") as f:
    ret = yaml.load(f, Loader=yaml.FullLoader)
  return ret


def _load_pkl(pkl_file: str):
  """Loads the RTL file from the given pickle file."""
  with open(pkl_file, "rb") as f:
    ret = pickle.load(f)
  return ret


def get_dataset_utilites(test_templates_dir: str, output_dir: str):
  """Construct vocabulary that contains vectorization information for TPs"""
  os.makedirs(output_dir, exist_ok=True)
  vocab_files = glob(os.path.join(output_dir, "vocab.*.yaml"))
  bvocab_filepath = os.path.join(output_dir, "vocab.branches.yaml")
  if bvocab_filepath in vocab_files:
    vocab_files.remove(bvocab_filepath)
  assert len(vocab_files) <= 1, "There should be only one or no vocab file"

  if not vocab_files:
    assert test_templates_dir, (
        f"No test templates directory given to generate vocab from")
    test_templates = glob(os.path.join(test_templates_dir, "*.yaml"))
    assert test_templates, f"No test templates found in '{test_templates_dir}'"
    # TODO: Handle multiple test templates later on.
    assert len(test_templates) == 1, (
        f"Data generator can handle one test template at a time, "
        f"but found {len(test_templates)} test templates")
    vocab = TestParameterVocab(test_template_path=test_templates[0])
    vocab_filepath = os.path.join(
        output_dir, f"vocab.{len(vocab.tokens)}.yaml")
    vocab.save_to_file(vocab_filepath)
  else:
    vocab = TestParameterVocab(vocab_filepath=vocab_files[0])
  bvocab_filepath = os.path.join(output_dir, "vocab.branches.yaml")
  bvocab = BranchVocab(vocab_filepath=bvocab_filepath)
  covered_tests_filepath = os.path.join(output_dir, "covered_tests.txt")
  covered_tests = CoveredTestList(covered_tests_filepath)
  dataset_path = os.path.join(output_dir, "dataset.npy")
  dataset_saver = DatasetSaver(dataset_path)
  utils = {"vocab": vocab, "bvocab": bvocab, "covered_tests": covered_tests,
           "dataset_saver": dataset_saver}
  return utils


def load_simulator_coverage(sim_cov_dir: str):
  """Loads the simulator coverage from the given directory.
     Coverage files are expected to be yaml files parsed from URG coverage
     reports.
  """
  ret = {}
  for fp in glob(os.path.join(f"{sim_cov_dir}/*.yaml")):
    cov = _load_yaml(fp)
    cov["filepath"] = fp
    ret[cov["module_name"]] = cov
  return ret


def load_design_graph(design_graph_dir: str):
  """Loads the design graph from the given directory.
    Design graph files are expected to be pickle files parsed from RTL files.
  """
  cand_pkl_path = os.path.join(design_graph_dir, "design_graph.pkl")
  if os.path.exists(cand_pkl_path):
    pkl_path = cand_pkl_path
  else:
    pkls = glob(os.path.join(design_graph_dir, "*.pkl"))
    assert len(pkls) == 1, (
        "There should be only one design graph file in the directory")
    pkl_path = pkls[0]
  return _load_pkl(pkl_path)


def preprocess_trace_conditions(trace_conditions: List[str]):
  """Preprocesses the trace conditions to match with Node.next_nodes conditions
  """
  ret = []
  for sig in trace_conditions:
    sig = sig.replace(", ", ",").strip()
    if " " in sig:  # If there are multiple items to conditions
      _sig = preprocess_trace_conditions(sig.split(" "))
      sig = " ".join(_sig)
    elif sig.startswith("{") and sig.endswith("}"):  # Handle concatenate
      _sig = sig[1:-1]
      _sig = preprocess_trace_conditions(_sig.split(","))
      sig = "{" + ",".join(_sig) + "}"
    elif sig == "X":  # If don't care, substitute with None
      sig = None
    elif "." in sig:  # If object attribute, only take attribute name
      sig = sig.split(".")[-1]
    elif "::" in sig:  # If pacakge attribute, only take attribute name
      sig = sig.split("::")[-1]
    elif "'b" in sig:  # If binary number, format it
      _sig = sig.split("'b")
      num_digits = int(_sig[0])
      _sig[1] = _sig[1].zfill(num_digits)
      sig = "'b".join(_sig)
    elif sig == "true":
      sig = "1"
    elif sig == "false":
      sig = "0"

    ret.append(sig)

  return tuple(ret)


def get_cdfg_subpath(branch_line_nums: List[int], trace_conditions: Tuple[int],
                     line_number_to_nodes: Dict[int, List[Node]],
                     covered_subpaths: Dict[Tuple[Node], Tuple[int, Tuple[Any]]]):
  """Return nodes in branch subpath with respect to trace conditions"""
  branch_nodes = []
  branch_conditions = []
  for ln in branch_line_nums:
    cnt = 0
    for node in line_number_to_nodes[ln]:
      if isinstance(node, BranchNode):
        branch_nodes.append(node)
        cnt += 1
    assert cnt == 1, "There should be one branch node per line"
    bn = branch_nodes[-1]
    branch_conditions.append([n[1] for n in bn.next_nodes])
  assert len(branch_nodes) == len(trace_conditions)
  for i, cond in enumerate(trace_conditions):
    if cond is None:
      continue
    assert cond in branch_conditions[i], (
        f"{cond} not in {branch_conditions[i]}")

  nodes_and_conditions = zip(branch_nodes, trace_conditions)
  start_node = branch_nodes[0]
  cdfg_subpath = start_node.to_list(conditions=nodes_and_conditions)

  assert all([
      n in cdfg_subpath or not c for (n, c) in nodes_and_conditions]), (
      "Trace subpath should include all branch nodes with non-dont-care "
      "conditions")
  subpath_signature = tuple(cdfg_subpath)
  if subpath_signature in covered_subpaths:
    _, covered_trace = covered_subpaths[subpath_signature]
    assert len(covered_trace) == len(trace_conditions)
    for i, (ct, t) in enumerate(zip(covered_trace, trace_conditions)):
      if ct != t:
        assert (set({ct, t}) == set({None, Condition.TRUE})
                and isinstance(branch_nodes[i].next_nodes[1][0], EndNode))
  covered_subpaths[subpath_signature] = (
      branch_line_nums[0], tuple(trace_conditions))
  return cdfg_subpath


def generate_dataset_inner(test_dir: str, design_graph: DesignGraph,
                           vocab: TestParameterVocab,
                           branch_vocab: BranchVocab):
  """Generates the dataset from the given coverage and CDFG directories."""
  # Load the simulator coverage
  sim_cov_dir = os.path.join(test_dir, "urg_report", "extracted")
  if not os.path.isdir(sim_cov_dir):
    print(f"Looked for extracted coverage information in '{sim_cov_dir}' "
          f"but no luck.")
    return
  sim_covs = load_simulator_coverage(sim_cov_dir)
  # Compose a dictionary of module graphs
  module_graphs = {}
  for module in design_graph.modules:
    module_name = module.module_name
    module_graphs[module_name] = module
  # Load test
  testfiles = glob(os.path.join(test_dir, "*.yaml"))
  assert len(testfiles) == 1, (
      f"There should be only one test file within {test_dir}")
  testfile = testfiles[0]
  tp_vector = vocab.vectorize_test(testfile)
  # Get module names
  modules = set(sim_covs.keys())
  irrelevant = modules - set(module_graphs.keys())
  if irrelevant:
    print(f"Warning: {len(irrelevant)} modules are irrelevant: {irrelevant}")
    print("These modules may be part of the testbench but not the actual "
          "design, or may not have coverpoints (i.e. branches within always).")
  modules = sorted(modules - irrelevant)

  # Sync coverage from the simulator to CDFG paths.
  sum_nodes = 0
  sum_coverpoints = 0
  covered_subpaths = {}
  synced_coverage = {}
  # For node to global index mapping
  node_to_index_global = design_graph.node_to_index
  tp_vector_list = []
  is_hit_list = []
  coverpoint_idx_list = []
  for module_name in modules:
    print(f"Syncing coverage for module: {module_name} - parsed from "
          f"'{sim_covs[module_name]['filepath']}'")
    module_coverage = {}
    cdfg = module_graphs[module_name]
    nodes = cdfg.nodes
    sum_nodes += len(nodes)
    line_number_to_nodes_local = cdfg.line_number_to_nodes

    for sim_cov in sim_covs[module_name]["coverages"]:
      branch_line_nums = sim_cov["line_num"]
      first_ln = branch_line_nums[0]
      if first_ln not in line_number_to_nodes_local:
        continue  # Skip branches that are not inside an always block
      branch_type = sim_cov["branch_type"]
      traces = sim_cov["traces"]
      if first_ln not in module_coverage:
        trace_len = len(traces[0]["trace"])
        assert len(branch_line_nums) == trace_len, (
            f"Branch line numbers ({len(branch_line_nums)}) != "
            f"trace length ({trace_len})")
        d = {
            "branch_line_nums": branch_line_nums,
            "branch_type": branch_type,
            "trace_len": trace_len,
            "traces": {
                # key: tuple of condition strings
                # value: {"coverpoint": tuple of node ids, "is_hit": bool}
            }
        }
        module_coverage[first_ln] = d

      ref_d = module_coverage[first_ln]
      for trace in traces:
        is_hit = bool(int(trace["cov"]))
        trace_conditions = list(trace["trace"])
        trace_conditions = preprocess_trace_conditions(trace_conditions)
        assert ref_d["trace_len"] == len(trace_conditions)
        trace_signature = tuple(trace_conditions)
        if trace_signature in ref_d["traces"]:
          ref_d["traces"][trace_signature]["is_hit"] |= is_hit
        else:
          # Find relevant node branches
          cdfg_subpath = get_cdfg_subpath(branch_line_nums, trace_conditions,
                                          line_number_to_nodes_local,
                                          covered_subpaths)
          coverpoint_signature = str(
              tuple(node_to_index_global[n] for n in cdfg_subpath))
          coverpoint_idx = branch_vocab.get_branch_index(coverpoint_signature)
          datapoint = {
              "test_parameters": tp_vector,
              "is_hit": is_hit,
              "coverpoint": coverpoint_idx}
          ref_d["traces"][trace_signature] = datapoint
          tp_vector_list.append(tp_vector)
          is_hit_list.append(is_hit)
          coverpoint_idx_list.append(coverpoint_idx)
          sum_coverpoints += 1
    synced_coverage[module_name] = module_coverage
  print(f"Total number of nodes: {sum_nodes}")
  print(f"Total number of coverpoints: {sum_coverpoints}")
  print(f"Total number of unique cdfg subpaths: {len(covered_subpaths)}")
  tp_vector_list = [
      np.array(elem, dtype=np.float32) for elem in tp_vector_list]
  is_hit_list = [
      np.array(elem, dtype=np.float32) for elem in is_hit_list]
  coverpoint_idx_list = [
      np.array(elem, dtype=np.int32) for elem in coverpoint_idx_list]
  return {
      "tp_vectors": np.vstack(tp_vector_list),
      "is_hits": np.vstack(is_hit_list),
      "coverpoints": np.vstack(coverpoint_idx_list)}


def generate_dataset(simulated_tests_dir: str, design_graph_dir: str,
                     output_dir: str, test_templates_dir: str = ""):
  """Generates the dataset from the given coverage and CDFG directories."""
  test_dirs = glob(os.path.join(simulated_tests_dir, "*"))
  design_graph = load_design_graph(design_graph_dir)
  assert len(test_dirs) > 0, (
      f"No test directories found in '{simulated_tests_dir}'")
  utils = get_dataset_utilites(test_templates_dir, output_dir)
  covered_tests = utils["covered_tests"]
  vocab = utils["vocab"]
  bvocab = utils["bvocab"]
  dataset_saver = utils["dataset_saver"]
  for test_dir in sorted(test_dirs):
    if not os.path.isdir(test_dir):
      continue
    if test_dir in covered_tests:
      print(f"Skipping {test_dir} as it has already been covered")
      continue

    print(f"Generating dataset with simulated test information in: {test_dir}")
    examples = generate_dataset_inner(test_dir, design_graph, vocab, bvocab)
    if not examples:
      continue
    dataset_saver.add(**examples)
    covered_tests.add(test_dir)

  # These may have been updated, so save them.
  dataset_saver.save_to_file()
  bvocab.save_to_file()
  covered_tests.save_to_file()


def main():
  parser = argparse.ArgumentParser()
  # Simulation coverage related arguments
  parser.add_argument("-ec", "--extract_coverage", default=False,
                      action="store_true",
                      help="Extract coverage from simulator output.")
  parser.add_argument("-sd", "--simulated_tests_dir",
                      default=False, required=True,
                      help=("Directory where tests and relevant coverage "
                            "information is located."))

  # CDFG related arguments
  parser.add_argument("-cg", "--construct_cdfg", default=False,
                      action="store_true",
                      help="Extract CDFGs from parsed_rtl_dir.")
  parser.add_argument("-pr", "--parsed_rtl_dir",
                      help=("Directory where verible generated ASTs "
                            "(parsed from RTLs) are located in json format"))
  parser.add_argument("-rd", "--rtl_dir",
                      help="Directory where the original RTL files are "
                           "located")
  parser.add_argument("-gd", "--design_graph_dir", required=True,
                      help="Directory containing the constructed CDFGs")

  # Test parameter related arguments
  parser.add_argument("-tt", "--test_templates_dir", default="",
                      help="Directory to test templates")

  # Dataset related arguments
  parser.add_argument("-od", "--output_dir", default="generated/dataset",
                      help="Directory to write the output files")
  args = parser.parse_args()
  extract_coverage = args.extract_coverage
  simulated_tests_dir = args.simulated_tests_dir
  construct_cdfg = args.construct_cdfg
  parsed_rtl_dir = args.parsed_rtl_dir
  rtl_dir = args.rtl_dir
  design_graph_dir = args.design_graph_dir
  test_templates_dir = args.test_templates_dir
  output_dir = args.output_dir

  if extract_coverage:
    extract_from_urg(report_dir=simulated_tests_dir, in_place=True)
  if construct_cdfg:
    assert parsed_rtl_dir and rtl_dir, (
        "Must specify both parsed_rtl_dir and rtl_dir to construct CDFGs")
    construct_design_graph(parsed_rtl_dir, rtl_dir, design_graph_dir)

  generate_dataset(simulated_tests_dir, design_graph_dir,
                   output_dir, test_templates_dir)


if __name__ == "__main__":
  main()
