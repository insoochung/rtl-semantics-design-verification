import os
import argparse
import yaml
import pickle
from glob import glob
from typing import List, Tuple, Dict

from constructor import RtlFile, Module
from graph import Node, BranchNode


def _load_yaml(filepath: str):
  """Loads a YAML file from the given filepath."""
  with open(filepath, "r") as f:
    ret = yaml.load(f, Loader=yaml.FullLoader)
  return ret


def _load_rtl_file_from_pkl(pkl_file: str):
  """Loads the RTL file from the given pickle file."""
  with open(pkl_file, "rb") as f:
    ret = pickle.load(f)
  return ret


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


def load_cdfgs(cdfg_dir: str):
  """Loads the CDFGs from the given directory.
      CDFGs are expected to be pickles of RtlFile class instances.
  """
  ret = {}
  for fp in glob(os.path.join(f"{cdfg_dir}/*.pkl")):
    cdfg = _load_rtl_file_from_pkl(fp)
    ret[os.path.basename(fp).split(".")[0]] = cdfg
  return ret


def get_branch(nodes: List[Node], line_number: int,
               trace_signature: Tuple[int], branch_type: str,
               node_to_index: Dict[Node, int],
               line_number_to_nodes: Dict[int, List[Node]]):
  """Return nodes in branch subpath with respect to trace signature"""
  start_node_cands = []
  for node in line_number_to_nodes[line_number]:
    if isinstance(node, BranchNode):
      start_node_cands.append(node)
  assert len(start_node_cands) == 1, "Multiple branch nodes on same line"
  start_node = start_node_cands[0]
  condition_block = start_node.to_list()
  num_cond = sum(1 for node in condition_block if isinstance(node, BranchNode))
  if num_cond != len(trace_signature):
    print(f"{branch_type} branch @ line {line_number}")
    print(f"Trace: {trace_signature}")
    print(
        f"# of conditions in condition block ({num_cond}) != # of conditions "
        f"in the trace signature: {len(trace_signature)}")
    # start_node.print_block()
    # assert False, (
    #     f"# of conditions in condition block ({num_cond}) != # of conditions "
    #     f"in the trace signature: {len(trace_signature)}")


def generate_dataset(sim_cov_dir: str, cdfg_dir: str, output_dir: str):
  """Generates the dataset from the given coverage and CDFG directories."""

  # Load the simulator coverage
  sim_covs = load_simulator_coverage(sim_cov_dir)
  # Load the CDFGs
  cdfgs = load_cdfgs(cdfg_dir)

  # Get module names
  modules = set(sim_covs.keys())
  irrelevant = modules - set(cdfgs.keys())
  if irrelevant:
    print(f"Warning: {len(irrelevant)} modules are irrelevant: {irrelevant}")
    print("These modules may be part of the testbench but not the actual "
          "design, or may not have coverpoints (i.e. branches within always).")
  modules = sorted(modules - irrelevant)

  # Sync coverage from the simulator to CDFG paths.
  synced_coverage = {}
  for module_name in modules:
    print(f"Syncing coverage for module: {module_name} - parsed from "
          f"'{sim_covs[module_name]['filepath']}'")
    module_coverage = {}
    cdfg = cdfgs[module_name]
    nodes = cdfg.nodes
    node_to_index = cdfg.node_to_index
    line_number_to_nodes = cdfg.line_number_to_nodes
    for sim_cov in sim_covs[module_name]["coverages"]:
      line_num = sim_cov["line_num"]
      if line_num not in line_number_to_nodes:
        continue  # Skip branches that are not inside an always block
      branch_type = sim_cov["branch_type"]
      traces = sim_cov["coverage"]
      if line_num not in module_coverage:
        d = {
            "line_num": line_num,
            "branch_type": branch_type,
            "trace_len": len(traces[0]["trace"]),
            "traces": {
                # key: tuple of condition strings
                # value: {"branch": tuple of node ids, "is_hit": bool}
            }
        }
        module_coverage[line_num] = d
      ref_d = module_coverage[line_num]
      for trace in traces:
        is_hit = bool(int(trace["cov"]))
        trace_signature = tuple(trace["trace"])
        assert ref_d["trace_len"] == len(trace_signature)
        if trace_signature in ref_d["traces"]:
          ref_d["traces"][trace_signature]["is_hit"] |= is_hit
        else:
          # Find relevant node branches
          branch = get_branch(nodes, line_num, trace_signature, branch_type,
                              node_to_index, line_number_to_nodes)
        assert 0, "TODO: implement"

  # TODO: serialize and save


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-sd", "--sim_cov_dir", required=True,
                      help="Directory containing the coverage YAMLs")
  parser.add_argument("-cd", "--cdfg_dir", required=True,
                      help="Directory containing the constructed CDFGs")
  parser.add_argument("-od", "--output_dir", required=True,
                      help="Directory to write the output files")
  args = parser.parse_args()
  generate_dataset(args.sim_cov_dir, args.cdfg_dir, args.output_dir)


if __name__ == "__main__":
  main()
