import os
import argparse
import yaml
import pickle
from glob import glob
from typing import List, Tuple, Dict, Any

from constructor import RtlFile, Module
from graph import Node, BranchNode, EndNode
from constants import Condition


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
    cdfg = _load_pkl(fp)
    ret[os.path.basename(fp).split(".")[0]] = cdfg
  return ret


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
  sum_nodes = 0
  sum_coverpoints = 0
  covered_subpaths = {}
  synced_coverage = {}
  for module_name in modules:
    print(f"Syncing coverage for module: {module_name} - parsed from "
          f"'{sim_covs[module_name]['filepath']}'")
    module_coverage = {}
    cdfg = cdfgs[module_name]
    nodes = cdfg.nodes
    sum_nodes += len(nodes)
    node_to_index = cdfg.node_to_index
    line_number_to_nodes = cdfg.line_number_to_nodes

    for sim_cov in sim_covs[module_name]["coverages"]:
      branch_line_nums = sim_cov["line_num"]
      first_ln = branch_line_nums[0]
      if first_ln not in line_number_to_nodes:
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
                                          line_number_to_nodes,
                                          covered_subpaths)
          coverpoint = tuple(node_to_index[n] for n in cdfg_subpath)
          ref_d["traces"][trace_signature] = {
              # TODO: add test parameters once we have them in YAML.
              # "test_parameters": trace["test_parameters"],
              "is_hit": is_hit,
              "coverpoint": coverpoint}
          sum_coverpoints += 1
    synced_coverage[module_name] = module_coverage
  print(f"Total number of nodes: {sum_nodes}")
  print(f"Total number of coverpoints: {sum_coverpoints}")
  print(f"Total number of unique cdfg subpaths: {len(covered_subpaths)}")
  os.makedirs(output_dir, exist_ok=True)
  synced_cov_path = os.path.join(output_dir, "synced_data.dict.pkl")
  with open(synced_cov_path, "wb") as f:
    pickle.dump(synced_coverage, f)
    print(f"Saved synced coverage to: {synced_cov_path}")
  assert synced_coverage == _load_pkl(synced_cov_path)  # Test load
  # TODO: serialize and save as DL training loader friendly format.


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-sd", "--sim_cov_dir", required=True,
                      help="Directory containing the coverage YAMLs")
  parser.add_argument("-cd", "--cdfg_dir", required=True,
                      help="Directory containing the constructed CDFGs")
  parser.add_argument("-od", "--output_dir", default="generated/dataset",
                      help="Directory to write the output files")
  args = parser.parse_args()
  generate_dataset(args.sim_cov_dir, args.cdfg_dir, args.output_dir)


if __name__ == "__main__":
  main()
