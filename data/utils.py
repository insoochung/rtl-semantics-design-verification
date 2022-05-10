import os
import pickle
import codecs

from glob import glob
from typing import List

import yaml
import tqdm
import numpy as np


CODEBERT_MODEL_NAME = "microsoft/codebert-base-mlm"


def load_yaml(filepath: str):
  """Loads a YAML file from the given filepath."""
  with codecs.open(filepath, "r", encoding="utf-8") as f:
    ret = yaml.load(f, Loader=yaml.FullLoader)
  return ret


def load_pkl(pkl_file: str):
  """Loads the RTL file from the given pickle file."""
  with codecs.open(pkl_file, "rb") as f:
    ret = pickle.load(f)
  return ret


def _load_s2v_model(model_name=CODEBERT_MODEL_NAME):
  from transformers import AutoModel, AutoTokenizer
  print("Loading S2V model...")
  model = AutoModel.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  print("S2V model loaded!")
  return model, tokenizer


class TestParameterCoverageHandler:
  def __init__(self, filepath: str = ""):
    self.filepath = filepath
    self.data = {}
    self.cov_to_dp = {}
    self.stale = False
    if filepath and os.path.exists(filepath):
      self.load_from_file()

  def load_from_file(self):
    self.data = np.load(self.filepath, allow_pickle=True).item()
    print(f"Loaded datapoints from dataset: {self.filepath} "
          f"// shape: {[(k, v.shape) for k, v in self.data.items()]}, ")

  def save_to_file(self):
    np.save(self.filepath, self.data)

  def add(self, tp_vectors, is_hits, coverpoints):
    self.stale = True
    if "tp_vectors" not in self.data:
      self.data["tp_vectors"] = tp_vectors
    else:
      self.data["tp_vectors"] = np.concatenate(
          (self.data["tp_vectors"], tp_vectors), axis=0)
    if "is_hits" not in self.data:
      self.data["is_hits"] = is_hits
    else:
      self.data["is_hits"] = np.concatenate(
          (self.data["is_hits"], is_hits), axis=0)
    if "coverpoints" not in self.data:
      self.data["coverpoints"] = coverpoints
    else:
      self.data["coverpoints"] = np.concatenate(
          (self.data["coverpoints"], coverpoints), axis=0)

    print(f"Added {len(tp_vectors)} datapoints to dataset:")
    print(f"- current shapes: tp_vectors {self.data['tp_vectors'].shape}, "
          f"is_hits {self.data['is_hits'].shape}, "
          f"coverpoints {self.data['coverpoints'].shape}")

  def arrange_dataset_by_coverpoint(self):
    print("Arranging dataset by coverpoint...")
    if not self.stale and len(self.cov_to_dp) > 0:
      return self.cov_to_dp

    cov_to_dp = {}
    for i in tqdm.tqdm(range(self.data["tp_vectors"].shape[0])):
      tp_vector = self.data["tp_vectors"][i]
      is_hit = self.data["is_hits"][i]
      coverpoint = int(self.data["coverpoints"][i][0])
      if coverpoint not in cov_to_dp:
        cov_to_dp[coverpoint] = {
            "tp_vectors": np.zeros(shape=(0, tp_vector.shape[-1])),
            "is_hits": np.zeros(shape=(0))}
      cov_to_dp[coverpoint]["tp_vectors"] = np.append(
          cov_to_dp[coverpoint]["tp_vectors"],
          tp_vector.reshape(1, -1), axis=0)
      cov_to_dp[coverpoint]["is_hits"] = np.append(
          cov_to_dp[coverpoint]["is_hits"],
          is_hit, axis=0)
    hit_rates = []
    num_th = 0
    for cov, dp in cov_to_dp.items():
      dp["hit_rate"] = np.mean(dp["is_hits"])
      hit_rates.append(dp["hit_rate"])
      if hit_rates[-1] >= 0.05 and hit_rates[-1] <= 0.95:
        num_th += 1

    self.cov_to_dp = cov_to_dp
    self.stale = False
    print("Dataset arranged by coverpoint.")
    return cov_to_dp


class NodeVocab:
  def __init__(self, vocab_filepath: str = ""):
    self.vocab_filepath = vocab_filepath
    self.vocab = {"nodes": [], "meta": {}}
    if vocab_filepath and os.path.exists(vocab_filepath):
      self.load_from_file()

  def load_from_file(self):
    self.vocab = load_yaml(self.vocab_filepath)
    print(f"Loaded vocab from {self.vocab_filepath}")

  def save_to_file(self):
    with codecs.open(self.vocab_filepath, "w") as f:
      yaml.dump(self.vocab, f)
    print(f"Saved vocab to {self.vocab_filepath}")

  def is_loaded(self):
    return len(self.vocab["nodes"]) > 0

  def load_s2v_model(self, model_name):
    model, tokenizer = _load_s2v_model(model_name)
    self.s2v = {"model": model, "tokenizer": tokenizer}

  def add_node_info(self, element_name, element_type, element_info):
    element = {
        "name": element_name, "type": element_type, "info": element_info}
    if element["type"] == "choice":
      element["info"] = sorted(list(element["info"]))
    self.vocab["nodes"].append(element)

  def add_meta(self, key, info):
    self.vocab["meta"][key] = info

  def vectorize(self, node):
    vecs = []
    for element in self.vocab["nodes"]:
      key = element["name"]
      etype = element["type"]
      val = getattr(node, key, None)
      if etype == "int":
        assert val is not None
        # Normalize
        val = float(element["info"]["max"] - val) / (
            element["info"]["max"] - element["info"]["min"] + 1e-6)
        vec = np.array([val])
      elif etype == "choice":
        assert val is not None
        idx = element["info"].index(val)
        vec = np.zeros(len(element["info"]))
        vec[idx] = 1.
      else:
        assert etype == "vec", f"Unknown element type {etype}!"
        if not val:
          if not hasattr(self, "s2v"):
            self.load_s2v_model(element["info"]["model_name"])
          toks = self.s2v["tokenizer"](node.text.strip(), return_tensors="pt")
          vec = self.s2v["model"](**toks)
          vec = vec["pooler_output"].detach().numpy().flatten()
        else:
          vec = val
        if not element["info"]["len"]:
          element["info"]["len"] = vec.shape[0]
        assert element["info"]["len"] == vec.shape[0]
      vecs.append(vec)
    return np.concatenate(vecs, axis=0)


class BranchVocab:
  def __init__(self, vocab_filepath: str = ""):
    self.vocab_filepath = vocab_filepath
    self.branches = []
    self.module_index_to_node_offset = []
    self.signature_to_index = {}
    self.s2v = {}
    if vocab_filepath and os.path.exists(vocab_filepath):
      self.load_from_file()

  def load_from_file(self):
    vocab = load_yaml(self.vocab_filepath)
    self.branches = vocab["branches"]  # List of branch signatures
    self.signature_to_index = vocab["signature_to_index"]
    self.module_index_to_node_offset = vocab["module_index_to_node_offset"]
    print(f"Loaded branch vocab from {self.vocab_filepath}")

  def save_to_file(self):
    with codecs.open(self.vocab_filepath, "w") as f:
      yaml.dump({
          "branches": self.branches,
          "signature_to_index": self.signature_to_index,
          "module_index_to_node_offset": self.module_index_to_node_offset
      }, f)

  def add_branch(self, branch_signature: str):
    self.branches.append(branch_signature)
    self.signature_to_index[branch_signature] = len(self.branches) - 1

  def get_branch_index(self, branch_signature: str):
    if branch_signature not in self.signature_to_index:
      self.add_branch(branch_signature)
    return self.signature_to_index[branch_signature]

  def set_module_index_to_node_offset(self, module_start_index: List[int]):
    self.module_index_to_node_offset = module_start_index

  def get_branch_signature_tuple(self, branch_index: int):
    sig = self.branches[branch_index]
    if isinstance(sig, str):
      sig = eval(sig)
    assert isinstance(sig, tuple)
    return sig

  def get_module_index(self, branch_index: int):
    branch_signature = self.get_branch_signature_tuple(branch_index)
    first_nidx = branch_signature[0]
    offset_range = self.module_index_to_node_offset + [9e9]
    for module_idx, n_offset in enumerate(offset_range):
      if n_offset <= first_nidx < offset_range[module_idx + 1]:
        return module_idx
    assert False, f"{first_nidx} not found within {offset_range}"

  def get_node_indices(self, branch_index: int, module_index: int = None):
    if not module_index:
      module_index = self.get_module_index(branch_index)
    branch_signature = self.get_branch_signature_tuple(branch_index)
    node_offset = self.module_index_to_node_offset[module_index]
    node_indices = [i - node_offset for i in branch_signature]
    return node_indices

  def get_mask(self, branch_index: int, module_index: int = None,
               mask_length: int = None, return_module_index: bool = False):
    node_indices = self.get_node_indices(branch_index, module_index)
    if not mask_length:
      mask_length = node_indices[-1] + 1
    mask = np.zeros(mask_length, dtype=np.float32)
    mask[node_indices] = 1.
    if return_module_index:
      return {"mask": mask, "module_index": module_index}
    return mask

  def get_sentence_vector(self, nodes: list, branch_index: int,
                          module_index: int = None):
    if not self.s2v:
      self.s2v["model"], self.s2v["tokenizer"] = _load_s2v_model(
          CODEBERT_MODEL_NAME)
    node_indices = self.get_node_indices(branch_index, module_index)
    node_text = []
    for i in node_indices:
      node_text.append(" ".join(nodes[i].text.strip().split()))
    branch_text = " ".join(node_text)
    toks = self.s2v["tokenizer"](branch_text, return_tensors="pt")
    vec = self.s2v["model"](**toks)
    vec = vec["pooler_output"].detach().numpy().flatten()
    return vec


class TestParameterVocab:
  def __init__(self, vocab_filepath: str = "", test_templates_dir: str = ""):
    self.vocab_filepath = vocab_filepath
    self.test_templates_dir = test_templates_dir
    self.tokens = None
    self.meta = None
    assert vocab_filepath or test_templates_dir, (
        "Must provide either vocab_filepath or test_templates_dir")
    if vocab_filepath and os.path.exists(vocab_filepath):
      self.load_from_file()  # Vocab filepath takes precedence
    elif test_templates_dir:
      self.generate_from_test_template()

  def load_from_file(self):
    d = load_yaml(self.vocab_filepath)
    self.tokens = d["tokens"]
    self.meta = d["param_info"]
    print(f"Loaded vocab from {self.vocab_filepath}")

  def save_to_file(self, vocab_filepath: str = ""):
    if vocab_filepath:
      self.vocab_filepath = vocab_filepath
    with codecs.open(self.vocab_filepath, "w") as f:
      yaml.dump({"tokens": self.tokens, "param_info": self.meta}, f)
    print(f"Saved vocab to {self.vocab_filepath}")

  def generate_from_test_template(self):
    test_parameters = {}
    tests = set()
    for test_template_path in glob(
            os.path.join(self.test_templates_dir, "*.yaml")):
      test_template = load_yaml(test_template_path)
      if "gen_opts" in test_template:
        for k, v in test_template["gen_opts"].items():
          assert k not in test_parameters, f"Duplicate parameter name {k}."
          test_parameters[k] = v
      if "rtl_test" in test_template:
        tests.add(test_template["rtl_test"])
    tests = sorted(list(tests))
    test_parameters["rtl_test"] = {
        "type": "choice",
        "values": tests,
        "default": tests[0]
    }
    vocab = []
    idx = 0
    for key in test_parameters:
      gen_opt = test_parameters[key]
      gen_type = gen_opt["type"]
      description = ""
      min_val = None
      max_val = None
      default = None
      is_one_hot = 0
      if "description" in gen_opt:
        description = gen_opt["description"]
      if gen_type in ["bool", "int"]:
        if "forced_default" in gen_opt:
          default = gen_opt["forced_default"]
          min_val = max_val = default
        else:
          default = gen_opt["default"]
          if gen_type == "bool":
            min_val, max_val = 0, 1
          else:
            min_val, max_val = gen_opt["min_val"], gen_opt["max_val"]
        vocab.append({
            "idx": idx, "key": key, "type": gen_type, "min_val": min_val,
            "max_val": max_val, "default": default, "is_one_hot": is_one_hot,
            "description": description})
        idx += 1
      elif gen_type == "choice":
        values = gen_opt["values"]
        default = gen_opt["default"]
        is_one_hot = 1
        for v in values:
          new_key = f"{key}+{v}"
          min_val, max_val = 0, 1
          new_default = int(default == v)
          vocab.append({
              "idx": idx, "key": new_key, "type": gen_type,
              "description": description, "min_val": min_val,
              "max_val": max_val, "default": new_default,
              "is_one_hot": is_one_hot})
          idx += 1
    self.tokens = vocab
    self.meta = test_parameters

  def get_test_parameters_from_vector(self, tp, return_one_hots=False):
    norm = []
    for _tp in tp:
      norm.append(_tp if 0 <= _tp <= 1 else 1 if _tp > 1 else 0)
    test_params = {}
    one_hots = {}

    for token in self.tokens:
      key = token["key"]
      idx = token["idx"]
      if token["type"] == "int":
        tp_elem = round(
          norm[idx] * (token["max_val"] - token["min_val"] + 1e-7)
          + token["min_val"])
      elif token["type"] == "bool":
        tp_elem = bool(norm[idx] > 0.5)
      elif token["type"] == "choice": # One-hot case, collect all choices
        choice_key, choice = key.split("+")
        if choice_key not in one_hots:
          one_hots[choice_key] = {"indices": [], "choices": [], "vals": []}
        one_hots[choice_key]["indices"].append(idx)
        one_hots[choice_key]["choices"].append(choice)
        one_hots[choice_key]["vals"].append(norm[idx])
        continue # Handle one hots later on
      else:
        raise NotImplementedError(f"Unknown type {token['type']}")
      test_params[key] = tp_elem

    for key in one_hots:
      idxs = one_hots[key]["indices"]
      vals = one_hots[key]["vals"]
      choices = one_hots[key]["choices"]
      assert len(idxs) == len(choices) and len(choices) == len(vals)
      select_idx = vals.index(max(vals))
      test_params[key] = choices[select_idx]

    if return_one_hots:
      return test_params, one_hots
    return test_params

  def normalize_test_params_vector(self, tp_vec):
    test_params = self.get_test_parameters_from_vector(tp_vec)
    res = [0.0 for _ in range(len(self.tokens))]
    for token in self.tokens:
      key = token["key"]
      idx = token["idx"]
      if token["type"] == "int":
        if token["max_val"] == token["min_val"]:
          res[idx] = 1.0
          continue
        res[idx] = (float(test_params[key])
                    / (token["max_val"] - token["min_val"] + 1e-7))
      elif token["type"] == "bool":
        res[idx] = 1.0 if test_params[key] else 0.0
      elif token["type"] == "choice":
        choice_key, choice = key.split("+")
        if test_params[choice_key] == choice:
          res[idx] = 1.0
      else:
        raise NotImplementedError(f"Unknown type {token['type']}")

    return res

  def vectorize_test_from_file(self, test_filepath: str, normalize=True):
    with codecs.open(test_filepath, "rb") as f:
      test = yaml.load(f, Loader=yaml.FullLoader)
    assert len(test) == 1, "Test file must contain only one test"
    test_parameters = test[0]["gen_opts"].split()
    test_parameters_vec = self.vectorize_test(normalize, test, test_parameters)

    return test_parameters_vec

  def vectorize_test(self, normalize, test, test_parameters):
      parsed_tp = {}
      for tp in test_parameters:
        key, val = tp.split("=")
        key = key[1:]
        parsed_tp[key] = val
        assert key in self.meta, f"Key '{key}' cannot be handled with this vocab"
        if self.meta[key]["type"] in ["int", "bool"]:
          parsed_tp[key] = int(val)
      parsed_tp["rtl_test"] = test[0]["rtl_test"]

      test_parameters_vec = []
      for token in self.tokens:
        key = token["key"]
        if token["is_one_hot"]:
          assert token["type"] == "choice"
          key, one_hot_val = key.split("+")
          test_parameters_vec.append(int(parsed_tp[key] == one_hot_val))
          continue
        tp_elem = parsed_tp[key]
        if normalize and token["type"] == "int":
          tp_elem = ((tp_elem - token["min_val"])
                   / (token["max_val"] - token["min_val"] + 1e-7))
        test_parameters_vec.append(tp_elem)
      return test_parameters_vec


class CoveredTestList:
  def __init__(self, filepath: str = ""):
    self.filepath = filepath
    self.covered_tests = set()
    if self.filepath and os.path.exists(self.filepath):
      self.load_from_file()

  def __contains__(self, key):
    return os.path.abspath(key) in self.covered_tests

  def add(self, test_filepath):
    self.covered_tests.add(os.path.abspath(test_filepath))

  def load_from_file(self):
    with codecs.open(self.filepath, "r", encoding="utf-8") as f:
      lines = f.readlines()
    for i, l in enumerate(lines):
      lines[i] = l.strip()
    self.covered_tests = set(lines)

  def save_to_file(self):
    with codecs.open(self.filepath, "w") as f:
      for t in sorted(list(self.covered_tests)):
        f.write(t + "\n")
