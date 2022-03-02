import os
import yaml
import numpy as np


class DatasetSaver:
  def __init__(self, filepath: str = ""):
    self.filepath = filepath
    self.data = {}
    if filepath and os.path.exists(filepath):
      self.load_from_file()

  def load_from_file(self):
    self.data = np.load(self.filepath)

  def save_to_file(self):
    np.save(self.filepath, self.data)

  def add(self, tp_vectors, is_hits, coverpoints):
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


class BranchVocab:
  def __init__(self, vocab_filepath: str = ""):
    self.vocab_filepath = vocab_filepath
    self.branches = []
    self.signature_to_index = {}
    if vocab_filepath and os.path.exists(vocab_filepath):
      self.load_from_file()

  def load_from_file(self):
    with open(self.vocab_filepath, "r") as f:
      vocab = yaml.load(f)
    self.branches = vocab["branches"]  # List of branch signatures
    self.signature_to_index = vocab["signature_to_index"]

  def save_to_file(self):
    with open(self.vocab_filepath, "w") as f:
      yaml.dump({
          "branches": self.branches,
          "signature_to_index": self.signature_to_index
      }, f)

  def add_branch(self, branch_to_index: str):
    self.branches.append(branch_to_index)
    self.signature_to_index[branch_to_index] = len(self.branches) - 1

  def get_branch_index(self, branch_to_index: str):
    if branch_to_index not in self.signature_to_index:
      self.add_branch(branch_to_index)
    return self.signature_to_index[branch_to_index]


class TestParameterVocab:
  def __init__(self, vocab_filepath: str = "", test_template_path: str = ""):
    self.vocab_filepath = vocab_filepath
    self.test_template_path = test_template_path
    assert vocab_filepath or test_template_path, (
        "Must provide either vocab_filepath or test_template_path")
    if vocab_filepath:
      self.load_from_file()  # Vocab filepath takes precedence
    elif test_template_path:
      self.generate_from_test_template()

  def load_from_file(self):
    with open(self.vocab_filepath, "rb") as f:
      d = yaml.load(f, Loader=yaml.FullLoader)
    self.tokens = d["tokens"]
    self.meta = d["param_info"]
    print(f"Loaded vocab from {self.vocab_filepath}")

  def save_to_file(self, vocab_filepath: str = ""):
    if vocab_filepath:
      self.vocab_filepath = vocab_filepath
    with open(self.vocab_filepath, "w") as f:
      yaml.dump({"tokens": self.tokens, "param_info": self.meta}, f)
    print(f"Saved vocab to {self.vocab_filepath}")

  def generate_from_test_template(self):
    with open(self.test_template_path, "rb") as f:
      test_template = yaml.load(f, Loader=yaml.FullLoader)

    test_parameters = test_template["gen_opts"]
    vocab = []
    idx = 0
    for key in test_template["gen_opts"]:
      gen_opt = test_template["gen_opts"][key]
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

  def vectorize_test(self, test_filepath: str, normalize=True):
    with open(test_filepath, "rb") as f:
      test = yaml.load(f, Loader=yaml.FullLoader)
    assert len(test) == 1, "Test file must contain only one test"
    test_parameters = test[0]["gen_opts"].split()
    parsed_tp = {}
    for tp in test_parameters:
      key, val = tp.split("=")
      key = key[1:]
      parsed_tp[key] = val
      assert key in self.meta, f"Key '{key}' cannot be handled with this vocab"
      if self.meta[key]["type"] in ["int", "bool"]:
        parsed_tp[key] = int(val)

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
    with open(self.filepath, "r") as f:
      lines = f.readlines()
    for i, l in enumerate(lines):
      lines[i] = l.strip()
    self.covered_tests = set(lines)

  def save_to_file(self):
    with open(self.filepath, "w") as f:
      for t in sorted(list(self.covered_tests)):
        f.write(t + "\n")
