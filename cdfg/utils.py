import re
from typing import List, Union


def get_log_fn(verbose=True):
  """Return a function that prints to stdout"""
  def log_fn(str=""):
    pass
  if verbose:
    log_fn = print

  return log_fn


def get_indent_str(indent):
  return " " * indent


def strip_comments(text):
  return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)


def preprocess_rtl_str(always_str, no_space=False, one_line=False):
  # 1. Remove comments
  res = strip_comments(always_str)
  # 2. Replace multiple spaces with a single space, but indents are preserved.
  lines = res.split("\n")
  for i, line in enumerate(lines):
    indent_size = len(line) - len(line.lstrip())
    lines[i] = " " * indent_size + " ".join(line.split()) + "\n"
  res = "".join(lines)
  if one_line:
    res = " ".join(res.split())
  if no_space:
    res = "".join(res.split())

  return res


def print_tags(verible_tree: dict, indent_size: int = 0):
  """Print the tags of the verible tree

  Args:
  verible_tree -- the verible tree to print the tags of (dict)
  """
  if verible_tree is None:
    return
  if "tag" in verible_tree.keys():
    print(get_indent_str(indent_size), verible_tree["tag"])
  if "children" in verible_tree.keys():
    for c in verible_tree["children"]:
      print_tags(c, indent_size + 2)
  elif "tree" in verible_tree.keys():
    print_tags(verible_tree["tree"], indent_size + 2)


def flatten_tree(verible_tree: dict):
  """DFS-traverse verible tree and return the flattened tree."""
  if verible_tree is None:
    return []
  if "children" in verible_tree.keys():
    res = []
    for c in verible_tree["children"]:
      res += flatten_tree(c)
    return res
  if "tree" in verible_tree.keys():
    return flatten_tree(verible_tree["tree"])
  return [verible_tree]


def find_subtree(verible_tree: dict, tags: List[str]):
  """Return a subtree of verible_tree with the given tag."""
  if verible_tree is None:
    return []
  if not isinstance(tags, list):
    tags = [tags]
  if "tag" in verible_tree.keys():
    if verible_tree["tag"] in tags:
      return [verible_tree]
  if "children" in verible_tree.keys():
    res = []
    for c in verible_tree["children"]:
      res += find_subtree(c, tags)
    return res
  if "tree" in verible_tree.keys():
    return find_subtree(verible_tree["tree"], tags)
  return []


def get_subtree_text_info(verible_tree: dict, rtl_content: str):
  """Return tuple of form (start_pos, end_pos, text) of the subtree."""
  l = flatten_tree(verible_tree)
  start, end = l[0]["start"], l[-1]["end"]
  ret = {}
  ret["text"] = rtl_content[start:end]
  ret["start"], ret["end"] = start, end
  return ret


def get_subtree_text(verible_tree: dict, rtl_content: str):
  """Return text of the subtree."""
  return get_subtree_text_info(verible_tree, rtl_content)["text"]


def get_leftmost_node(nodes: Union[tuple, list]):
  """Get the leftmost node of a block."""
  if isinstance(nodes, tuple) or isinstance(nodes, list):
    return get_leftmost_node(nodes[0])
  return nodes


def get_rightmost_node(nodes: Union[tuple, list]):
  """Get the rightmost node of a block."""
  if isinstance(nodes, tuple) or isinstance(nodes, list):
    return get_rightmost_node(nodes[-1])
  return nodes
