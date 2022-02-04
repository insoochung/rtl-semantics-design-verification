import os
import json
from enum import Enum
from typing import List

from utils import get_indent_str


class Tag:
  MODULE = "kModuleDeclaration"
  ALWAYS = "kAlwaysStatement"
  ALWAYS_CONTENT = "kProceduralTimingControlStatement"
  ALWAYS_CONDITION = "kEventControl"
  TERNARY_EXPRESSION = "kConditionExpression"
  IF_STATEMENT = "kConditionalStatement"
  CASE_STATEMENT = "kCaseStatement"
  FOR_LOOP_STATEMENT = "kForLoopStatement"
  SEQ_BLOCK = "kSeqBlock"
  BLOCK_ITEM_LIST = "kBlockItemStatementList"

  # Assignments
  ASSIGNMENT = "kNetVariableAssignment"
  ASSIGNMENT_MODIFY = "kAssignModifyStatement"
  NON_BLOCKING_ASSIGNMENT = "kNonblockingAssignmentStatement"

  # Keywords
  BEGIN = "kBegin"
  END = "kEnd"

  # Categories
  BRANCH_STATEMENTS = [IF_STATEMENT, CASE_STATEMENT, TERNARY_EXPRESSION]
  ASSIGNMENTS = [ASSIGNMENT, ASSIGNMENT_MODIFY, NON_BLOCKING_ASSIGNMENT]


class Condition:
  TRUE = "true"
  FALSE = "false"
  DEFAULT = "default"


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


def get_subtree_text_info(verible_tree, rtl_content):
  """Return tuple of form (start_pos, end_pos, text) of the subtree."""
  l = flatten_tree(verible_tree)
  start, end = l[0]["start"], l[-1]["end"]
  ret = {}
  ret["text"] = rtl_content[start:end]
  ret["start"], ret["end"] = start, end
  return ret


def connect_nodes(node: "Node", next_node: "Node",
                  condition=Condition.DEFAULT):
  """Connect two nodes."""
  # TODO: test merging
  # if node.is_end and next_node.is_end and condition == Condition.DEFAULT:
  #   # If both nodes are arbitrary end nodes, merge them together.
  #   for p in node.prev_nodes:
  #     cond = p.remove_next_node(node)
  #     p.add_next_node(next_node, cond)
  #   del node
  # node.add_next_node(next_node, condition)
  next_node.add_prev_node(node)


def construct_assignment(verible_tree: dict, rtl_content: str,
                         ignore_inner_branches: bool = False):
  """Construct assignments from the verible tree."""
  tag = verible_tree["tag"]
  assert ignore_inner_branches or tag in Tag.ASSIGNMENTS, (
      f"Cannot construct assignment from {tag}")
  branchs = find_subtree(verible_tree, Tag.BRANCH_STATEMENTS)
  if not ignore_inner_branches and len(branchs) > 0:
    # TODO: Handle assignment with conditional statements.
    return None
    # assert 0, f"{tag} has branchs: {[b['tag'] for b in branchs]}"
  # Handle assignment without conditional statements.
  node = Node(verible_tree, rtl_content)
  print(node)
  return node


def construct_seq_block(verible_tree: dict, rtl_content: str):
  """Construct a series of nodes for a sequence block."""
  assert verible_tree["tag"] == Tag.SEQ_BLOCK
  # print_tags(verible_tree)
  assert len(verible_tree["children"]) == 3
  begin, end = verible_tree["children"][0], verible_tree["children"][2]
  assert begin["tag"] == Tag.BEGIN and end["tag"] == Tag.END, (
      f"Tags of first and last children node in sequence block should have "
      f"tags {Tag.BEGIN} and {Tag.END}, instead got {begin['tag']} and "
      f"{end['tag']}."
  )
  statement_list = verible_tree["children"][1]
  assert statement_list["tag"] == Tag.BLOCK_ITEM_LIST, (
      f"Tag of middle child of sequence block should be {Tag.BLOCK_ITEM_LIST},"
      f" instead got {statement_list['tag']}."
  )
  nodes = []
  for statement in statement_list["children"]:
    tag = statement["tag"]
    if tag == Tag.IF_STATEMENT:
      # TODO: handle if statements correctly
      nodes.append(
          construct_assignment(statement, rtl_content,
                               ignore_inner_branches=True))
    elif tag == Tag.CASE_STATEMENT:
      # TODO: handle case statements correctly
      nodes.append(
          construct_assignment(statement, rtl_content,
                               ignore_inner_branches=True))
    elif tag == Tag.FOR_LOOP_STATEMENT:
      # TODO: need to decide how to handle for loops with if statements inside
      nodes.append(
          construct_assignment(statement, rtl_content,
                               ignore_inner_branches=True))
    elif tag in Tag.ASSIGNMENTS:
      nodes.append(
          construct_assignment(statement, rtl_content,
                               ignore_inner_branches=True))
    else:
      assert False, "Unsupported statement in sequence block."

  for i, n in enumerate(nodes):  # Connect nodes
    if i == 0:
      continue
    connect_nodes(nodes[i - 1], n)

  return nodes


class RtlFile:
  """Class to manage a RTL file.

  Attributes:
  filepath -- the path to the RTL file (str)
  verible_tree -- the verible tree of the RTL file (dict)
  rtl_content -- the content of the RTL file (str)
  modules -- a list of Module objects (List(Module))
  """

  def __init__(self, verible_tree: dict, filepath: str):
    """Construct a RtlFile object from a verible_tree."""
    self.filepath = filepath
    self.verible_tree = verible_tree
    with open(self.filepath, "r") as f:
      self.rtl_content = f.read()
    self.modules = []
    self.construct_modules()

  def construct_modules(self):
    """Construct all modules found in the file."""
    module_subtrees = find_subtree(self.verible_tree, Tag.MODULE)
    assert len(module_subtrees) <= 1, (
        f"{self.filepath} has {len(module_subtrees)} module subtrees, "
        f"but there should be at most one.")
    if module_subtrees:
      self.modules = [Module(module_subtrees[0], self.rtl_content)]


class Module:
  """Class to manage a module.

  Attributes:
  rtl_content -- the content of the RTL file (str)
  verible_tree -- the verible tree of the module (dict)
  always_graphs -- a list of AlwaysGraph objects (List(AlwaysNode))
  """

  def __init__(self, verible_tree: dict, rtl_content: str):
    self.rtl_content = rtl_content
    self.verible_tree = verible_tree
    self.always_graphs = []
    self.construct_always_graphs()

  def construct_always_graphs(self):
    """Construct all graphs of always blocks found in the module."""
    always_subtrees = find_subtree(self.verible_tree, Tag.ALWAYS)
    self.always_graphs = [AlwaysNode(t, self.rtl_content)
                          for t in always_subtrees]


class Node:
  """Class to manage a node in a CDFG.

  Attributes:
  rtl_content -- the content of the RTL file (str)
  verible_tree -- the verible tree of the node (dict)
  start -- the start position of the node in the rtl_content (int)
  end -- the end position of the node in the rtl_content (int)
  text -- the text of the node (str)
  type -- the type of the node (str)
  condition -- the condition of the node, only for branch nodes (str)
  is_end -- whether the node is the end of a block (bool)
  prev_nodes -- a list of previous nodes (List(Node))
  next_nodes -- a list of next node, condition pairs (List(Node, str))
  """

  def __init__(self, verible_tree: dict = None, rtl_content: str = ""):
    self.rtl_content = rtl_content
    self.verible_tree = verible_tree
    self.start, self.end = -1, -1
    self.text = ""
    self.type = ""
    self.condition = ""
    self.is_end = False
    self.prev_nodes = []
    self.next_nodes = []
    self.construct_node()

  def __str__(self):
    return f"({self.type}): {' '.join(self.text.split())}"

  def update_text_and_type(self):
    """Update the text and type of the node."""
    assert self.verible_tree is not None, "verible_tree must be set."
    assert self.rtl_content, "rtl_content must be set."
    text_info = get_subtree_text_info(
        self.verible_tree, self.rtl_content)
    self.start, self.end, self.text = (
        text_info["start"], text_info["end"], text_info["text"])
    self.type = self.verible_tree["tag"]

  def construct_node(self):
    """Construct the node according to the verible tree."""
    self.update_text_and_type()

  def add_next_node(self, next_node: "Node",
                    next_condition: str = Condition.DEFAULT):
    """Add a next node to the node.

    Keyword arguments:
    next_condition -- the condition of the next node (str)
    """
    for n, cond in self.next_nodes:
      assert cond != next_condition, (
          f"Node already has a next node with condition '{next_condition}' "
          f"leading to {n}.")
    self.next_nodes.append(next_node, next_condition)

  def add_prev_node(self, prev_node: "Node"):
    """Add a previous node to the node.
    """
    if prev_node not in self.prev_nodes:
      self.prev_nodes.append(prev_node)

  def remove_next_node(self, next_node: "Node"):
    """Remove a next node, and return the condition of the removed node"""
    idx = -1
    for i, (node, cond) in enumerate(self.next_nodes):
      if node == next_node:
        idx = i
        break
    assert idx > i, f"Node '{next_node}' not found in next_nodes."
    self.next_nodes.pop(idx)

  def remove_prev_node(self, prev_node: "Node"):
    """Remove a previous node"""
    self.prev_nodes.remove(prev_node)


class AlwaysNode(Node):
  """Class to manage a graph of a always block."""

  def update_text_and_type(self):
    super().update_text_and_type()
    self.type = self.verible_tree["children"][0]["tag"]
    assert "always" in self.type, (
        f"{self.type} is not a inspected type of node.")

  def construct_node(self):
    """Construct the node according to the verible tree."""
    self.update_text_and_type()
    children = self.verible_tree["children"]
    assert len(children) == 2
    if self.type in ["always_ff", "always"]:
      content = children[1]["children"]
      assert len(content) == 2
      condition, seq_block = content[0], content[1]
      assert condition["tag"] == Tag.ALWAYS_CONDITION
      self.condition = get_subtree_text_info(
          condition, self.rtl_content)["text"]
    else:
      assert self.type in ["always_comb", "always_latch"], (
          f"Unknown '{self.type}' type.")
      seq_block = children[1]
    assert seq_block["tag"] == Tag.SEQ_BLOCK
    nodes = construct_seq_block(seq_block, self.rtl_content)
    assert nodes, "Seq block is empty."
    self.end_node = EndNode()  # Arbitrary end node.
    connect_nodes(self, nodes[0])
    connect_nodes(nodes[-1], self.end_node)

  def print_graph(self, indent=0):
    """Print the graph of the always block."""
    print(f"{self.type} {self.condition}")
    for node in self.next_nodes.values():
      print(f"{node}")


class EndNode(Node):
  """Node class that specifies an arbitrary end"""

  def construct_node(self):
    self.is_end = True
    self.type = "end"
