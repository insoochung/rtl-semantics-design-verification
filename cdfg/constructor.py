import re
import os
import sys
import argparse
import pickle
from typing import Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constants import Tag, Condition
from cdfg.graph import (Node, BranchNode, EndNode, AlwaysNode,
                        DummyNode)
from cdfg.parser import get_verible_parsed_rtl
from cdfg.utils import (preprocess_rtl_str,
                        find_subtree,
                        get_subtree_text,
                        get_leftmost_node,
                        get_rightmost_node,
                        get_case_item_tree)



def _get_start_end_node(nodes: Union[tuple, list]):
  """Get real start and end nodes of a block."""
  assert len(nodes)
  start_node = get_leftmost_node(nodes)
  end_node = get_rightmost_node(nodes)
  assert isinstance(start_node, Node) and isinstance(end_node, Node)
  return start_node, end_node


def connect_nodes(node: "Node", next_node: "Node",
                  condition=None):
  """Connect two nodes."""
  node = get_rightmost_node(node)
  next_node = get_leftmost_node(next_node)
  # TODO: merge redundant nodes.
  next_node.add_prev_node(node)
  node.add_next_node(next_node, condition)


def construct_statement(verible_tree: dict, rtl_content: str,
                        block_depth: int = 0):
  if (is_ternary_assignment(verible_tree)
          or is_ternary_expression(verible_tree)):
    return construct_statement_with_ternary_cond(verible_tree, rtl_content,
                                                 block_depth=block_depth)
  else:
    return construct_terminal(verible_tree, rtl_content,
                              block_depth=block_depth)


def construct_terminal(verible_tree: dict, rtl_content: str,
                       block_depth: int = 0):
  """Construct assignments from the verible tree."""
  tag = verible_tree["tag"]
  assert tag in Tag.TERMINAL_STATEMENTS + Tag.EXPRESSIONS, (
      f"Cannot construct assignment from "
      f"({get_subtree_text(verible_tree, rtl_content)} / {tag})")
  # Handle assignment without conditional statements.
  node = Node(verible_tree, rtl_content, block_depth=block_depth)
  return node


def construct_if_else_statement(verible_tree: dict, rtl_content: str,
                                block_depth: int = 0):
  """Construct if-else statements from the verible tree."""
  tag = verible_tree["tag"]
  children = verible_tree["children"]
  assert len(children) <= 2, (
      f"{tag} has more than two children: {[c['tag'] for c in children]}.\n"
      f"An if clause and/or an else clause is expected.")
  # Construct branch node.
  nodes = []
  if_clause = children[0]
  assert if_clause["tag"] == Tag.IF_CLAUSE
  assert len(if_clause["children"]) == 2
  branch_node = if_clause["children"][0]
  assert branch_node["tag"] == Tag.IF_HEADER
  branch_node = BranchNode(branch_node, rtl_content, block_depth=block_depth)
  nodes.append(branch_node)
  # Construct if-body node.
  if_body_node = if_clause["children"][1]
  assert if_body_node["tag"] == Tag.IF_BODY
  assert len(if_body_node["children"]) == 1
  block_node = if_body_node["children"][0]
  assert block_node["tag"] in Tag.TERMINAL_STATEMENTS + Tag.BLOCK_STATEMENTS, (
      f"{block_node['tag']} is not a terminal statement or sequence block.")
  if block_node["tag"] in Tag.BLOCK_STATEMENTS:
    if_nodes = construct_block(
        block_node, rtl_content, block_depth=block_depth + 1)
  else:  # Tag.TERMINAL_STATEMENTS
    if_node = construct_statement(block_node, rtl_content,
                                  block_depth=block_depth + 1)
    if_nodes = [if_node]

  # Construct an end node:
  branch_node.end_node = end_node = EndNode(block_depth=block_depth)
  # Connect if-body node to branch node.
  connect_nodes(branch_node, if_nodes[0], condition=Condition.TRUE)
  # Connect if-body node to end node.
  connect_nodes(if_nodes[-1], end_node)
  nodes += if_nodes

  # Construct else-body node.
  if len(children) == 2:
    else_clause = children[1]
    assert else_clause["tag"] == Tag.ELSE_CLAUSE
    assert len(else_clause["children"]) == 2
    assert else_clause["children"][1]["tag"] == Tag.ELSE_BODY
    else_body_node = else_clause["children"][1]
    assert len(else_body_node["children"]) == 1, f"{else_body_node}"
    block_node = else_body_node["children"][0]
    assert block_node["tag"] in Tag.TERMINAL_STATEMENTS + Tag.BLOCK_STATEMENTS, (
        f"{block_node['tag']} is not a terminal statement or sequence block.")
    if block_node["tag"] in Tag.BLOCK_STATEMENTS:
      else_nodes = construct_block(
          block_node, rtl_content, block_depth=block_depth + 1)
    else:  # Tag.TERMINAL_STATEMENTS
      else_node = construct_statement(block_node, rtl_content,
                                      block_depth=block_depth + 1)
      else_nodes = [else_node]

    # Connect else-body node to branch node.
    connect_nodes(branch_node, else_nodes[0], condition=Condition.FALSE)
    # Connect else-body node to end node.
    connect_nodes(else_nodes[-1], end_node)
    nodes += else_nodes
  else:  # If there is no else clause, connect the branch node to the end node.
    connect_nodes(branch_node, end_node, condition=Condition.FALSE)
  nodes.append(end_node)
  return nodes


def construct_case_statement(verible_tree: dict, rtl_content: str,
                             block_depth: int = 0):
  """Construct case statement from verible tree"""
  tag = verible_tree["tag"]
  children = verible_tree["children"]
  assert len(children) == 5
  children_tags = [None if c is None else c["tag"] for c in children]
  assert children_tags[0] in [Tag.UNIQUE, None], children_tags
  assert children_tags[1] in [Tag.CASE, Tag.CASEZ], children_tags
  assert children_tags[2] == Tag.PARENTHESIS_GROUP, children_tags
  assert children_tags[3] == Tag.CASE_ITEM_LIST, children_tags
  assert children_tags[4] == Tag.ENDCASE, children_tags
  # Construct branch node.
  branch_node = BranchNode(verible_tree=verible_tree, rtl_content=rtl_content,
                           block_depth=block_depth)
  # Construct case-item-list node.
  default_node = None
  nodes = []
  conditions_list = []
  for case_item in get_case_item_tree(verible_tree)["children"]:
    children = case_item["children"]
    children_tags = [c["tag"] for c in children]
    assert len(children) == 3
    assert children_tags[0] in [Tag.DEFAULT, Tag.EXPRESSION_LIST]
    assert children_tags[1] == Tag.COLON
    assert children_tags[2] in Tag.BLOCK_STATEMENTS + Tag.TERMINAL_STATEMENTS
    # Construct case-item-list node.
    node = children[2]
    if children_tags[2] in Tag.BLOCK_STATEMENTS:
      node = construct_block(node, rtl_content, block_depth=block_depth + 1)
    else:  # Tag.TERMINAL_STATEMENTS
      node = construct_statement(
          node, rtl_content, block_depth=block_depth + 1)
    # Connect case-item-list node to branch node.
    # Process case conditions
    condition = get_subtree_text(children[0], rtl_content)
    condition = preprocess_rtl_str(condition, no_space=True)
    conditions = re.split(r',\s*(?![^{}]*\})', condition)
    nodes.append(node)
    conditions_list.append(conditions)
    if children_tags[0] == Tag.DEFAULT:
      assert default_node is None, "Multiple default cases"
      default_node = get_leftmost_node(node)
  # Construct an end node:
  branch_node.end_node = end_node = EndNode(block_depth=block_depth)
  for cond, node in zip(conditions_list, nodes):
    # Connect case-item-list node to branch node and end node.
    connect_nodes(branch_node, node, condition=cond)
    connect_nodes(node, end_node)

  if default_node:
    # Elaborate default node lead condition.
    assert default_node.lead_condition == Condition.DEFAULT
    non_default_conds = []
    for n in nodes:
      n = get_leftmost_node(n)
      if n == default_node:
        continue
      non_default_conds.append(n.lead_condition)
    default_node.lead_condition = f"!({' || '.join(non_default_conds)})"
  else:
    # Connect branch node to end node.
    connect_nodes(branch_node, end_node)

  return branch_node, end_node


def construct_for_loop_statement(verible_tree: dict, rtl_content: str,
                                 block_depth: int = 0):
  """Construct for loop statement from verible tree"""
  tag = verible_tree["tag"]
  assert tag == Tag.FOR_LOOP_STATEMENT
  children = verible_tree["children"]
  assert len(children) == 2
  assert children[0]["tag"] == Tag.FOR_LOOP_CONDITION
  assert children[1]["tag"] == Tag.SEQ_BLOCK
  # Construct start_node
  start_node = Node(verible_tree=verible_tree, rtl_content=rtl_content,
                    block_depth=block_depth)
  # Construct sequence block nodes.
  nodes = construct_block(children[1], rtl_content,
                          block_depth=block_depth + 1)
  # Connect sequence block nodes to start node.
  connect_nodes(start_node, nodes[0])
  return start_node, nodes[-1]


def is_ternary_assignment(verible_tree: dict):
  """Check if the verible tree is ternary assignment"""
  return (verible_tree["tag"] in Tag.ASSIGNMENTS
          and find_subtree(verible_tree, Tag.TERNARY_EXPRESSION))


def is_ternary_expression(verible_tree: dict):
  """Check if the verible tree is ternary expression"""
  return (verible_tree["tag"] in Tag.EXPRESSIONS
          and find_subtree(verible_tree, Tag.TERNARY_EXPRESSION))


def construct_statement_with_ternary_cond(verible_tree: dict, rtl_content: str,
                                          block_depth: int = 0):
  """Construct ternary assignment from verible tree"""
  assert (is_ternary_assignment(verible_tree)
          or is_ternary_expression(verible_tree))
  tag = verible_tree["tag"]
  all_ternary_trees = find_subtree(verible_tree, Tag.TERNARY_EXPRESSION)
  assert len(all_ternary_trees) == 1, (
      f"{get_subtree_text(verible_tree, rtl_content)}\n"
      f"-> Multiple ternary expressions within one assignment not "
      f"handled.")

  # Construct ternary expression.
  ternary_tree = all_ternary_trees[0]
  start_node = Node(verible_tree, rtl_content=rtl_content,
                    block_depth=block_depth)
  branch_node = BranchNode(verible_tree=ternary_tree, rtl_content=rtl_content,
                           block_depth=block_depth)
  trees = {
      Condition.TRUE: ternary_tree["children"][2],
      Condition.FALSE: ternary_tree["children"][4]
  }
  nodes = {}
  for cond, tree in trees.items():
    nodes[cond] = construct_statement(tree, rtl_content,
                                      block_depth=block_depth + 1)

  # Construct start and end nodes.
  end_node = Node(verible_tree, rtl_content=rtl_content,
                  block_depth=block_depth)

  # Connect them all together.
  connect_nodes(start_node, branch_node)
  connect_nodes(branch_node, nodes[Condition.TRUE], condition=Condition.TRUE)
  connect_nodes(branch_node, nodes[Condition.FALSE], condition=Condition.FALSE)
  connect_nodes(nodes[Condition.TRUE], end_node)
  connect_nodes(nodes[Condition.FALSE], end_node)

  # Update text to have no overlaps
  start_node.set_end(branch_node.start - 1)
  end_node_start = max([p.end for p in end_node.prev_nodes] + [start_node.end])
  end_node.set_start(end_node_start)

  return start_node, end_node


def construct_block(verible_tree: dict, rtl_content: str,
                    block_depth: int = 0):
  """Construct block from the verible tree."""
  tag = verible_tree["tag"]
  assert tag in Tag.BLOCK_STATEMENTS, f"{tag} is not a block statement"

  if tag == Tag.SEQ_BLOCK:
    return construct_seq_block(verible_tree, rtl_content,
                               block_depth=block_depth)
  elif tag == Tag.IF_ELSE_STATEMENT:
    return construct_if_else_statement(verible_tree, rtl_content,
                                       block_depth=block_depth)
  elif tag == Tag.CASE_STATEMENT:
    return construct_case_statement(verible_tree, rtl_content,
                                    block_depth=block_depth)
  elif tag == Tag.FOR_LOOP_STATEMENT:
    return construct_for_loop_statement(verible_tree, rtl_content,
                                        block_depth=block_depth)
  else:
    assert 0, f"Cannot construct block from {tag}, not implemented."


def construct_seq_block(verible_tree: dict, rtl_content: str,
                        block_depth: int = 0):
  """Construct a series of nodes for a sequence block."""
  assert verible_tree["tag"] == Tag.SEQ_BLOCK
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
    if tag in Tag.BRANCH_STATEMENTS:
      new_nodes = construct_block(statement, rtl_content,
                                  block_depth=block_depth)

    elif tag == Tag.FOR_LOOP_STATEMENT:
      new_nodes = construct_block(statement, rtl_content,
                                  block_depth=block_depth)
    elif tag in Tag.TERMINAL_STATEMENTS:
      new_nodes = construct_statement(statement, rtl_content,
                                      block_depth=block_depth)
    else:
      assert False, (
          f"Unsupported statement "
          f"'{get_subtree_text(statement, rtl_content)}' "
          f"in sequence block.")

    nodes.append(new_nodes)
  for i, n in enumerate(nodes):  # Connect nodes
    if i == 0:
      continue
    connect_nodes(nodes[i - 1], n)
  if not nodes:
    return [DummyNode(block_depth=block_depth)]

  return _get_start_end_node(nodes)  # start_node, end_node


def construct_always_node(verible_tree: dict, rtl_content: str, block_depth: int = 0):
  """Construct always node and its children nodes."""
  always_node = AlwaysNode(verible_tree, rtl_content, block_depth)
  children = always_node.verible_tree["children"]
  assert len(children) == 2
  if always_node.type in ["always_ff", "always"]:
    content = children[1]["children"]
    assert len(content) == 2
    condition, body = content[0], content[1]
    assert condition["tag"] == Tag.ALWAYS_CONDITION
    always_node.condition = get_subtree_text(
        condition, always_node.rtl_content)
  else:
    assert always_node.type in ["always_comb", "always_latch"], (
        f"Unknown '{always_node.type}' type.")
    body = children[1]

  if body["tag"] in Tag.BLOCK_STATEMENTS:
    body_nodes = construct_block(
        body, always_node.rtl_content,
        block_depth=always_node.block_depth + 1)
    assert body_nodes, "Seq block is empty."
    # Arbitrary end always_node.
  else:
    body_nodes = construct_statement(body, always_node.rtl_content,
                                     block_depth=always_node.block_depth + 1)
    if not isinstance(body_nodes, list):
      body_nodes = [body_nodes]

  always_node.end_node = EndNode(block_depth=always_node.block_depth)
  connect_nodes(always_node, body_nodes[0])
  connect_nodes(body_nodes[-1], always_node.end_node)
  # Loop back to the start node.
  connect_nodes(always_node.end_node, always_node)

  # Post process the always node.
  always_node.update_condition_vars()
  always_node.update_assigned_vars()
  always_node.print_block()

  return always_node


def construct_design_graph(parsed_rtl_dir, rtl_dir, output_dir):
  design_graph = DesignGraph(parsed_rtl_dir, rtl_dir)
  os.makedirs(output_dir, exist_ok=True)
  pkl_name = os.path.join(output_dir, "design_graph.pkl")
  with open(pkl_name, "wb") as f:
    pickle.dump(design_graph, f)
  print("Saved design graph to {}".format(pkl_name))


class DesignGraph:
  """Class to manage CDFGs in a design"""

  def __init__(self, parsed_rtl_dir: str, rtl_dir: str):
    self.parsed_rtl_dir = parsed_rtl_dir
    self.rtl_dir = rtl_dir
    self.construct_rtl_files()
    self.postprocess()

  def construct_rtl_files(self):
    """Construct RtlFile objects from parsed RTL files."""
    parsed_rtl = get_verible_parsed_rtl(self.parsed_rtl_dir,
                                        orig_dir=self.rtl_dir)
    self.rtl_files = []
    for filepath, verible_tree in parsed_rtl.items():
      filename = os.path.basename(filepath)
      print(f"-- Constructing CDFGs from: {filepath} --")
      rtl_file_obj = RtlFile(verible_tree, filepath)
      if rtl_file_obj.num_always_blocks == 0:
        print(f"-- Skipping {filename} because it has no always blocks --\n")
        del rtl_file_obj
        continue
      self.rtl_files.append(rtl_file_obj)
    print(
        f"-- CDFGs successfully constructed from design {self.rtl_dir}! --\n")

  def postprocess(self):
    """Postprocess DesignGraph the after RtlFile objects are created."""
    # Line up all nodes within the design graph
    self.nodes = []
    self.node_to_index = {}
    idx_offset = 0
    for rtl_file in self.rtl_files:
      nodes = rtl_file.nodes
      self.nodes.extend(nodes)
      for i, n in enumerate(nodes):
        self.node_to_index[n] = i + idx_offset
      idx_offset = len(self.nodes)


class RtlFile:
  """Class to manage a RTL file.

  Attributes:
  filepath -- the path to the RTL file (str)
  verible_tree -- the verible tree of the RTL file (dict)
  rtl_content -- the content of the RTL file (str)
  modules -- a list of Module objects (List(Module))
  num_always_blocks -- sum of always blocks in all modules (int)
  """

  def __init__(self, verible_tree: dict, filepath: str):
    """Construct a RtlFile object from a verible_tree."""
    self.filepath = filepath
    self.verible_tree = verible_tree
    with open(self.filepath, "r") as f:
      self.rtl_content = f.read()
    self.modules = []
    self.construct_modules()
    self.postprocess()

  def construct_modules(self):
    """Construct all modules found in the file."""
    module_subtrees = find_subtree(self.verible_tree, Tag.MODULE)
    if len(module_subtrees) > 1:
      print(
          f"{self.filepath} has {len(module_subtrees)}(!= 1) module subtrees, "
          f"this may not work with other components of this tool.")
    if module_subtrees:
      self.modules = [Module(mst, self.rtl_content) for mst in module_subtrees]

  def postprocess(self):
    """Post-process the RtlFile object after all modules are constructed."""
    self.num_always_blocks = sum([len(m.always_graphs) for m in self.modules])
    # Line up all nodes within RTL module
    self.nodes = []
    for m in self.modules:
      self.nodes.extend(m.to_list())
    for n in self.nodes:  # Postprocess next_node conditions
      n.update_next_node_conditions()
    # Create line number to nodes mapping
    self.line_number_to_nodes = {}
    self.node_to_index = {}
    for i, n in enumerate(self.nodes):
      line_num = n.line_num
      if line_num not in self.line_number_to_nodes:
        self.line_number_to_nodes[line_num] = []
      self.line_number_to_nodes[line_num].append(n)
      self.node_to_index[n] = i


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
    # TODO: Construct continuous assignments outside of always blocks.

  def construct_always_graphs(self):
    """Construct all graphs of always blocks found in the module."""
    always_subtrees = find_subtree(self.verible_tree, Tag.ALWAYS)
    for t in always_subtrees:
      always_node = construct_always_node(t, self.rtl_content)
      self.always_graphs.append(always_node)
    num_always = len(self.always_graphs)

    # Connect data edges
    for i in range(num_always):
      for j in range(i + 1, num_always):
        x, y = self.always_graphs[i], self.always_graphs[j]
        if x.assigned_vars & y.condition_vars:
          connect_nodes(x.end_node, y, condition=Condition.DATA)
        if x.condition_vars & y.assigned_vars:
          connect_nodes(y.end_node, x, condition=Condition.DATA)

  def to_list(self):
    """Return a list of all nodes in the module."""
    ret = []
    for g in self.always_graphs:
      ret += g.to_list()
    return ret


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-pr", "--parsed_rtl_dir", required=True,
                      help=("Directory where verible generated ASTs "
                            "(parsed from RTLs) are located in json format"))
  parser.add_argument("-rd", "--rtl_dir", required=True,
                      help="Directory where the original RTL files are "
                           "located")
  parser.add_argument("-od", "--output_dir", default="generated/cdfgs",
                      help="Directory where parsed CDFGs are saved")
  args = parser.parse_args()
  construct_design_graph(**vars(args))
