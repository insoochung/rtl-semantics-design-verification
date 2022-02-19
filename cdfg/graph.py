from typing import List, Tuple

from constants import Tag, Condition
from utils import (get_indent_str,
                   preprocess_rtl_str,
                   find_subtree,
                   get_subtree_text_info,
                   get_branch_condition_tree,
                   get_case_item_tree,
                   get_symbol_identifiers_in_tree)


class Node:
  """Class to manage a node in a CDFG.

  Attributes:
  rtl_content -- the content of the RTL file (str)
  verible_tree -- the verible tree of the node (dict)
  start -- the start position of the node in the rtl_content (int)
  end -- the end position of the node in the rtl_content (int)
  line_num -- the line number of the node's statement in the rtl_content (int)
  text -- the text of the node (str)
  type -- the type of the node (str)
  condition -- the condition of the node, only for branch nodes (str)
  prev_nodes -- a list of previous nodes (List(Node))
  next_nodes -- a list of next node, condition pairs (List(Node, str))
  block_depth -- the depth of the block the node is in (int)
  end_node -- the end node of the block, exists only if this is a start node
              of a block (Node)
  condition_vars -- a set of variables that are used in the condition of the
                    node and its descendents (Set(str))
  assigned_vars -- a set of variables that are assigned in the node and its
                   descendents (Set(str))
  """

  def __init__(self, verible_tree: dict = None, rtl_content: str = "",
               block_depth: int = 0):
    self.rtl_content = rtl_content
    self.verible_tree = verible_tree
    self.start, self.end = -1, -1
    self.line_num = -1
    self.text = ""
    self.type = ""
    self.condition = ""
    self.lead_condition = ""
    self.prev_nodes = []
    self.next_nodes = []
    self.block_depth = block_depth
    self.end_node = None
    if verible_tree is not None:
      self.update_text_and_type()
      self.update_line_num()
    self.assigned_vars = set()
    self.condition_vars = set()

  def __str__(self):
    prefix = self.type
    prefix += f" @{self.get_id()}"
    if self.condition:
      prefix += f" / cond.: {self.condition}"
    if self.lead_condition:
      prefix += f" / lead cond.: {self.lead_condition}"
    if self.condition_vars:
      prefix += f" / cond. vars: {self.condition_vars}"
    if self.assigned_vars:
      prefix += f" / assigned vars: {self.assigned_vars}"
    if self.prev_nodes:
      prefix += f" / fan_in: {[n.get_id() for n in self.prev_nodes]}"
    if self.next_nodes:
      prefix += (f" / fan_out: "
                 f"{[(n[0].get_id(), n[1]) for n in self.next_nodes]}")
    s = self.get_one_line_str()
    if s and "always" not in self.type:
      return f"({prefix}): {s}"
    else:
      return f"({prefix})"

  def get_id(self):
    if self.line_num > 0:
      return f"L{self.line_num}"
    else:
      return f"A{str(id(self))[-4:]}"

  def get_one_line_str(self):
    ret = preprocess_rtl_str(self.text, one_line=True)
    return ret

  def update_text_and_type(self, force_start=None, force_end=None):
    """Update the text and type of the node."""
    assert self.verible_tree is not None, "verible_tree must be set."
    assert self.rtl_content, "rtl_content must be set."
    text_info = get_subtree_text_info(
        self.verible_tree, self.rtl_content)
    self.start, self.end, self.text = (
        text_info["start"], text_info["end"], text_info["text"])
    if force_start:
      self.start = force_start
    if force_end:
      self.end = force_end
    if force_start or force_end:
      self.text = self.rtl_content[self.start:self.end]
    self.type = self.verible_tree["tag"]

  def update_line_num(self):
    """Update the line number of the node."""
    assert self.rtl_content, "rtl_content must be set."
    self.line_num = self.rtl_content[:self.start].count("\n") + 1

  def update_text(self):
    self.text = self.rtl_content[self.start:self.end]

  def set_start(self, start: int):
    """Set the start of the node"""
    self.start = start
    self.update_text()

  def set_end(self, end: int):
    """Set the end of the node"""
    self.end = end
    self.update_text()

  def add_next_node(self, next_node: "Node",
                    next_condition: str = None):
    """Add a next node to the node.

    Keyword arguments:
    next_condition -- the condition of the next node (str)
    """
    if next_condition != Condition.DATA:
      # Only data edges are allowed to have multiple non-unique connections
      for n, cond in self.next_nodes:
        assert cond != next_condition, (
            f"Tried to add connection -> {cond}: {next_node}. "
            f"Connection {self.type} -> {cond}: {next_condition} already "
            f"exists.")
    self.next_nodes.append((next_node, next_condition))
    if isinstance(next_condition, list) and len(next_condition) == 1:
      next_condition = next_condition[0]
    if next_condition is None:  # Procedural connection
      return
    elif next_condition == Condition.DEFAULT:  # Default connection
      next_node.lead_condition = Condition.DEFAULT
      return
    elif next_condition == Condition.DATA:  # Connected by a data edge.
      next_node.lead_condition = Condition.DATA
      return

    assert self.condition, (
        f"{self} has no condition, tried to add {next_condition}->{next_node}")
    if next_condition == Condition.TRUE:
      next_node.lead_condition = self.condition
    elif next_condition == Condition.FALSE:
      next_node.lead_condition = f"!({self.condition})"
    else:
      if not isinstance(next_condition, list):
        next_node.lead_condition = f"{self.condition} == {next_condition}"
      else:  # If multiple conditions are possible, aggregate them using OR.
        next_conditions = [
            f"{self.condition} == {c}" for c in next_condition]
        next_node.lead_condition = f"({' || '.join(next_conditions)})"

  def add_prev_node(self, prev_node: "Node"):
    """Add a previous node to the node."""
    if prev_node not in self.prev_nodes:
      self.prev_nodes.append(prev_node)

  def remove_next_node(self, next_node: "Node"):
    """Remove a next node, and return the condition of the removed node"""
    idx = -1
    conds = []
    for i, (node, cond) in enumerate(self.next_nodes):
      if node == next_node:
        idx = i
        self.next_nodes.pop(idx)
        conds.append(cond)
    assert idx > -1, f"Node '{next_node}' not found in next_nodes."
    return conds

  def remove_prev_node(self, prev_node: "Node"):
    """Remove a previous node"""
    self.prev_nodes.remove(prev_node)

  def to_list(self, conditions: List[Tuple["Node", str]] = []):
    """Return the list of nodes in the path starting from this node to end node

    Keyword arguments:
    conditions -- the list of pairs of branch node and condition, whenver
                  a branch node is encountered, the condition is utilized for
                  traversal (List[Tuple[Node, str]])
    """
    ret = [self]
    conditions = list(conditions)

    next_nodes = self.next_nodes
    for node, condition in conditions:
      if condition and node == self:
        # If one of the conditions is relevant, only traverse that node
        next_nodes = [n for n in next_nodes if n[1] == condition]
        assert len(
            next_nodes) == 1, (
            f"Can't find condition '{condition}' from next node conditions: "
            f"{[(n[1]) for n in self.next_nodes]}")
        break

    # Traverse the next nodes.
    for n, _ in next_nodes:
      if n.block_depth > self.block_depth and n != self.end_node:
        ret.extend(n.to_list(conditions))
        while (len(ret[-1].next_nodes) == 1
               and ret[-1].next_nodes[0][0].block_depth > self.block_depth):
          next_n, _ = ret[-1].next_nodes[0]
          ret.extend(next_n.to_list(conditions))
          if (next_n == self.end_node):
            break

    if self.end_node:
      ret.extend(self.end_node.to_list(conditions))
    return ret

  def print_block(self):
    """Print the block starting from this node."""
    l = self.to_list()
    print()
    for n in l:
      print(get_indent_str(n.block_depth * 2) + str(n))
    print()

  def update_next_node_conditions(self):
    """Update the conditions of the next nodes."""
    for i, (n, cond) in enumerate(self.next_nodes):
      if cond is None:
        continue
      if not isinstance(cond, list):
        cond = [cond]
      for cond_i, _cond in enumerate(cond):
        if "'b" in _cond:  # Remove dummy underscores in numbers
          cond[cond_i] = _cond.replace("_", "")
      self.next_nodes[i] = (n, " ".join(cond))


class EndNode(Node):
  """Node class that specifies an arbitrary end"""

  def __init__(self, verible_tree: dict = None, rtl_content: str = "",
               block_depth: int = 0):
    super().__init__(verible_tree=verible_tree,
                     rtl_content=rtl_content, block_depth=block_depth)
    self.type = "end"


class BranchNode(Node):
  """Node class that specifies a branching point"""

  def __init__(self, verible_tree: dict = None, rtl_content: str = "",
               block_depth: int = 0):
    super().__init__(verible_tree=verible_tree,
                     rtl_content=rtl_content, block_depth=block_depth)
    self.update_condition()

  def update_condition(self):
    """Update the condition of the node if it is a branch node."""
    condition_tree = get_branch_condition_tree(self.verible_tree)
    text_info = get_subtree_text_info(
        condition_tree, self.rtl_content)
    self.condition = " ".join(text_info["text"].split())
    self.update_text_and_type(force_end=text_info["end"])


class AlwaysNode(Node):
  """Class to manage a graph of a always block."""

  def update_text_and_type(self, force_start=None, force_end=None):
    super().update_text_and_type(force_start=force_start, force_end=force_end)
    self.type = self.verible_tree["children"][0]["tag"]
    assert "always" in self.type, (
        f"{self.type} is not a inspected type of node.")

  def update_condition_vars(self):
    """Find and track the vars used in descendent condition statements"""
    conditional_subtrees = find_subtree(self.verible_tree,
                                        Tag.CONDITION_STATEMENTS)
    ids = set()
    for subtree in conditional_subtrees:
      condition_tree = get_branch_condition_tree(subtree)
      ids |= get_symbol_identifiers_in_tree(condition_tree, self.rtl_content)

      if subtree["tag"] == Tag.CASE_STATEMENT:
        # If the condition is a case statement, see if case items contain
        # variables.
        for case_item in get_case_item_tree(subtree)["children"]:
          expr_list = case_item["children"][0]
          assert expr_list["tag"] in [Tag.EXPRESSION_LIST, Tag.DEFAULT], (
              f"{expr_list['tag']} is not an expected type of node.")
          ids |= get_symbol_identifiers_in_tree(expr_list, self.rtl_content)
    self.condition_vars = ids

  def update_assigned_vars(self):
    """Find and update the vars assigned within always."""
    assign_subtrees = find_subtree(self.verible_tree, Tag.ASSIGNMENTS)
    ids = set()
    for subtree in assign_subtrees:
      lhs_subtree = subtree["children"][0]
      assert lhs_subtree["tag"] == Tag.LVALUE, (
          f"{lhs_subtree['tag']} is not an expected type of node.")
      ids |= get_symbol_identifiers_in_tree(lhs_subtree, self.rtl_content)
    self.assigned_vars = ids
