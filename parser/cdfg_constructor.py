# from lark.visitors import Vistor

import copy
from os import terminal_size
from re import A
from lark.tree import Tree
from lark.reconstruct import Reconstructor, is_id_continue

TERMINAL_NODES = ["statement", "statement_or_null", "case_condition"]
BRANCH_NODES = ["if_else_statement", "case_statement"]
PROCEDURAL_NODES = ["seq_block"]
PROMOTION_REQUIRED = ["if_else_statement", "case_statement", "case_item", "always_block"]

class CdfgNodePair:
  def __init__(self, start_node, end_node):
    self.start_node = start_node
    self.end_node = end_node
  
  def append_prev_nodes(self, prev_node):
    self.start_node.append_prev_nodes(prev_node)
  
  def append_next_nodes(self, next_node):
    self.end_node.append_next_nodes(next_node)
  
  def __str__(self):
    return str(self.start_node) + "\n" + str(self.end_node)

  def to_list_topdown(self):
    return self.start_node.to_list_topdown() + self.end_node.to_list_topdown()

class CdfgNode:
  def __init__(self, iterable=(), **kwargs):
    self.__dict__.update(iterable, **kwargs)
    self.prev_nodes = []
    self.next_nodes = []

  def append_prev_nodes(self, prev_node):
    if isinstance(prev_node, CdfgNode): 
      self.prev_nodes.append(prev_node)
    elif isinstance(prev_node, CdfgNodePair):
      self.prev_nodes.append(prev_node.end_node)
    else:
      assert False, f"{prev_node} is not a CdfgNode or CdfgNodePair"
  
  def append_next_nodes(self, next_node):
    if isinstance(next_node, CdfgNode): 
      self.next_nodes.append(next_node)
    elif isinstance(next_node, CdfgNodePair):
      self.next_nodes.append(next_node.start_node)
    else:
      assert False, f"{next_node} is not a CdfgNode or CdfgNodePair"
  
  def to_list_topdown(self):
    ret = [self]
    for node in self.children:
      ret.extend(node.to_list_topdown())
    return ret
  
  def set_node_num(self, n):
    self.node_num = n

  def __str__(self):
    return (f"{get_indent_str(self.indent)}{self.partial_str} "
            "// {" 
            f"\"node_num\": {self.node_num}, "
            f"\"type\": \"{self.type}\", "
            f"\"prev_nodes\": {[x.node_num for x in self.prev_nodes]}, "
            f"\"next_nodes\": {[x.node_num for x in self.next_nodes]}"
            "}")

def is_terminal(lark_tree):
  if not lark_tree.children:
    return True
  
  if lark_tree.data not in TERMINAL_NODES:
    return False
  for d in TERMINAL_NODES:
    for c in lark_tree.children:
      if list(c.find_data(d)) != []:
        return False
  return True

def require_promotion(lark_tree):
  return lark_tree.data in PROMOTION_REQUIRED

def get_indent_str(indent):
    return " " * indent

def is_same_as_only_child(tree):
  if len(tree.children) == 1:
    m = tree.meta
    only_child = tree.children[0] 
    if isinstance(only_child, Tree):
      child_m = only_child.meta
      if m.start_pos == child_m.start_pos and m.end_pos == child_m.end_pos:
        return True
  return False

def connect_cdfg(prev, next):
  prev.append_next_nodes(next)
  next.append_prev_nodes(prev)

def maybe_promote_empty_node(node):
  if (node.partial_str.strip() != ""
      and len(node.prev_nodes) == 1
      and len(node.next_nodes) == 1):
    return node

def get_partial_str(s, start_pos, end_pos):
    return s[start_pos:end_pos].strip()

def get_cdfg(always_str, lark_tree, indent=0, prepend_type=None):
  if not isinstance(lark_tree, Tree):
    return None
  lark_tree_type = lark_tree.data.strip()
  if prepend_type:
    lark_tree_type = f"{prepend_type},{lark_tree_type}"
    
  if is_same_as_only_child(lark_tree):
    # If the tree is identical to its only child, return the child with type signatures of its ancestors.
    return get_cdfg(always_str, lark_tree.children[0], indent, prepend_type=lark_tree_type)

  m = lark_tree.meta
  start_pos = m.start_pos
  end_pos = m.end_pos
  init_data = {"full_str": always_str,
               "partial_str": get_partial_str(always_str, start_pos, end_pos),
               "lark_tree": lark_tree,
               "lark_meta": m,
               "indent": indent,
               "start_pos": start_pos,
               "end_pos": end_pos,
               "is_terminal": is_terminal(lark_tree),
               "type": lark_tree_type,
               "children": [],}

  if init_data["is_terminal"]:
    # Return a simple node if it's terminal.
    return CdfgNode(init_data)

  # Otherwise, construct a pair of start-end nodes with their children.
  # Some children nodes (e.g. condition node in if_else_statement) need to be promoted to its parent node.
  _children = lark_tree.children
  cutoff_idx = 0
  while cutoff_idx < len(_children) and not isinstance(_children[cutoff_idx], Tree):
    # Token children are automatically promoted to the parent node.
    cutoff_idx += 1
  _children = _children[cutoff_idx:]
  if require_promotion(lark_tree):
    if "case_statement" in lark_tree_type:
      # If the node is a case statement, promote the condition node to its parent node.
      if _children[0].data == "condition":
        _children = _children[1:]
    elif "if_else_statement" in lark_tree_type:
      if _children[0].data == "condition":
        _children = _children[1:]
    elif "case_item" in lark_tree_type:
      if _children[0].data == "case_condition":
        _children = _children[1:]
    else:
      assert False, f"No rule for {lark_tree_type} implemented."

  # Build start node
  start_init_data = copy.deepcopy(init_data)
  start_init_data["end_pos"] = _children[0].meta.start_pos - 1
  start_init_data["partial_str"] = get_partial_str(always_str, start_init_data["start_pos"], start_init_data["end_pos"])
  # Build children node
  children = []
  for i, c in enumerate(_children):
    children.append(get_cdfg(always_str, c, indent=indent + 2))
  start_init_data["children"] = children
  start_node = CdfgNode(start_init_data)
  
  # Build end node
  end_init_data = copy.deepcopy(init_data)
  end_init_data["start_pos"] = _children[-1].meta.end_pos + 1
  end_init_data["partial_str"] = get_partial_str(always_str, end_init_data["start_pos"], end_init_data["end_pos"])
  end_node = CdfgNode(end_init_data)

  # Connect start, end, and children nodes in between.
  if any(x in lark_tree_type for x in BRANCH_NODES):
    # If the node is a branch node, connect the start and end nodes via a child node.
    for c in children:
      connect_cdfg(start_node, c)
      connect_cdfg(c, end_node)
  else:
    # If the node is a procedural node, start - child0 - child1 - ... - end.
    connect_cdfg(start_node, children[0])
    for i, c in enumerate(children):
      if i > 0:
        connect_cdfg(children[i - 1], c)
    connect_cdfg(children[-1], end_node)
  
  return CdfgNodePair(start_node, end_node)
def number_cdfg_nodes_topdown(root, offset=0):
  nodes = root.to_list_topdown()
  for i, n in enumerate(nodes):
    n.set_node_num(i + offset)
  return nodes

def construct_cdfg_for_always_block(always_str, parser):
  always_str_oneline = " ".join(always_str.split())
  lark_root = parser.parse(always_str_oneline)

  print(always_str)
  cdfg = get_cdfg(always_str_oneline, lark_root)
  nodes = number_cdfg_nodes_topdown(cdfg)
  connect_cdfg(nodes[-1], nodes[0]) # Connect the last node to the first node.
  for n in nodes:
    print(n)
  
  input()
