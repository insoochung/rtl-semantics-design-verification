import copy
from lark.tree import Tree

from utils import preprocess_always_str

TERMINAL_NODES = ["statement", "statement_or_null", "case_condition", "always_condition"]
BRANCH_NODES = ["if_else_statement", "case_statement"]
PROCEDURAL_NODES = ["seq_block"]
CONDITION_STATEMENTS = ["if_else_statement", "case_statement", "case_item", "for_statement", "always_statement"]
PROMOTION_REQUIRED = CONDITION_STATEMENTS + ["always_block", "seq_block"]

class CdfgNodePair:
  def __init__(self, start_node, end_node):
    self.start_node = start_node
    self.end_node = end_node

  def __str__(self):
    return str(self.start_node) + "\n" + str(self.end_node)

  def append_prev_nodes(self, prev_node):
    self.start_node.append_prev_nodes(prev_node)

  def append_next_nodes(self, next_node):
    self.end_node.append_next_nodes(next_node)

  def to_list(self):
    return self.start_node.to_list() + self.end_node.to_list()

  def get_start_pos(self):
    return self.start_node.start_pos

  def get_end_pos(self):
    return self.end_node.end_pos

  def update_start_pos(self, start_pos):
    self.start_node.update_start_pos(start_pos)

  def update_end_pos(self, end_pos):
    self.end_node.update_end_pos(end_pos)


class CdfgNode:
  def __init__(self, iterable=(), **kwargs):
    self.__dict__.update(iterable, **kwargs)
    self.prev_nodes = []
    self.next_nodes = []

  def get_start_pos(self):
    return self.start_pos

  def get_end_pos(self):
    return self.end_pos

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

  def to_list(self):
    ret = [self]
    for node in self.children:
      ret.extend(node.to_list())
    return ret

  def set_node_num(self, n):
    self.node_num = n

  def __str__(self):
    return (f"{get_indent_str(self.indent)}{self.partial_str} "
            "// {"
            f"\"node_num\": {self.node_num}, "
            f"\"type\": \"{self.type}\", "
            # f"\"start_pos\": {self.start_pos}, "
            # f"\"end_pos\": {self.end_pos}, "
            f"\"prev_nodes\": {[x.node_num for x in self.prev_nodes]}, "
            f"\"next_nodes\": {[x.node_num for x in self.next_nodes]}"
            "}")

  def update_start_pos(self, start_pos):
    self.start_pos = start_pos
    self.update_partial_str()

  def update_end_pos(self, end_pos):
    self.end_pos = end_pos
    self.update_partial_str()

  def update_partial_str(self):
    self.partial_str = get_partial_str(self.full_str, self.start_pos, self.end_pos)

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
    return s[start_pos:end_pos + 1].strip()

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
  _children = []
  for c in lark_tree.children:
    if isinstance(c, Tree): _children.append(c)
  if require_promotion(lark_tree):
    # If promotable nodes (e.g. condition node) to its parent node.
    if any(x in lark_tree_type for x in CONDITION_STATEMENTS):
      if _children[0].data.endswith("condition"):
        _children = _children[1:]
    elif "seq_block" in lark_tree_type:
      if _children[0].data == "block_identifier":
        _children = _children[1:]
        if _children[-1].data == "block_identifier":
          _children = _children[:-1]
    else:
      assert False, f"No rule for {lark_tree_type} implemented."
  if len(_children) == 0:
    assert False, f"Something wrong... Should contain children but there are none '{init_data}'"

  # Build start node
  start_init_data = copy.deepcopy(init_data)
  start_init_data["end_pos"] = _children[0].meta.start_pos - 1
  start_init_data["partial_str"] = get_partial_str(always_str, start_init_data["start_pos"], start_init_data["end_pos"])

  # Build children node
  children = []
  for i, c in enumerate(_children):
    child = get_cdfg(always_str, c, indent=indent + 2)
    if i > 0:
      children[-1].update_end_pos(max(child.get_start_pos() - 1, children[-1].get_end_pos()))
    children.append(child)
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
    connect_cdfg(start_node, end_node) # When no possible condition is met.
  else:
    # If the node is a procedural node, start - child0 - child1 - ... - end.
    connect_cdfg(start_node, children[0])
    for i, c in enumerate(children):
      if i > 0:
        connect_cdfg(children[i - 1], c)
    connect_cdfg(children[-1], end_node)

  return CdfgNodePair(start_node, end_node)

def number_cdfg_nodes(root, offset=0):
  nodes = root.to_list()
  for i, n in enumerate(nodes):
    n.set_node_num(i + offset)
  return nodes

def stringify_cdfg(cdfg):
  nodes = number_cdfg_nodes(cdfg)
  return "\n".join(str(n) for n in nodes) + "\n"

def construct_cdfg_for_always_block(always_str, parser):
  always_str_oneline = " ".join(preprocess_always_str(always_str).split())
  lark_root = parser.parse(always_str_oneline)
  cdfg = get_cdfg(always_str_oneline, lark_root)
  nodes = number_cdfg_nodes(cdfg)
  connect_cdfg(nodes[-1], nodes[0]) # Connect the last node to the first node.

  # Confirm CDFG reconstruction is equivalent to the original always block.
  cdfg_str = stringify_cdfg(cdfg)
  orig = preprocess_always_str(always_str, no_space=True)
  comp = preprocess_always_str(cdfg_str, no_space=True)
  assert orig == comp, f"\n{orig}\n!=\n{comp}"

  # Print to assess manually.
  print(f"\nOriginal:\n{always_str}\nReconstructed:\n{cdfg_str}")
  input("String equivalence is verified. Proceed to check the correctness of the node information.\n"
        "Press Enter to continue...")
