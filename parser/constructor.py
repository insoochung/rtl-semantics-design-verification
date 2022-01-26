import copy
from lark.tree import Tree

from utils import preprocess_always_str, get_indent_str, get_partial_str
from cdfg import Cdfg, CdfgNode, CdfgNodePair

TERMINAL_NODES = ["statement", "statement_or_null", "case_condition", "always_condition"]
DONT_AGGREGATE_NODES = ["ternary_assignment", "ternary_expression"]
BRANCH_NODES = ["if_else_statement", "case_statement", "ternary_assignment"]
PROCEDURAL_NODES = ["seq_block"]
CONDITION_STATEMENTS = ["if_else_statement", "case_statement", "case_item", "for_statement", "always_statement", "ternary_expression"]
PROMOTION_REQUIRED = CONDITION_STATEMENTS + ["always_block", "seq_block", "ternary_assignment", "assignment"]

def is_terminal(lark_tree, terminal_nodes=TERMINAL_NODES, dont_aggregate_nodes=TERMINAL_NODES + DONT_AGGREGATE_NODES):
  # If there are no Tree children, the node is terminal
  _children = []
  for c in lark_tree.children:
    if isinstance(c, Tree):
      _children.append(c)
  if not _children:
    return True

  # Check Tree's type to determine if the node should be terminal
  if lark_tree.data not in terminal_nodes:
    return False

  # If a tree include these nodes as children, it shouldn't be a terminal node.
  for d in dont_aggregate_nodes:
    for c in lark_tree.children:
      if isinstance(c, Tree) and list(c.find_data(d)) != []:
        return False
  return True

def require_promotion(lark_tree):
  return lark_tree.data in PROMOTION_REQUIRED

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

def get_cdfg(always_str, lark_tree, indent=0, prepend_type=None,
             terminal_nodes=TERMINAL_NODES, dont_aggregate_nodes=TERMINAL_NODES + DONT_AGGREGATE_NODES):
  if not isinstance(lark_tree, Tree):
    return None
  lark_tree_type = lark_tree.data.strip()
  if prepend_type:
    lark_tree_type = f"{prepend_type},{lark_tree_type}"

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
               "is_terminal": is_terminal(lark_tree, terminal_nodes=terminal_nodes,
                                          dont_aggregate_nodes=dont_aggregate_nodes),
               "type": lark_tree_type,
               "children": [],}

  if init_data["is_terminal"]:
    # Return a simple node if it's terminal.
    return CdfgNode(init_data)

  if is_same_as_only_child(lark_tree):
    # If the tree is identical to its only child, return the child with type signatures of its ancestors.
    return get_cdfg(always_str, lark_tree.children[0], indent, prepend_type=lark_tree_type,
                    terminal_nodes=terminal_nodes, dont_aggregate_nodes=dont_aggregate_nodes)


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
    elif "assignment" in lark_tree_type:
      assert _children[0].data == "lvalue"
      _children = _children[1:]
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
  if "assignment" in lark_tree_type:
    # Usually, assignments are too fine-grained to be a single node.
    # However, when assignments involve ternary expressions, identified nodes
    # should be finer grained than statements (from here and on in deeper recursions.
    terminal_nodes = list(set(terminal_nodes + ["expression"]))

  for i, c in enumerate(_children):
    child = get_cdfg(always_str, c, indent=indent + 2,
                     terminal_nodes=terminal_nodes, dont_aggregate_nodes=dont_aggregate_nodes)
    if i > 0:
      children[-1].update_end_pos(max(child.get_start_pos() - 1, children[-1].get_end_pos()))
    children.append(child)
  start_init_data["children"] = children
  start_node = CdfgNode(start_init_data)

  # Build end node
  end_init_data = copy.deepcopy(init_data)
  end_init_data["start_pos"] = _children[-1].meta.end_pos + 1
  end_init_data["partial_str"] = get_partial_str(always_str, end_init_data["start_pos"], end_init_data["end_pos"])
  end_init_data["type"] += ",end"
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

def number_cdfg_nodes(root, node_offset=0):
  nodes = root.to_list()
  for i, n in enumerate(nodes):
    n.set_node_num(i + node_offset)
  return nodes

def stringify_cdfg(cdfg, node_offset):
  nodes = number_cdfg_nodes(cdfg, node_offset)
  return "\n".join(str(n) for n in nodes) + "\n"

def construct_cdfg_for_always_block(always_str, parser, node_offset=0,
                                    check_equivalence=True,
                                    manual_inspection=False):
  always_str_oneline = " ".join(preprocess_always_str(always_str).split())
  lark_root = parser.parse(always_str_oneline)
  cdfg = get_cdfg(always_str_oneline, lark_root)
  nodes = cdfg.to_list()
  connect_cdfg(nodes[-1], nodes[0]) # Connect the last node to the first node.

  # Confirm CDFG reconstruction is equivalent to the original always block.
  cdfg_str = stringify_cdfg(cdfg, node_offset=node_offset)
  if check_equivalence:
    orig = preprocess_always_str(always_str, no_space=True)
    comp = preprocess_always_str(cdfg_str, no_space=True)
    assert orig == comp, f"\n{orig}\n!=\n{comp}\n{always_str}\n!=\n{cdfg_str}"

  if manual_inspection: # Print to assess manually.
    print(f"\nOriginal:\n{always_str}\nReconstructed:\n{cdfg_str}")
    if check_equivalence:
      print("String equivalence is verified. "
            "Proceed to check the correctness of the node information.\n")
    input("Press Enter to continue...")

  ret = {"cdfg": cdfg, "num_nodes": len(nodes), "cdfg_str": cdfg_str}
  return ret
