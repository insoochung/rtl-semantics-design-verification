import copy
from lark.tree import Tree

from utils import preprocess_rtl_str, get_partial_str, assert_rtls_equivalence
from cdfg import Cdfg, CdfgNode, CdfgNodePair, connect_nodes, stringify_cdfg

MINIMAL_CANDIDATES = ["statement", "statement_or_null", "case_condition", "always_condition"]
DONT_AGGREGATE_NODES = ["ternary_assignment", "ternary_expression"]
BRANCH_NODES = ["if_else_statement", "case_statement", "ternary_expression"]
PROCEDURAL_NODES = ["seq_block"]
CONDITION_STATEMENTS = ["if_else_statement", "case_statement", "case_item", "for_statement", "always_statement", "ternary_expression"]
PROMOTION_REQUIRED = CONDITION_STATEMENTS + ["always_block", "seq_block", "ternary_assignment", "assignment"]

def is_minimal_node(lark_tree, minimal_candidates=MINIMAL_CANDIDATES, dont_aggregate_nodes=MINIMAL_CANDIDATES + DONT_AGGREGATE_NODES):
  # If there are no Tree children, the node is minimal
  _children = []
  for c in lark_tree.children:
    if isinstance(c, Tree):
      _children.append(c)
  if not _children:
    return True

  # Check Tree's type to determine if the node should be minimal
  if lark_tree.data not in minimal_candidates:
    return False

  # If a tree include these nodes as children, it shouldn't be a minimal node.
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

def get_cdfg_node(always_str, lark_tree, indent=0, prepend_type=None,
             minimal_candidates=MINIMAL_CANDIDATES, dont_aggregate_nodes=MINIMAL_CANDIDATES + DONT_AGGREGATE_NODES):
  if not isinstance(lark_tree, Tree):
    return None
  lark_tree_type = lark_tree.data.strip()
  if prepend_type:
    lark_tree_type = f"{prepend_type},{lark_tree_type}"

  m = lark_tree.meta
  start_pos = m.start_pos
  end_pos = m.end_pos
  init_data = {"full_str": always_str,
               "statement": get_partial_str(always_str, start_pos, end_pos),
               "lark_tree": lark_tree,
               "lark_meta": m,
               "indent": indent,
               "start_pos": start_pos,
               "end_pos": end_pos,
               "is_minimal_node": is_minimal_node(lark_tree, minimal_candidates=minimal_candidates,
                                          dont_aggregate_nodes=dont_aggregate_nodes),
               "type": lark_tree_type,}

  if init_data["is_minimal_node"]:
    # Return a simple node if it's minimal.
    return CdfgNode(init_data)

  if is_same_as_only_child(lark_tree):
    # If the tree is identical to its only child, return the child with type signatures of its ancestors.
    return get_cdfg_node(always_str, lark_tree.children[0], indent, prepend_type=lark_tree_type,
                    minimal_candidates=minimal_candidates, dont_aggregate_nodes=dont_aggregate_nodes)


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
  start_init_data["statement"] = get_partial_str(always_str, start_init_data["start_pos"], start_init_data["end_pos"])

  # Build children node
  children = []
  if "assignment" in lark_tree_type:
    # Usually, assignments are too fine-grained to be a minimal node.
    # However, when assignments involve ternary expressions, identified nodes
    # should be finer grained than statements (from here and on in deeper recursions.
    minimal_candidates = list(set(minimal_candidates + ["expression"]))

  for i, c in enumerate(_children):
    child = get_cdfg_node(always_str, c, indent=indent + 2,
                     minimal_candidates=minimal_candidates, dont_aggregate_nodes=dont_aggregate_nodes)
    if i > 0:
      children[-1].update_end_pos(max(child.get_start_pos() - 1, children[-1].get_end_pos()))
    children.append(child)
  start_node = CdfgNode(start_init_data)

  # Build end node
  end_init_data = copy.deepcopy(init_data)
  end_init_data["start_pos"] = _children[-1].meta.end_pos + 1
  end_init_data["statement"] = get_partial_str(always_str, end_init_data["start_pos"], end_init_data["end_pos"])
  end_init_data["type"] += ",end"
  end_node = CdfgNode(end_init_data)

  # Connect start, end, and children nodes in between.
  if any(x in lark_tree_type for x in BRANCH_NODES):
    # If the node is a branch node, connect the start and end nodes via a child node.
    for c in children:
      connect_nodes(start_node, c)
      connect_nodes(c, end_node)

    # TODO: Add check for the connectivity assumption.
    # This connection scheme assumes that there exists a branch for all cases.

  else:
    # If the node is a procedural node, start - child0 - child1 - ... - end.
    connect_nodes(start_node, children[0])
    for i, c in enumerate(children):
      if i > 0:
        connect_nodes(children[i - 1], c)
    connect_nodes(children[-1], end_node)

  return CdfgNodePair(start_node, end_node)

def construct_cdfg_for_always_block(always_str, parser, node_offset=0,
                                    check_equivalence=True,
                                    manual_inspection=False):
  indent = len(always_str) - len(always_str.lstrip())
  always_str_oneline = " ".join(preprocess_rtl_str(always_str).split())
  lark_root = parser.parse(always_str_oneline)
  cdfg = Cdfg(get_cdfg_node(always_str_oneline, lark_root, indent))
  nodes = cdfg.to_list()
  connect_nodes(nodes[-1], nodes[0]) # Connect the last node to the first node.

  # Confirm CDFG reconstruction is equivalent to the original always block.
  cdfg_str = stringify_cdfg(cdfg, node_offset=node_offset)
  if check_equivalence:
    assert_rtls_equivalence(always_str, cdfg_str)

  if manual_inspection: # Print to assess manually.
    print(f"\nOriginal:\n{always_str}\nReconstructed:\n{cdfg_str}")
    if check_equivalence:
      print("String equivalence is verified. "
            "Proceed to check the correctness of the node information.\n")
    input("Press Enter to continue...")

  return cdfg
