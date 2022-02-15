import re
from typing import List, Union

from utils import get_indent_str, preprocess_rtl_str


class Tag:
  MODULE = "kModuleDeclaration"
  ALWAYS = "kAlwaysStatement"
  ALWAYS_CONTENT = "kProceduralTimingControlStatement"
  ALWAYS_CONDITION = "kEventControl"
  TERNARY_EXPRESSION = "kConditionExpression"
  IF_ELSE_STATEMENT = "kConditionalStatement"
  IF_CLAUSE = "kIfClause"
  IF_HEADER = "kIfHeader"
  IF_BODY = "kIfBody"
  ELSE_CLAUSE = "kElseClause"
  ELSE_BODY = "kElseBody"
  CASE_STATEMENT = "kCaseStatement"
  FOR_LOOP_STATEMENT = "kForLoopStatement"
  FOR_LOOP_CONDITION = "kLoopHeader"
  SEQ_BLOCK = "kSeqBlock"
  BLOCK_ITEM_LIST = "kBlockItemStatementList"
  STATEMENT = "kStatement"

  NULL_STATEMENT = "kNullStatement"
  CASE_ITEM_LIST = "kCaseItemList"
  EXPRESSION_LIST = "kExpressionList"
  EXPRESSION = "kExpression"
  UNARY_EXPRESSION = "kUnaryExpression"
  UNARY_PREFIX_EXPRESSION = "kUnaryPrefixExpression"
  BINARY_EXPRESSION = "kBinaryExpression"
  CONCATENATE_EXPRESSION = "kConcatenationExpression"
  NUMBER = "kNumber"
  REFERENCE = "kReferenceCallBase"
  SYMBOL_IDENTIFIER = "SymbolIdentifier"
  LVALUE = "kLPValue"
  PARENTHESIS_GROUP = "kParenGroup"

  # Assignments
  ASSIGNMENT = "kNetVariableAssignment"
  ASSIGNMENT_MODIFY = "kAssignModifyStatement"
  NON_BLOCKING_ASSIGNMENT = "kNonblockingAssignmentStatement"

  # Keywords
  BEGIN = "kBegin"
  END = "kEnd"
  DEFAULT = "default"
  UNIQUE = "unique"
  CASE = "case"
  CASEZ = "casez"
  ENDCASE = "endcase"
  COLON = ":"
  SEMICOLON = ";"
  ASSIGN = "="
  ASSIGN_NONBLOCK = "<="

  # Categories
  BRANCH_STATEMENTS = [IF_ELSE_STATEMENT, CASE_STATEMENT, TERNARY_EXPRESSION]
  CONDITION_STATEMENTS = [IF_HEADER, CASE_STATEMENT]
  BLOCK_STATEMENTS = BRANCH_STATEMENTS + [SEQ_BLOCK, FOR_LOOP_STATEMENT]
  TERMINAL_STATEMENTS = [ASSIGNMENT, ASSIGNMENT_MODIFY,
                         NON_BLOCKING_ASSIGNMENT, STATEMENT, NULL_STATEMENT]
  ASSIGNMENTS = [ASSIGNMENT, ASSIGNMENT_MODIFY, NON_BLOCKING_ASSIGNMENT]
  ASSIGN_OPERATORS = [ASSIGN, ASSIGN_NONBLOCK]
  EXPRESSIONS = [EXPRESSION, UNARY_EXPRESSION, BINARY_EXPRESSION, REFERENCE,
                 CONCATENATE_EXPRESSION, NUMBER, PARENTHESIS_GROUP,
                 UNARY_PREFIX_EXPRESSION]


class Condition:
  TRUE = "true"
  FALSE = "false"
  DEFAULT = "default"
  DATA = "data"


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


def get_branch_condition_tree(branch_tree: dict):
  """Return the condition expression tree of a conditional statement."""
  tag = branch_tree["tag"]
  assert tag in Tag.CONDITION_STATEMENTS + [Tag.TERNARY_EXPRESSION], (
      f"{tag} is does not have an innate condition.")
  children = branch_tree["children"]
  if tag == Tag.IF_HEADER:
    condition_tree = children[-1]
    assert condition_tree["tag"] == Tag.PARENTHESIS_GROUP
  elif tag == Tag.CASE_STATEMENT:
    condition_tree = children[2]
    assert condition_tree["tag"] == Tag.PARENTHESIS_GROUP
  elif tag == Tag.TERNARY_EXPRESSION:
    condition_tree = children[0]
  else:
    assert 0, f"Cannot extract branch node from {tag}"

  return condition_tree


def get_case_item_tree(case_statement_tree: dict):
  """Return the case item tree of a case statement."""
  assert case_statement_tree["tag"] == Tag.CASE_STATEMENT
  children = case_statement_tree["children"]
  assert children[3]["tag"] == Tag.CASE_ITEM_LIST
  return children[3]


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


def get_symbol_idendifiers_in_tree(verible_tree: dict, rtl_content: str,
                                   ignore_indexing_variables: bool = True,
                                   ignore_object_attributes: bool = True,
                                   ignore_constants: bool = True):
  """Return a list of symbol identifiers in the verible tree."""
  ret = set()
  for t in find_subtree(verible_tree, Tag.SYMBOL_IDENTIFIER):
    cand_var = get_subtree_text(t, rtl_content)
    ret.add(cand_var)

  if ignore_indexing_variables:
    for t in find_subtree(verible_tree, "kDimensionScalar"):
      ret -= get_symbol_idendifiers_in_tree(t,
                                            rtl_content, False, False, False)
  if ignore_object_attributes:
    for t in find_subtree(verible_tree, "kHierarchyExtension"):
      ret -= get_symbol_idendifiers_in_tree(t,
                                            rtl_content, False, False, False)
  if ignore_constants:
    constants = set()
    for v in ret:
      if v.isupper():  # Assumes all uppercase names for constants.
        constants.add(v)
    ret -= constants
  return ret


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


def get_start_end_node(nodes: Union[tuple, list]):
  """Get real start and end nodes of a block."""
  start_node = get_leftmost_node(nodes)
  end_node = get_rightmost_node(nodes)
  assert isinstance(start_node, Node) and isinstance(end_node, Node)
  return start_node, end_node


def connect_nodes(node: "Node", next_node: "Node",
                  condition=None):
  """Connect two nodes."""
  node = get_rightmost_node(node)
  next_node = get_leftmost_node(next_node)
  if node.is_end and next_node.is_end and condition is None:
    # If both nodes are arbitrary end nodes, merge them together.
    for p in node.prev_nodes:
      conds = p.remove_next_node(node)
      assert len(conds) == 1, f"Unexpected conditions: {conds}"
      p.add_next_node(next_node, conds[0])
    del node
    return
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
  # Construct an end node:
  end_node = EndNode(block_depth=block_depth)

  # Construct if-body node.
  if_body_node = if_clause["children"][1]
  assert if_body_node["tag"] == Tag.IF_BODY
  assert len(if_body_node["children"]) == 1
  block_node = if_body_node["children"][0]
  assert block_node["tag"] in Tag.TERMINAL_STATEMENTS + [Tag.SEQ_BLOCK]
  if block_node["tag"] == Tag.SEQ_BLOCK:
    if_nodes = construct_block(
        block_node, rtl_content, block_depth=block_depth + 1)
  else:  # Tag.TERMINAL_STATEMENTS
    if_node = construct_statement(block_node, rtl_content,
                                  block_depth=block_depth + 1)
    if_nodes = [if_node]
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
    else_nodes = construct_block(
        block_node, rtl_content, block_depth=block_depth + 1)
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
  # Construct an end node:
  end_node = EndNode(block_depth=block_depth)
  # Construct case-item-list node.
  default_node = None
  nodes = []
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
    connect_nodes(branch_node, node, condition=conditions)
    # Connect case-item-list node to end node.
    connect_nodes(node, end_node)
    nodes.append(node)
    if children_tags[0] == Tag.DEFAULT:
      assert default_node is None, "Multiple default cases"
      default_node = get_leftmost_node(node)

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
  start_node = Node(verible_tree, rtl_content=rtl_content,
                    block_depth=block_depth)

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

  start_node, end_node = get_start_end_node(nodes)
  return start_node, end_node


def construct_always_node(verible_tree: dict, rtl_content: str, block_depth: int = 0):
  """Construct always node and its children nodes."""
  always_node = AlwaysNode(verible_tree, rtl_content, block_depth)
  children = always_node.verible_tree["children"]
  assert len(children) == 2
  if always_node.type in ["always_ff", "always"]:
    content = children[1]["children"]
    assert len(content) == 2
    condition, seq_block = content[0], content[1]
    assert condition["tag"] == Tag.ALWAYS_CONDITION
    always_node.condition = get_subtree_text(
        condition, always_node.rtl_content)
  else:
    assert always_node.type in ["always_comb", "always_latch"], (
        f"Unknown '{always_node.type}' type.")
    seq_block = children[1]
  assert seq_block["tag"] == Tag.SEQ_BLOCK
  block_nodes = construct_block(
      seq_block, always_node.rtl_content,
      block_depth=always_node.block_depth + 1)
  assert block_nodes, "Seq block is empty."
  # Arbitrary end always_node.
  always_node.end_node = EndNode(block_depth=always_node.block_depth)
  connect_nodes(always_node, block_nodes[0])
  connect_nodes(block_nodes[-1], always_node.end_node)
  # Loop back to the start node.
  connect_nodes(always_node.end_node, always_node)

  always_node.update_condition_vars()
  always_node.update_assigned_vars()
  always_node.print_graph()
  return always_node


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
    self.num_always_blocks = sum([len(m.always_graphs) for m in self.modules])

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
  is_end -- whether the node is the end of a block (bool)
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
    self.is_end = False
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
    if self.line_num >= 0:
      prefix += f" @L{self.line_num}"
    if self.condition:
      prefix += f" / cond.: {self.condition}"
    if self.lead_condition:
      prefix += f" / lead cond.: {self.lead_condition}"
    if self.condition_vars:
      prefix += f" / cond. vars: {self.condition_vars}"
    if self.assigned_vars:
      prefix += f" / assigned vars: {self.assigned_vars}"
    s = self.get_one_line_str()
    if s and "always" not in self.type:
      return f"({prefix}): {s}"
    else:
      return f"({prefix})"

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
      next_node.lead_condition = f"!{self.condition}"
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

  def to_list(self):
    ret = [self]
    for n, _ in self.next_nodes:
      if n.block_depth > self.block_depth and n != self.end_node:
        ret.extend(n.to_list())
        while (len(ret[-1].next_nodes) == 1
               and ret[-1].next_nodes[0][0].block_depth > self.block_depth):
          next_n, _ = ret[-1].next_nodes[0]
          if next_n == self.end_node:
            break
          ret.extend(next_n.to_list())

    if self.end_node:
      ret.extend(self.end_node.to_list())
    return ret


class EndNode(Node):
  """Node class that specifies an arbitrary end"""

  def __init__(self, verible_tree: dict = None, rtl_content: str = "",
               block_depth: int = 0):
    super().__init__(verible_tree=verible_tree,
                     rtl_content=rtl_content, block_depth=block_depth)
    self.is_end = True
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
      ids |= get_symbol_idendifiers_in_tree(condition_tree, self.rtl_content)

      if subtree["tag"] == Tag.CASE_STATEMENT:
        # If the condition is a case statement, see if case items contain
        # variables.
        for case_item in get_case_item_tree(subtree)["children"]:
          expr_list = case_item["children"][0]
          assert expr_list["tag"] in [Tag.EXPRESSION_LIST, Tag.DEFAULT], (
              f"{expr_list['tag']} is not an expected type of node.")
          ids |= get_symbol_idendifiers_in_tree(expr_list, self.rtl_content)
    self.condition_vars = ids

  def update_assigned_vars(self):
    """Find and update the vars assigned within always."""
    assign_subtrees = find_subtree(self.verible_tree, Tag.ASSIGNMENTS)
    ids = set()
    for subtree in assign_subtrees:
      lhs_subtree = subtree["children"][0]
      assert lhs_subtree["tag"] == Tag.LVALUE, (
          f"{lhs_subtree['tag']} is not an expected type of node.")
      ids |= get_symbol_idendifiers_in_tree(lhs_subtree, self.rtl_content)
    self.assigned_vars = ids

  def print_graph(self):
    """Print the graph of the always block."""
    l = self.to_list()
    print()
    for n in l:
      print(get_indent_str(n.block_depth * 2) + str(n))
    print()
