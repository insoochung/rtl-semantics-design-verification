from utils import get_partial_str, get_indent_str

def connect_nodes(prev, next):
  prev.append_next_nodes(next)
  next.append_prev_nodes(prev)

def number_cdfg_nodes(root, node_offset=0, force=False):
  nodes = root.to_list()
  for i, n in enumerate(nodes):
    n.set_node_num(i + node_offset, force=force)
  return nodes

def stringify_cdfg(cdfg, node_offset=0):
  nodes = number_cdfg_nodes(cdfg, node_offset, force=False)
  prefix = "// <ALWAYS_BLOCK> "
  prefix += "{" + f"\"condition_variables\": {list(cdfg.condition_variables)}, "
  prefix += f"\"assigned_variables\": {list(cdfg.assigned_variables)}" + "}\n"
  ret = "\n".join(str(n) for n in nodes) + "\n"
  postfix = "// </ALWAYS_BLOCK>\n"
  ret = prefix + ret + postfix
  return ret

def _maybe_connect_cdfgs(cdfg_a, cdfg_b):
  assert cdfg_a != cdfg_b
  if cdfg_a.assigned_variables & cdfg_b.condition_variables:
    connect_nodes(cdfg_a.root, cdfg_b.root)
  if cdfg_b.assigned_variables & cdfg_a.condition_variables:
    connect_nodes(cdfg_b.root, cdfg_a.root)

def maybe_connect_cdfgs(cdfgs):
  for i, cdfg_a in enumerate(cdfgs):
    for cdfg_b in cdfgs[i + 1:]:
      _maybe_connect_cdfgs(cdfg_a, cdfg_b)

class CdfgNode:
  """ Member attributes:
  - full_str (str): Always string from which this node is formed.
  - statement (str): Statement corresponding to this node.
  - lark_tree (lark.Tree): lark.Tree object this node corresponds to.
  - lark_meta (lark.Meta): Meta information from the lark.Tree object,
  - indent (int): Indent size.
  - start_pos (int): Start position of the statement within full str.
  - end_pos (int): End position of the statement within full str.
  - is_terminal (bool): Whether this is a terminal node.
  - type (str): A string stating the type of this node,
  - children (List[lark.Tree]): List of children.}
  """
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

  def set_node_num(self, n, force=False):
    if not hasattr(self, "node_num"):
      self.node_num = n
    elif self.node_num != n:
      if force:
        self.node_num = n
        print("node_num is overwritten: {self.node_num}->{n}")

  def __str__(self):
    return (f"{get_indent_str(self.indent)}{self.statement} "
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
    self.update_statement()

  def update_end_pos(self, end_pos):
    self.end_pos = end_pos
    self.update_statement()

  def update_statement(self):
    self.statement = get_partial_str(self.full_str, self.start_pos, self.end_pos)

  def is_reducible(self):
    return (self.statement.strip() == ""
      and "always" not in self.type
      and len(self.prev_nodes) == 1 and len(self.prev_nodes[0].next_nodes) == 1
      and len(self.next_nodes) == 1 and len(self.next_nodes[0].prev_nodes) == 1)

  def replace_next_node(self, old_node, new_node):
    self.next_nodes.remove(old_node)
    self.next_nodes.append(new_node)

  def replace_prev_node(self, old_node, new_node):
    self.prev_nodes.remove(old_node)
    self.prev_nodes.append(new_node)

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

class Cdfg:
  def __init__(self, root: CdfgNodePair):
    self.root = root
    self.start_node = root.start_node
    self.end_node = root.end_node
    self.identify_assigned_variables()
    self.identify_condition_variables()
    # Assign root node's methods to self
    self.to_list = self.root.to_list

  def _identify_variables(self, node_type):
    lark_root = self.start_node.lark_tree
    ret = set()
    for n in lark_root.find_data(node_type):
      for id in n.scan_values(lambda x: x.type == "IDENTIFIER"):
        ret.add(id.value)
    return ret

  def identify_assigned_variables(self):
    self.assigned_variables = self._identify_variables("lvalue")

  def identify_condition_variables(self):
    self.condition_variables = self._identify_variables("condition")

  def __str__(self):
    return stringify_cdfg(self.root)