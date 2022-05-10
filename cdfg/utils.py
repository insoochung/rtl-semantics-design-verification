import re
import os
import sys

from typing import List, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.constants import Tag


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
    return re.sub("//.*?\n|/\*.*?\*/", "", text, flags=re.S)


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


def get_symbol_identifiers_in_tree(
    verible_tree: dict,
    rtl_content: str,
    ignore_indexing_variables: bool = True,
    ignore_object_attributes: bool = True,
    ignore_constants: bool = True,
):
    """Return a list of symbol identifiers in the verible tree."""
    ret = set()
    for t in find_subtree(verible_tree, Tag.SYMBOL_IDENTIFIER):
        cand_var = get_subtree_text(t, rtl_content)
        ret.add(cand_var)

    if ignore_indexing_variables:
        for t in find_subtree(verible_tree, "kDimensionScalar"):
            ret -= get_symbol_identifiers_in_tree(t, rtl_content, False, False, False)
    if ignore_object_attributes:
        for t in find_subtree(verible_tree, "kHierarchyExtension"):
            ret -= get_symbol_identifiers_in_tree(t, rtl_content, False, False, False)
    if ignore_constants:
        constants = set()
        for v in ret:
            if v.isupper():  # Assumes all uppercase names for constants.
                constants.add(v)
        ret -= constants
    return ret


def get_branch_condition_tree(branch_tree: dict):
    """Return the condition expression tree of a conditional statement."""
    tag = branch_tree["tag"]
    assert tag in Tag.CONDITION_STATEMENTS + [
        Tag.TERNARY_EXPRESSION
    ], f"{tag} is does not have an innate condition."
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
