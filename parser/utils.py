import re


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
  return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)


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
