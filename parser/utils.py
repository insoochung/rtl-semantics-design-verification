import os
import json

from zipfile import ZipFile
from glob import glob

import config

def get_log_fn(verbose=True):
  """Return a function that prints to stdout"""
  def log_fn(str=""):
    pass
  if verbose:
    log_fn = print

  return log_fn

def get_indent_str(indent):
  return " " * indent

def print_tags(verible_dict, indent_size=0):
  """Print the tags of the verible_dict"""
  if verible_dict is None:
    return
  if "tag" in verible_dict.keys():
    print(get_indent_str(indent_size), verible_dict["tag"])
  if "children" in verible_dict.keys():
    for c in verible_dict["children"]:
      print_tags(c, indent_size + 2)
  elif "tree" in verible_dict.keys():
    print_tags(verible_dict["tree"], indent_size + 2)