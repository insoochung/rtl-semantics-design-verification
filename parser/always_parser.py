import utils
import config
import re

from lark import Lark
from lark.reconstruct import Reconstructor

def strip_comments(text):
    return re.sub('//.*?\n|/\*.*?\*/', '', text, flags=re.S)

def preprocess_always_str(always_str):
  # 1. Remove comments
  res = strip_comments(always_str)
  # 2. Merge multi-line statements into a single line.
  lines = res.split("\n")
  for i, line in enumerate(lines):
    if any(x in line for x in ["case", "if", "else", ":", ";", "begin", "end"]):
      lines[i] += "\n"
  # 3. Replace multiple spaces with a single space, but indents are preserved.
  res = "".join(lines)
  lines = res.split("\n")
  for i, line in enumerate(lines):
    indent_size = len(line) - len(line.lstrip())
    lines[i] = " " * indent_size + " ".join(line.split()) + "\n"
  res = "".join(lines)
  return res

def test_parsing_integrity(always_str, lark_rules=config.LARK_RULES):
  test_str = preprocess_always_str(always_str)
  parser = Lark.open(lark_rules, maybe_placeholders=False)
  root = parser.parse(test_str)
  reconstructor = Reconstructor(parser)
  reduced_always = "".join(test_str.split())
  reconstructed_always = reconstructor.reconstruct(root, insert_spaces=False).replace(" ", "")
  assert reduced_always == reconstructed_always, \
    "Reduced and reconstructed always blocks are not the same. \n{}\n{}".format(
      reduced_always, reconstructed_always)

if __name__ == "__main__":
  from tqdm import tqdm
  res = utils.parse_rtl()
  for filename, modules in res.items():
    for module_name, (line_num, always_blocks) in modules.items():
      print(f"Parsing always blocks in '{module_name}' to check reconstruction integrity.")
      for always_block in tqdm(always_blocks):
        line_num, always_str = always_block
        test_parsing_integrity(always_str)
