import code_parser
import config
from lark import Lark

def preprocess_always_str(always_str):
  # 1. Remove comments and break multi statement lines in always blocks.
  lines = always_str.split("\n")
  lines = [line.split("//")[0] for line in lines]
  lines = [line for line in lines if line.strip()]
  always_str = "\n".join(lines)

  # 2. Merge multi-line statements into a single line.
  lines = always_str.split("\n")
  for i, line in enumerate(lines):
    if any(x in line for x in ["case", "if", "else", ":", ";", "begin", "end"]):
      lines[i] += "\n"

  # 3. Replace multiple spaces with a single space.
  res = "".join(lines)
  lines = res.split("\n")
  for i, line in enumerate(lines):
    indent_size = len(line) - len(line.lstrip())
    lines[i] = " " * indent_size + " ".join(line.split()) + "\n"
  res = "".join(lines)
  return res

def parse_always_str(always_str):
  parser = Lark.open("ibex_always.lark", rel_to=__file__)
  tree = parser.parse(always_str)
  print(tree.pretty())

if __name__ == "__main__":
  res = code_parser.parse()
  for filename, modules in res.items():
    for module_name, (line_num, always_blocks) in modules.items():
      for always_block in always_blocks:
        line_num, always_str = always_block
        always_str = preprocess_always_str(always_str)
        parse_always_str(always_str)
        input()
