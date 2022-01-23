import utils
import config
import re

from lark import Lark
from lark.reconstruct import Reconstructor
from utils import preprocess_always_str
from cdfg_constructor import construct_cdfg_for_always_block

def get_parser_and_reconstructor(lark_rules):
  parser = Lark.open(lark_rules, maybe_placeholders=False, propagate_positions=True)
  reconstructor = Reconstructor(parser)
  return parser, reconstructor

def _test_parsing_integrity(always_str, parser, reconstructor):
  test_str = preprocess_always_str(always_str)
  root = parser.parse(test_str)
  reduced_always = "".join(test_str.split())
  reconstructed_always = reconstructor.reconstruct(root, insert_spaces=False).replace(" ", "")
  assert reduced_always == reconstructed_always, \
    "Reduced and reconstructed always blocks are not the same. \n{}\n{}".format(
      reduced_always, reconstructed_always)

def test_parsing_integrity(parsed_rtl, parser, reconstructor):
  from tqdm import tqdm
  for filename, modules in parsed_rtl.items():
    for module_name, (line_num, always_blocks) in modules.items():
      print(f"Parsing always blocks in '{module_name}' to check reconstruction integrity.")
      for always_block in tqdm(always_blocks):
        line_num, always_str = always_block
        _test_parsing_integrity(always_str, parser, reconstructor)

def reformat_always_for_cdfg(always_str, parser, reconstructor):
  construct_cdfg_for_always_block(always_str, parser)

if __name__ == "__main__":
  parsed_rtl = utils.parse_rtl()
  parser, reconstructor = get_parser_and_reconstructor(config.ALWAYS_BLOCK_RULES)
  test_parsing_integrity(parsed_rtl, parser, reconstructor)
  for filename, modules in parsed_rtl.items():
    for module_name, (line_num, always_blocks) in modules.items():
      for always_block in always_blocks:
        line_num, always_str = always_block
        reformat_always_for_cdfg(always_str, parser, reconstructor)
