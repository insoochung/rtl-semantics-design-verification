import config
import os

from lark import Lark
from lark.reconstruct import Reconstructor
from tqdm import tqdm

from utils import preprocess_always_str, parse_rtl
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
  for filepath, modules in parsed_rtl.items():
    for module_name, (line_num, always_blocks) in modules.items():
      print(f"Parsing always blocks in '{module_name}' to check reconstruction integrity.")
      for always_block in tqdm(always_blocks):
        _, always_str = always_block
        _test_parsing_integrity(always_str, parser, reconstructor)

def _reformat_always_for_cdfg(always_str, parser, node_offset=0):
  return construct_cdfg_for_always_block(always_str, parser, node_offset=node_offset)

def reformat_always_for_cdfg(parsed_rtl, parser, write_to_file=False, write_dir=None, postfix=None):
  def get_new_filepath(orig_path, write_dir=None, postfix=None):
    assert write_dir or postfix, \
      "To get new write path 'write_dir' or 'postfix' should be provided."
    if write_dir:
      os.makedirs(write_dir, exist_ok=True)
      return os.path.join(write_dir, os.path.basename(orig_path))
    else: # postfix
      return orig_path + postfix

  cdfgs = {}
  for filepath, modules in parsed_rtl.items():
    cdfgs[filepath] = {}
    cdfgs[filepath]["text"] = ""
    offset = 0
    prev_line_num = 0
    with open(filepath, "r") as f:
      orig_lines = f.readlines()
    orig_lines = [""] + orig_lines # To match the saved line numbers (1-based)
    reformatted_text = ""
    print(f"Generting CDFGs from always blocks in '{filepath}'.")
    for module_name, (_, always_blocks) in modules.items():
      cdfgs[filepath][module_name] = []
      for always_block in tqdm(always_blocks):
        line_num, always_str = always_block
        always_lines = always_str.split("\n")
        # Check if the line number is correct
        first_line = always_lines[0].strip()
        first_line_actual = orig_lines[line_num]
        indent_actual = len(first_line_actual) - len(first_line_actual.lstrip())
        first_line_actual = first_line_actual.strip()
        assert first_line == first_line_actual, \
          f"Line number mismatch: {first_line} (line # {line_num}) != {first_line_actual}"
        # Prepend the lines in between always blocks
        reformatted_text = reformatted_text + "".join(orig_lines[prev_line_num + 1:line_num])
        # Replace the original always block with the reformatted one
        ret = _reformat_always_for_cdfg(always_str, parser, offset)

        reformatted_text = reformatted_text + "\n".join(
          " " * indent_actual + l for l in ret["cdfg_str"].split("\n")
        ).rstrip() + "\n" * 2
        prev_line_num = line_num + len(always_lines) - 1
        offset += ret["num_nodes"]
        cdfgs[filepath][module_name].append(ret)
    # Append the last lines
    reformatted_text = reformatted_text + "".join(orig_lines[prev_line_num + 1:])
    cdfgs[filepath]["text"] = reformatted_text
    if write_to_file:
      new_filepath = get_new_filepath(filepath, write_dir=write_dir, postfix=postfix)
      with open(new_filepath, "w") as f:
        f.write(reformatted_text)
      print(f"Reformatted RTL written to '{new_filepath}'.\n")
    # TODO: Add data edges between CDFGs.

if __name__ == "__main__":
  parsed_rtl = parse_rtl()
  parser, reconstructor = get_parser_and_reconstructor(config.ALWAYS_BLOCK_RULES)
  test_parsing_integrity(parsed_rtl, parser, reconstructor)
  reformat_always_for_cdfg(parsed_rtl, parser, write_to_file=True,
                           write_dir=os.path.join(config.BASE_DIR, "reformatted"))
