import config
import os

from lark import Lark
from lark.reconstruct import Reconstructor
from tqdm import tqdm

from utils import preprocess_always_str, parse_rtl, get_log_fn
from constructor import construct_cdfg_for_always_block
from cdfg import maybe_connect_cdfgs

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

def test_parsing_integrity(parsed_rtl, parser, reconstructor, verbose=False):
  log = get_log_fn(verbose)
  log("-- Start: Testing parsing integrity... --")
  for filepath, modules in parsed_rtl.items():
    for module_name, (line_num, always_blocks) in modules.items():
      log(f"Parsing always blocks in '{module_name}' to check reconstruction integrity.")
      for always_block in tqdm(always_blocks):
        _, always_str = always_block
        _test_parsing_integrity(always_str, parser, reconstructor)
  log("-- Done: Parsing integrity verified! --\n")

def generate_cdfgs(parsed_rtl, parser, verbose=False):
  log = get_log_fn(verbose)
  log("-- Start: Generating CDFGs from always block strings... --")
  cdfgs = {}
  for filepath, modules in parsed_rtl.items():
    cdfgs[filepath] = {}
    log(f"Generting CDFGs from always blocks in '{filepath}'.")
    offset = 0
    for module_name, (_, always_blocks) in modules.items():
      cdfgs[filepath][module_name] = []
      for _, always_str in tqdm(always_blocks):
        # Replace the original always block with the reformatted one
        cdfg = construct_cdfg_for_always_block(always_str, parser, offset)
        offset += len(cdfg)
        cdfgs[filepath][module_name].append(cdfg)
      # Add data edges between CDFGs.
      module_cdfgs = cdfgs[filepath][module_name]
      maybe_connect_cdfgs(module_cdfgs)
  log("-- Done: CDFGs generated! --\n")
  return cdfgs

def reformat_rtl_based_on_cdfgs(parsed_rtl, cdfgs, write_to_file=False, write_dir=None, postfix=None, verbose=False):
  log = get_log_fn(verbose)
  log("-- Start: Reformatting RTL code files given CDFGs... --")
  def _get_new_filepath(orig_path, write_dir=None, postfix=None):
    assert write_dir or postfix, \
      "To get new write path 'write_dir' or 'postfix' should be provided."
    if write_dir:
      os.makedirs(write_dir, exist_ok=True)
      return os.path.join(write_dir, os.path.basename(orig_path))
    else: # postfix
      return orig_path + postfix

  ret = {}
  for filepath, modules in parsed_rtl.items():
    prev_line_num = 0
    with open(filepath, "r") as f:
      orig_lines = f.readlines()
    orig_lines = [""] + orig_lines # To match the saved line numbers (1-based)
    non_always_text = []
    always_cdfgs = []
    for module_name, (_, always_blocks) in modules.items():
      module_cdfgs = cdfgs[filepath][module_name]
      for cdfg, always_block in zip(module_cdfgs, always_blocks):
        line_num, always_str = always_block
        always_lines = always_str.split("\n")
        # Check if the line number is correct
        first_line = always_lines[0].strip()
        first_line_actual = orig_lines[line_num].strip()
        assert first_line == first_line_actual, \
          f"Line number mismatch: {first_line} (line # {line_num}) != {first_line_actual}"
        # Prepend the lines in between always blocks
        non_always_text.append("".join(orig_lines[prev_line_num:line_num]))
        prev_line_num = line_num + len(always_lines)
      always_cdfgs += module_cdfgs

    # Replace always blocks in the original file with the reformatted ones
    reformatted_text = ""
    assert len(always_cdfgs) == len(non_always_text)
    for i, cdfg in enumerate(always_cdfgs):
      reformatted_text += non_always_text[i]
      reformatted_text += str(cdfg) + "\n"
    # Append the last lines
    reformatted_text = reformatted_text + "".join(orig_lines[prev_line_num:])
    ret[filepath] = reformatted_text
    if write_to_file:
      new_filepath = _get_new_filepath(filepath, write_dir=write_dir, postfix=postfix)
      with open(new_filepath, "w") as f:
        f.write(reformatted_text)
      log(f"Reformatted RTL written to '{new_filepath}'.")
  log("-- Done: RTL code files reformatted! --\n")
  return ret

def reduce_cdfgs(cdfgs, verbose=False):
  log = get_log_fn(verbose)
  for filepath, modules in cdfgs.items():
    for module_name, module_cdfgs in modules.items():
      for cdfg in module_cdfgs:
        cdfg.reduce()

if __name__ == "__main__":
  parsed_rtl = parse_rtl()
  parser, reconstructor = get_parser_and_reconstructor(config.ALWAYS_BLOCK_RULES)
  # test_parsing_integrity(parsed_rtl, parser, reconstructor, verbose=True)
  cdfgs = generate_cdfgs(parsed_rtl, parser, verbose=True)
  reduce_cdfgs(cdfgs, verbose=True)
  reformat_rtl_based_on_cdfgs(
    parsed_rtl, cdfgs, write_to_file=True, write_dir=config.REFORMATTED_DIR, verbose=True)
