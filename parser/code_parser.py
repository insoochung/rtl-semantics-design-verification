import os
from glob import glob

BASE_DIR = os.path.abspath(
  os.path.dirname(
    os.path.dirname(__file__)))
IBEX_DIR = os.path.join(BASE_DIR, "ibex")
RTL_DIR = os.path.join(IBEX_DIR, "rtl")

def get_modules_from_file(filepath):
  """ Return a list of module strings from a file """
  modules = {}
  with open(filepath, "r") as f:
    lines = f.readlines()
  
  module_str = ""
  module_name = ""
  do_append = False
  for i, line in enumerate(lines):
    if line.startswith("module"):
      do_append = True
      module_name = line.split(" ")[1].strip()
      start_linenum = i + 1
    
    if do_append:
      module_str += " ".join(line.split()) + "\n"
    
    if line.startswith("endmodule"):
      do_append = False
      if module_name:
        modules[module_name] = (start_linenum, module_str)
      module_str = ""
      module_name = ""

  return modules

def get_always_blocks_from_modules(module_str, line_num):
  """ From module RTL code string, return a list of always blocks """
  always_blocks = []
  do_append = False
  always_str = ""
  always_linenum = 0
  cnt_begin = 0
  for i, line in enumerate(module_str.split("\n")):
    if line.startswith("always"):
      do_append = True
      always_linenum = line_num + i
    
    if do_append:
      always_str += line + "\n"
      # We are assuming one statement for a line in an always block.
      assert line.count(";") <= 1 or line.startswith("for"), (
        "Line '{}' has more than one semicolon".format(line))
      if "begin" in line:
        cnt_begin += 1
      if "end" in line:
        cnt_begin -= 1

      if cnt_begin == 0: # end of always block
        do_append = False
        if always_str:
          always_blocks.append((always_linenum, always_str))
        always_str = ""

  return always_blocks

def parse(rtl_dir=RTL_DIR, verbose=True):
  if verbose:
    log_fn = print
  else:
    def log_fn(str=""):
      pass

  log_fn(f"Attempting to parse files in {rtl_dir}")
  sv_filepaths = glob(os.path.join(rtl_dir, "*.sv"))
  files_to_modules = {}
  for filepath in sv_filepaths:
    filename = os.path.basename(filepath)
    files_to_modules[filename] = get_modules_from_file(filepath)
    log_fn("Module extraction: filename={}, # modules={}".format(
      filename, len(files_to_modules[filename])))
  log_fn()
  cnt = 0
  res = {}
  for filename, modules in files_to_modules.items():
    res[filename] = {}
    for module_name, (line_num, module_str) in modules.items():
      always_blocks = get_always_blocks_from_modules(module_str, line_num)
      log_fn("Always extraction: filename={}, modulename={}, # always={}".format(
        filename, module_name, len(always_blocks)))
      res[filename][module_name] = (line_num, always_blocks)
      cnt += len(always_blocks)
  log_fn()
  log_fn("Total # always blocks: {} (should be 193)".format(cnt))

  return res

if __name__ == "__main__":
  parse()