import os
from glob import glob

BASE_DIR = os.path.abspath(
  os.path.dirname(
    os.path.dirname(__file__)))
IBEX_DIR = os.path.join(BASE_DIR, "ibex")
RTL_DIR = os.path.join(IBEX_DIR, "rtl")

def has_token(line, token):
  return token in line.split("//")[0].split() # comment is not taken into account

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
      module_str += line
    
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
    if line.strip().startswith("always"):
      do_append = True
      always_linenum = line_num + i
    
    if do_append:
      always_str += line + "\n"
      # We are assuming one statement for a line in an always block.
      assert line.count(";") <= 1 or line.strip().startswith("for"), (
        "Line '{}' has more than one semicolon".format(line))
      if has_token(line, "begin"):
        cnt_begin += 1
      if has_token(line, "end"):
        cnt_begin -= 1

      if cnt_begin == 0: # end of always block
        do_append = False
        if always_str:
          always_str = always_str.rstrip()
          # Make sure that we are saving a complete always block
          first_line = always_str.split("\n")[0] # this should begin with "always"
          last_line = always_str.split("\n")[-1] # this should be equal to "end"
          assert (len(first_line) - len(first_line.lstrip(' '))
                  == len(last_line) - len(last_line.lstrip(' '))), (
                    "Indent should be the same for start and end of the always block.",
                    always_str, always_linenum
                  )
          last_line = last_line.split("//")[0].strip() # remove comments
          last_line = last_line.split(":")[0].strip() # remove ": sync_write"
          assert first_line.strip().startswith("always") and last_line.strip() == "end", (
                    "Always block should start with 'always' and end with 'end'"
                  )
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
      log_fn("Always extraction: filename={}, modulename={}".format(
        filename, module_name))
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