import argparse

from graph import RtlFile
from parser import get_verible_parsed_rtl

def get_rtl_file_obj(filepath, verible_tree):
  return RtlFile(filepath, verible_tree)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-pr", "--parsed_rtl_dir", required=True,
    help=("Directory where verible generated ASTs (parsed from RTLs) "
           "are located in json format"))
  parser.add_argument("-rd", "--rtl_dir", required=True,
    help="Directory where the original RTL files are located")
  args = parser.parse_args()
  parsed_rtl = get_verible_parsed_rtl(
    args.parsed_rtl_dir, orig_dir=args.rtl_dir)
  for filepath, verible_tree in parsed_rtl.items():
    get_rtl_file_obj(verible_tree, filepath)
