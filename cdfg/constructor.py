import os
import argparse
import pickle

from graph import RtlFile
from parser import get_verible_parsed_rtl


def construct_cdfgs(parsed_rtl_dir, rtl_dir, output_dir):
  parsed_rtl = get_verible_parsed_rtl(parsed_rtl_dir,
                                      orig_dir=rtl_dir)

  os.makedirs(output_dir, exist_ok=True)
  for filepath, verible_tree in parsed_rtl.items():
    filename = os.path.basename(filepath)
    print(f"-- Constructing CDFGs from: {filepath} --")
    rtl_file_obj = RtlFile(verible_tree, filepath)
    if rtl_file_obj.num_always_blocks == 0:
      print(f"-- Skipping {filename} because it has no always blocks --\n")
      continue
    pkl_name = filename.replace(".sv", ".rtlfile.pkl")
    with open(os.path.join(output_dir, pkl_name), "wb") as f:
      pickle.dump(rtl_file_obj, f)
    print(f"-- CDFGs successfully constructed! --\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-pr", "--parsed_rtl_dir", required=True,
                      help=("Directory where verible generated ASTs "
                            "(parsed from RTLs) are located in json format"))
  parser.add_argument("-rd", "--rtl_dir", required=True,
                      help="Directory where the original RTL files are "
                           "located")
  parser.add_argument("-od", "--output_dir", default="generated/cdfgs",
                      help="Directory where parsed CDFGs are saved")
  args = parser.parse_args()
  construct_cdfgs(**vars(args))
