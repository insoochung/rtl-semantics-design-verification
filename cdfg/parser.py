import os
import json
import sys
import codecs
from zipfile import ZipFile
from glob import glob

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from cdfg.utils import get_log_fn


def get_verible_parsed_rtl(parsed_dir,
                           orig_dir=None,
                           verbose=True):
  """Return a dict of parsed verible json trees each read as a dict

  This reads the verible json tree files in the parsed_dir
  Args:
  parsed_dir -- the directory where the verible json tree files are stored
  orig_dir -- the directory where the original RTL files are stored, this will
    override the directory specified in the json file
  verbose -- whether to print to stdout

  Returns:
  A dict of verible json trees with keys that point to original RTL files.
  """
  log_fn = get_log_fn(verbose)

  log_fn(f"-- Start: attempting to parse files in {parsed_dir} --")
  json_files = glob(f"{parsed_dir}/*.json")
  if len(json_files) == 0:
    zip = glob(f"{parsed_dir}/*.zip")
    log_fn(f"No json files found in {parsed_dir}, attempting to unzip {zip}")
    assert len(zip) == 1, (
        f"Only one zip file should be present: {zip}")
    zip = zip[0]
    with ZipFile(zip, "r") as z:
      z.extractall(parsed_dir)
    json_files = glob(f"{parsed_dir}/*.json")

  assert len(json_files) > 0, f"Still, no json files found in {parsed_dir}"
  res = {}
  for json_filepath in json_files:
    with codecs.open(json_filepath, "r", encoding="utf-8") as file:
      log_fn(f"Reading {json_filepath}")
      j = json.load(file)
      assert len(j.keys()) == 1, (
          f"Only one root key of path to original RTL file should be present: "
          f"{j.keys()}")
      key = list(j.keys())[0]
      orig_filepath = key
      if orig_dir is not None:
        orig_filepath = os.path.join(orig_dir, os.path.basename(key))
        log_fn(f"Original filepath overridden: {key} -> {orig_filepath}")
      res[orig_filepath] = j[key]
  log_fn(f"-- End: files parsed in {parsed_dir} --")

  return res
