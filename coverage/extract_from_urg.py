#!/usr/bin/python3
import os
import argparse
from bs4 import BeautifulSoup
import re
import json
import yaml

BRANCH_TEMP_FILENAME = ".branch_temp.html"
TABLE_TEMP_FILENAME = ".table_temp.html"


def extract_from_directory(report_dir, output_dir):
  """Extract branch coverage data from a directory of coverage reports"""
  for filename in os.listdir(report_dir):
    if not filename.endswith(".html") or not filename.startswith("mod"):
      continue

    cov_name = filename[:-5]  # removes trailing .html
    report_path = os.path.join(report_dir, filename)
    if write_branch_to_temp(report_path, output_dir) == 1:
      print(f"Branch found in: {report_path}")
      branch_dict = set_branch_dict(output_dir)
      write_yaml_file(branch_dict, output_dir, cov_name=cov_name)
    else:
      print(f"Branch or module name not found in: {report_path}")


def create_branch_temp(output_dir):
  """Create a temporary file to store the branch related HTML"""
  temp_filepath = os.path.join(output_dir, BRANCH_TEMP_FILENAME)
  try:
    file_temp = open(temp_filepath, "w+")
  except:
    print(f"Failed to open temporary file: {file_temp}")
    exit
  return file_temp


def create_branch_table_temp(output_dir):
  """Create a temporary file to store the branch coverage HTML tables"""
  temp_filepath = os.path.join(output_dir, TABLE_TEMP_FILENAME)
  try:
    file_temp = open(temp_filepath, "w+")
  except:
    print(f"Failed to open temporary file: {file_temp}")
    exit
  return file_temp


def write_branch_to_temp(html_file, output_dir):
  """Gather relevant branch coverage data from HTML file and write to temp"""
  write_mode = 0
  branch_in_file = False
  modulename_in_file = False

  print(f"Parsing branch from: {html_file}")
  with create_branch_temp(output_dir) as branch_fp, \
          create_branch_table_temp(output_dir) as table_fp:
    with open(html_file, "r") as file:
      for line in file:
        if "Branch Coverage for" in line:
          write_mode = 1
          branch_in_file = True
        elif "pre class" in line and write_mode == 1:
          write_mode = 2
        elif ("Assert Coverage for" in line or "Line Coverage for" in line):
          write_mode = 0

        if write_mode == 1:
          table_fp.write(line)
        elif write_mode == 2:
          branch_fp.write(line)
        elif line.strip().startswith("<span class=titlename>Module :"):
          table_fp.write(line)
          branch_fp.write(line)
          modulename_in_file = True

  return branch_in_file and modulename_in_file


def set_branch_dict(output_dir):
  """Gather branch coverage data from the temporary files and return a dict"""
  branch_dict = {"branch_type": [], "line_pos": [], "cov_score": []}
  # Open the temporary file which has the coverage table
  temp_filepath = os.path.join(output_dir, TABLE_TEMP_FILENAME)
  with open(temp_filepath, "r") as file:
    parsed_html = BeautifulSoup(file.read(), "html.parser")

  tables = parsed_html.find_all("table")
  for table in tables:
    rows = table.find_all("tr")
    for row in rows:
      entries = row.find_all("td")
      if len(entries) > 0:
        branch_dict["branch_type"].append(entries[0].string)
        branch_dict["line_pos"].append(entries[1].string)
        branch_dict["cov_score"].append(entries[4].string)

  return branch_dict


def write_yaml_file(branch_dict, output_dir, cov_name="default_cov"):
  """Write the branch coverage data to a YAML file"""
  with open(os.path.join(output_dir, BRANCH_TEMP_FILENAME), "r") as file:
    parsed_html = BeautifulSoup(file.read(), "html.parser")
  module_name = parsed_html.find_all("span", class_="titlename")
  module_name = module_name[0].find("a").text
  code_lines = parsed_html.find_all("pre")
  tables = parsed_html.find_all("table", class_="noborder")
  yaml_entry_strs = []
  assert len(tables) == len(code_lines), (
      f"Mismatch between number of tables and parsed data: "
      f"# tables({len(tables)}) != # code_lines({len(code_lines)})")

  for i, (table, code_line) in enumerate(zip(tables, code_lines)):
    yaml_entry = {}

    # Extract line numbers of branch locations
    line_nums = []
    lines = str(code_line).split("\n")
    prev_branch_num = 0
    prev_line_num = 0
    for i, line in enumerate(lines):
      _line = line.strip()
      branch_num = re.search(r'<font color="(red|green)">-\d+-', _line)
      if not branch_num:  # Try to get line number
        pl_cand = _line.split()
        if not pl_cand:
          continue
        if pl_cand[0].isnumeric():
          prev_line_num = int(pl_cand[0])
        continue
      branch_num = int(branch_num.group(0).split("-")[1])
      assert prev_branch_num == branch_num - 1, (
          f"Branch number not in order: {branch_num} != {prev_branch_num + 1}")
      line_nums.append(prev_line_num)
      prev_branch_num = branch_num

    # Get first line of the code snippet
    init_code_line = -1
    for l in str(code_line).split("\n"):
      if l.split()[0].isnumeric():
        init_code_line = l.split()[0]
        break

    _idx = branch_dict["line_pos"].index(init_code_line)
    yaml_entry["line_num"] = [int(n) for n in line_nums]
    yaml_entry["branch_type"] = branch_dict["branch_type"][_idx]
    yaml_entry["traces"] = []
    rows = table.find_all("tr")
    for row in rows:
      # Aggregate trace-level coverage
      entries = row.find_all("td")
      if len(entries) == 0:
        continue
      trace = []
      cov = False
      if len(entries) == 2 and ")->(" in entries[0].string:
        # Convert right arrow connected traces to dense format
        dense_entries = ["-"] * len(line_nums) + [entries[-1]]
        for entry in entries[0].string.split("->"):
          entry = entry.strip()[1:-1]  # strip parentheses
          entry = entry.split(".")
          if len(entry) == 2:
            idx, entry = entry
          else:
            if entry[0][0] == "!":
              idx = int(entry[0][1:])
              entry = "false"
            else:
              idx = int(entry[0])
              entry = "true"
          idx = int(idx) - 1
          dense_entries[idx] = entry.strip()
        entries = dense_entries

      for i, entry in enumerate(entries):
        entry_str = entry
        if not isinstance(entry_str, str):
          entry_str = str(entry.string)
        if entry_str not in ["Covered", "Not Covered", "-"]:
          # Handle condition
          if entry_str.startswith("CASEITEM"):
            entry_str = entry_str.split(":")
            assert len(entry_str) == 2, (
                f"Unexpected format of {entry_str}")
            entry_str = entry_str[1]
          trace.append(entry_str.strip())
        elif entry_str == "-":
          trace.append("X")
        else:
          assert entry_str in ["Covered", "Not Covered"], entry_str
          assert i == len(entries) - 1, "Coverage data should be last"
          cov = entry_str == "Covered"
      yaml_entry["traces"].append({"trace": trace, "cov": cov})
      assert (len(yaml_entry["traces"][-1]["trace"])
              == len(yaml_entry["line_num"])), (
          f"Mismatch between number of lines and trace length: "
          f"# lines({len(yaml_entry['line_num'])}) != "
          f"# trace({len(yaml_entry['traces'][-1]['trace'])})"
          f"{len(entries)}")
    # Stringify to enable comparison and avoid duplicate entries
    yaml_entry_str = json.dumps(yaml_entry, sort_keys=True)
    if yaml_entry_str not in yaml_entry_strs:
      yaml_entry_strs.append(yaml_entry_str)

  yaml_entries = [json.loads(s) for s in yaml_entry_strs]
  for i, entry in enumerate(yaml_entries):
    entry["id"] = i
  yaml_content = {"module_name": module_name, "coverages": yaml_entries}
  yaml_path = os.path.join(output_dir, f"{cov_name}.yaml")
  with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)
  print(f"Wrote coverage of '{module_name}' to: {yaml_path}")


def cleanup_temp_files(output_dir):
  """Remove the temporary files"""
  for filename in os.listdir(output_dir):
    if filename.endswith("_temp.html"):
      temp_fp = os.path.join(output_dir, filename)
      if os.path.exists(temp_fp):
        os.remove(temp_fp)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-rp", "--report_path", required=False,
                      help=("Path to a coverage HTML path, generated with "
                             "ibex testbench"))
  parser.add_argument("-rd", "--report_dir", required=False,
                      help=("Path to the directory (e.g. urgReport) which "
                             "contains multiple coverage HTML reports"))
  parser.add_argument("-od", "--output_dir",
                      default="generated/branch_cov",
                      help=("Path to the directory where the parsed coverage "
                             "YAML files will be written"))

  args = parser.parse_args()
  report_path = args.report_path
  report_dir = args.report_dir
  output_dir = args.output_dir
  if report_path and report_dir or (not report_path and not report_dir):
    print("Please specify either a report path or a report directory "
          "(but not both).")
    exit(1)
  if not output_dir:
    output_dir = os.path.join(os.getcwd(), "branch_cov")
  os.makedirs(output_dir, exist_ok=True)

  if report_dir:
    extract_from_directory(report_dir, output_dir)
  else:
    write_branch_to_temp(report_path, output_dir)
    branch_dict = set_branch_dict(output_dir)
    write_yaml_file(branch_dict, output_dir)

  cleanup_temp_files(output_dir)


if __name__ == "__main__":
  main()
