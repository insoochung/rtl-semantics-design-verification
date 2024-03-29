#!/usr/bin/python3
import os
import argparse
from bs4 import BeautifulSoup
import re
import json
import yaml
import itertools

BRANCH_TEMP_FILENAME = ".branch_temp.html"
TABLE_TEMP_FILENAME = ".table_temp.html"


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
    os.makedirs(output_dir, exist_ok=True)
    with create_branch_temp(output_dir) as branch_fp, create_branch_table_temp(
        output_dir
    ) as table_fp:
        with open(html_file, "r") as file:
            for line in file:
                if "Branch Coverage for" in line:
                    write_mode = 1
                    branch_in_file = True
                elif "pre class" in line and write_mode == 1:
                    write_mode = 2
                elif "Assert Coverage for" in line or "Line Coverage for" in line:
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
        soup = BeautifulSoup(file.read(), "html.parser")

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            entries = row.find_all("td")
            if len(entries) > 0:
                branch_dict["branch_type"].append(entries[0].string)
                branch_dict["line_pos"].append(entries[1].string)
                branch_dict["cov_score"].append(entries[4].string)

    return branch_dict


def write_branch_yaml_file(branch_dict, output_dir, cov_name="default_cov"):
    """Write the branch coverage data to a YAML file"""
    with open(os.path.join(output_dir, BRANCH_TEMP_FILENAME), "r") as file:
        soup = BeautifulSoup(file.read(), "html.parser")
    module_name = soup.find_all("span", class_="titlename")
    module_name = module_name[0].find("a").text
    code_lines = soup.find_all("pre")
    tables = soup.find_all("table", class_="noborder")
    yaml_entry_strs = []
    assert len(tables) == len(code_lines), (
        f"Mismatch between number of tables and parsed data: "
        f"# tables({len(tables)}) != # code_lines({len(code_lines)})"
    )

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
            assert (
                prev_branch_num == branch_num - 1
            ), f"Branch number not in order: {branch_num} != {prev_branch_num + 1}"
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
                        assert len(entry_str) == 2, f"Unexpected format of {entry_str}"
                        entry_str = entry_str[1]
                    trace.append(entry_str.strip())
                elif entry_str == "-":
                    trace.append("X")
                else:
                    assert entry_str in ["Covered", "Not Covered"], entry_str
                    assert i == len(entries) - 1, "Coverage data should be last"
                    cov = entry_str == "Covered"
            yaml_entry["traces"].append({"trace": trace, "cov": cov})
            assert len(yaml_entry["traces"][-1]["trace"]) == len(
                yaml_entry["line_num"]
            ), (
                f"Mismatch between number of lines and trace length: "
                f"# lines({len(yaml_entry['line_num'])}) != "
                f"# trace({len(yaml_entry['traces'][-1]['trace'])})"
                f"{len(entries)}"
            )
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
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith("_temp.html"):
                temp_fp = os.path.join(output_dir, filename)
                if os.path.exists(temp_fp):
                    os.remove(temp_fp)


def extract_from_directory(
    report_dir, output_dir="", in_place=False, coverage_type="branch"
):
    """Extract branch coverage data from a directory of coverage reports"""
    assert in_place or output_dir, "Please specify an output directory"
    file_prefix = "mod" if coverage_type == "branch" else "grp"
    for root, _, filenames in os.walk(report_dir):
        if in_place:
            output_dir = os.path.join(root, "extracted")
        for filename in filenames:
            if not filename.endswith(".html") or not filename.startswith(file_prefix):
                continue
            cov_name = filename[:-5]  # removes trailing .html
            report_path = os.path.join(root, filename)
            extract_from_file(report_path, output_dir, coverage_type, cov_name)


def get_cross_coverage_attr_names(row):
    attr_names = []
    cols = row.findChildren("td")
    for col in cols:
        if col.text in ["COUNT", "AT LEAST"]:
            break
        attr_names.append(col.text)
    return attr_names


def get_uncovered_bin_names(col, coverage, attr_name=None):
    bin_name = str(col.string)
    if bin_name.startswith("["):
        bin_name = bin_name[1:-1]  # Unwrap []
        if "," in bin_name:
            bin_names = [x.strip() for x in bin_name.split(",")]
        elif "-" in bin_name:
            start, end = [x.strip() for x in bin_name.split("-")]
            assert start.startswith("auto[") and end.startswith(
                "auto["
            ), f"Assumption does not hold for {bin_name}"
            start_idx = int(start[5:-1])
            end_idx = int(end[5:-1])
            bin_names = [f"auto[{i}]" for i in range(start_idx, end_idx + 1)]
        else:
            bin_names = [bin_name]
    else:
        if "*" in bin_name:
            for key in coverage.keys():
                if attr_name in key:
                    bin_names = list(coverage[key].keys())
                    break
        else:
            bin_names = [bin_name]
    return bin_names


def extract_functional_coverage(report_path, output_dir, cov_name="default_func_cov"):
    with open(report_path, "r") as file:
        soup = BeautifulSoup(file.read(), "html.parser")
    # var_table = soup.find("span", text=re.compile(
    #     r"Variables for Group.*")).find_next("table")
    # var_summary = {}
    # for row in var_table.find_all("tr")[1:]:
    #   cols = row.find_all("td")
    #   var_summary[cols[0].text.strip()] = int(cols[1].text)
    print(f"Extracting functional coverage from: {report_path}")
    coverpoints = [
        span
        for span in soup.find_all("span", attrs={"id": re.compile(r".*")})
        if "Group" not in span.text
    ]
    coverpoint_ids = set(coverpoint["id"] for coverpoint in coverpoints)
    coverage = {}
    for coverpoint in coverpoints:
        coverpoint_id = coverpoint["id"]
        coverage[coverpoint_id] = {}
        next_span = coverpoint
        covered_tables = []
        uncovered_tables = []
        while True:  # Get coverage table for this coverpoint
            next_span = next_span.find_next("span")
            if next_span is None:
                break
            next_span_id = next_span["id"] if "id" in next_span.attrs else None
            if next_span_id in coverpoint_ids:
                break
            if next_span.text in ["Covered bins", "Bins"]:
                covered_tables.append(next_span.find_next("table"))
            elif next_span.text in ["Uncovered bins", "Element holes"]:
                uncovered_tables.append(next_span.find_next("table"))

        for table in covered_tables:  # Covered tables
            attr_names = []
            for row in table.findChildren("tr"):
                if row["class"][0] == "sortablehead":
                    if "Cross" in coverpoint.text:
                        # Get bin names differently if cross coverage
                        attr_names = get_cross_coverage_attr_names(row)
                    continue
                cols = row.findChildren("td")
                if len(attr_names) == 0:
                    cnt_idx = 1
                    bin_name = str(cols[0].string)
                else:
                    cnt_idx = len(attr_names)
                    bin_name = ".".join(
                        [cols[i].string for i in range(len(attr_names))]
                    )
                assert not bin_name.startswith("["), bin_name
                assert not any(x in bin_name for x in ["*", ","]), bin_name
                coverage[coverpoint_id][bin_name] = int(cols[cnt_idx].string)
            if len(attr_names):
                coverage[coverpoint_id]["cross_attributes"] = attr_names

        for table in uncovered_tables:
            attr_names = []
            for row in table.findChildren("tr"):
                if row["class"][0] == "sortablehead":
                    if "Cross" in coverpoint.text:
                        # Get bin names differently if cross coverage
                        attr_names = get_cross_coverage_attr_names(row)
                    continue
                cols = row.findChildren("td")
                if len(attr_names) == 0:
                    bin_names = get_uncovered_bin_names(cols[0], coverage)
                else:
                    all_bin_names = []
                    for i, attr_name in enumerate(attr_names):
                        bin_names = get_uncovered_bin_names(
                            cols[i], coverage, attr_name
                        )
                        all_bin_names.append(bin_names)
                    all_bin_names = [
                        ".".join(s) for s in itertools.product(*all_bin_names)
                    ]
                    bin_names = all_bin_names

                for bin_name in bin_names:
                    coverage[coverpoint_id][bin_name] = 0

            if len(attr_names):
                coverage[coverpoint_id]["cross_attributes"] = attr_names

    yaml_path = os.path.join(output_dir, f"{cov_name}.yaml")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing coverage to: {yaml_path}")
    with open(yaml_path, "w") as file:
        yaml.dump(coverage, file)


def extract_from_file(
    report_path: str,
    output_dir: str,
    coverage_type: str = "branch",
    cov_name: str = "default_coverage",
):
    if coverage_type == "branch":
        if write_branch_to_temp(report_path, output_dir) == 1:
            print(f"Branch found in: {report_path}")
            branch_dict = set_branch_dict(output_dir)
            write_branch_yaml_file(branch_dict, output_dir, cov_name=cov_name)
        else:
            print(f"Branch or module name not found in: {report_path}")
    elif coverage_type == "functional":
        extract_functional_coverage(report_path, output_dir, cov_name)
    else:
        raise NotImplementedError(f"Coverage type {coverage_type} not supported")
    cleanup_temp_files(output_dir)


def extract(
    report_path: str = "",
    report_dir: str = "",
    output_dir: str = "",
    in_place: bool = False,
    coverage_type: str = "branch",
):
    """Extract coverage from the HTML reports"""
    if report_path and report_dir or (not report_path and not report_dir):
        print(
            "Please specify either a report path or a report directory "
            "(but not both)."
        )
        exit(1)
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "branch_cov")

    if report_dir:
        extract_from_directory(report_dir, output_dir, in_place, coverage_type)
    else:
        assert not in_place, "In-place extraction is not supported for single file"
        extract_from_file(report_path, output_dir, coverage_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rp",
        "--report_path",
        required=False,
        help=("Path to a coverage HTML path, generated with " "ibex testbench"),
    )
    parser.add_argument(
        "-rd",
        "--report_dir",
        required=False,
        help=(
            "Path to the directory (e.g. urgReport) which "
            "contains multiple coverage HTML reports"
        ),
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        default="",
        help=(
            "Path to the directory where the parsed coverage "
            "YAML files will be written"
        ),
    )
    parser.add_argument(
        "-ip",
        "--in_place",
        action="store_true",
        default=False,
        help=(
            "Whether to place the parsed coverage YAML files "
            "in the same directory as the original reports"
        ),
    )
    parser.add_argument(
        "-ct",
        "--coverage_type",
        default="branch",
        choices=["branch", "functional"],
        help="Type of coverage to extract",
    )

    args = parser.parse_args()
    extract(
        args.report_path,
        args.report_dir,
        args.output_dir,
        args.in_place,
        args.coverage_type,
    )


if __name__ == "__main__":
    main()
