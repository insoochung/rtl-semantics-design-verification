# Learning semantic representation of design for verification

## Quick start

```bash
# 1. Pull submodules
git submodule update --init --recursive

# 2. Install python dependencies (tested with python 3.6)
pip install -r requirements.txt

# 3. Run CDFG generation and RTL reformatting.
python parser/parser.py

# 4. Examine reformatted RTL for generated node information

vim reformatted/ibex_alu.sv # An example
...

```

## Parse using verible

```bash
# 1. Build and install verible commands according to the documents

# 2.a. Parse to json format
for sv in $RTL_DIR/*sv
do
  verible-verilog-syntax  $sv  --printtree --export_json > $PARSE_RESULTS_DIR/$(basename -- $sv).json
done

# 3.a. Parse to tree format
for sv in $RTL_DIR/*sv
do
  verible-verilog-syntax  $sv  --printtree > $PARSE_RESULTS_DIR/$(basename -- $sv).tree
done

```