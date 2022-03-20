# Steps to generate ASTs

1. Build and install verible commands according to the documents [here](third_party/verible/). This [script](https://github.com/google/riscv-dv/blob/07606315fb0ce03e1ecfbbf9e846e0385aeaacd9/verilog_style/build-verible.sh) provides a good reference.


2. Parse RTL files using verible. The json files are utilized by CDFG constructor, and tree files are more readable.

```bash
# A. Parse design to json format
for sv in $RTL_DIR/*sv
do
  verible-verilog-syntax  $sv  --printtree --export_json > $PARSE_RESULTS_DIR/$(basename -- $sv).json
done

# B. Parse design to tree format
for sv in $RTL_DIR/*sv
do
  verible-verilog-syntax  $sv  --printtree > $PARSE_RESULTS_DIR/$(basename -- $sv).tree
done
``````
