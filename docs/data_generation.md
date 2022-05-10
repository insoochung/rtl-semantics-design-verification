# Data generation

## 0. Overview

To train a design2vec framework, we need datapoints in form of:
```json
{
  "input":
    {
      "test_params": "param1=val1,param2=val2,param3=val3,...",
      "branch": "branch_number"
    },
  "result(is_hit)": "0 or 1"
}
```

In order to generate such datapoints, we need to:

1. Simulate design to get coverage information per set of test parameters via URG reports.
2. Parse RTL representation into a set of ASTs using verible.
3. Convert ASTs into CDFGs (uses results from step 2).
4. Match the branches in URG reports to subpaths within CDFGs (uses results from step 1).
5. Generate data for NN training and evaluation (uses results from step 3, 4).

## 1. Obtaining branch coverage per design

`README.md` in each design's directory explains the process.
- [ibex_v1](../designs/ibex_v1/)

## 2. Converting RTL to AST

You can find AST files in `parsed_rtl` within the directory of each design. Use that, or refer to [this document](verible.md) to generate your own.

## 3. Converting AST to CDFG

This step will convert AST json files generated from previous step 2 to CDFGs.

```bash
python cdfg/constructor.py \
  --parsed_rtl_dir designs/ibex_v1/parsed_rtl/ \
  --rtl_dir designs/ibex_v1/ibex/rtl \
  --output_dir $DATA_DIR/d2v_data/graph
```

## 4. Matching branches to CDFG subpaths

The HTML coverage report generated from step 1 is converted into YAML files containing coverage information.

```bash
python coverage/extract_from_urg.py \
  --in_place \
  --report_dir ~/generated/simulated
```

## 5. Vectorize CDFG and branch coverage information

This final step vectorizes data so it's ready to be fed to NN.

```bash
# Vectorize CDFG to feed to GCN
python data/cdfg_datagen.py \
  --design_graph_filepath $DATA_DIR/d2v_data/graph/design_graph.pkl \
  --output_dir $DATA_DIR/generated/d2v_data/graph

# Vectorize test parameters and coverage data to feed to NN trainling loop
python data/tp_cov_datagen.py \
  --design_graph_dir $DATA_DIR/d2v_data/graph/ \
  --test_templates_dir designs/ibex_v1/test_templates/ \
  --output_dir $DATA_DIR/generated/d2v_data/tp_cov
```