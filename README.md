# Learning semantic representation of design for verification

## How to

### Setup dependencies

#### Base dependencies

This step will install commonly required dependencies for using this repository.

```bash
# Pull submodules
git submodule update --init --recursive

# Install python dependencies (tested with python 3.6)
pip install -r requirements.txt
```

### Generate data

#### 0. Overview

To train a design2vec framework, we need datapoints in form of:
```json
{
  "input":
    {
      "test_params": "param1=val1,param2=val2,param3=val3,...",
      "branch": "{module_name}/{line_num}/{trace_num}"
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

#### 1. Obtaining branch coverage per design

`README.md` in each design's directory explains the process.
- [ibex_v1](designs/ibex_v1/)

#### 2. Converting RTL to AST

This too is explained in each `README.md`' of designs.
- [ibex_v1](designs/ibex_v1/)

#### 3. Converting AST to CDFG

This step will convert AST json files generated from previous step 2 to CDFGs.

```bash
python cdfg/constructor.py --parsed_rtl_dir $PARSE_RESULTS_DIR --rtl_dir $RTL_DIR --output_dir $CDFG_DIR
```

#### 4. Matching branches to CDFG subpaths

The HTML coverage report generated from step 1 is converted into YAML files containing coverage information.

```bash
python coverage/extract_from_urg.py --in_place --report_dir $URG_REPORT_DIR
```

#### 5. Generate training and test data

This final step creates data in format required for design2vec training.

```bash
# Vectorize CDFG to feed to GCN
python data/cdfg_datagen.py --design_graph_filepath $CDFG_DIR/design_graph.pkl --output_dir $NN_DATA_DIR/cdfgs
# Vectorize test parameters and coverage data to feed to NN trainling loop
python data/tp_cov_datagen.py --design_graph_dir $CDFG_DIR --test_templates_dir $TEST_TEMPLATES_DIR --output_dir $NN_DATA_DIR/tp_coverage
```