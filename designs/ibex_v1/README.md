# IBEX v1

This document explain steps to simulate the design and parsing the design to ASTs (RTL2AST). These steps are required to obtain datapoints to train and test design2vec frameworks.

## Simulation

Simulation is required to get actual coverage with respect design and test parameters. This section explains how to simulate the design and get URG coverage reports.

1. Set required environment variables.

```bash
# Set required environment variables.
source set_env_vars.sh
source set_vcs_vars.sh # This should be modified per environment.
```

2. Download and/or build dependencies.

```bash
# Download and extract IBEX toolchain
./setup_toolchain.sh

# Build spike ISS for co-simulation.
apt-get install device-tree-compiler
./setup_spike.sh

```

3. Simulate and generate URG reports.

```bash
cd ibex/dv/uvm/core_ibex

make SIMULATOR=vcs ISS=spike ITERATIONS=1 COV=1

# You can find URG coverage report below.
ls out/rtl_sim/urgReport

```

## Parsing design to ASTs

1. Build and install verible commands according to the documents [here](../../third_party/verible/).


2. Parse RTL files using verible.

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
```

## Automated test generation and test simulation

1. Follow first two steps of `Simulation` to set all environment variables and build dependencies.

2. Generate tests.

```bash
python generate_tests.py --template_dir ./test_templates/ --output_dir $GENERATED_TESTS_DIR --num_tests $NUM_TESTS
# If you want to run a generated test
# 1. Replace 'ibex/dv/uvm/core_ibex/riscv_dv_extension/testlist.yaml' with the generated test.
# 2. Follow step 3 in `Simulation` section.
```

3. Run tests

```bash
python run_tests.py --tests_dir $GENERATED_TESTS_DIR --output_dir $SIMULATION_RESULTS_DIR --verification_dir ibex/dv/uvm/core_ibex
```

## Next

[Top-level document](../../README.md) explains how to generate data, once simulation results and ASTs become available.