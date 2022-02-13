# Learning semantic representation of design for verification

## How to

### Setup dependencies

#### Base dependencies
```bash
# Pull submodules
git submodule update --init --recursive

# Install python dependencies (tested with python 3.6)
pip install -r requirements.txt
```

#### Simulator dependencies
```bash
# Set required environment variables.
source set_env_vars.sh
source set_vcs_vars.sh # This should be modified per environment.

# Download and extract IBEX toolchain
./setup_toolchain.sh

# Build spike ISS for co-simulation.
apt-get install device-tree-compiler
./setup_spike.sh

```

### Construct CDFGs

#### Parse RTL using verible

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

### Run simulations for data generation

TODO

### Parse simulation reports to training data points

TODO