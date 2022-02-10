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

<!-- ### Parse IBEX RTL

```bash
# Run CDFG generation and RTL reformatting.
# - Reformatting is required to match branches in coverage reports with nodes
#   in parsed CDFGs.
python parser/parser.py

# Swap RTL code with reformatted RTL.


``` -->
