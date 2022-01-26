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
