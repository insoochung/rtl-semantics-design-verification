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

#### Overview

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
3. Convert ASTs into CDFGs.
4. Match the branches in URG reports to subpaths within CDFGs.

#### Design simulation and RTL2AST

`README.md` in each design's directory explains the process.
- [ibex_v1](designs/ibex_v1/)

#### Parse simulation reports to training data points

TODO