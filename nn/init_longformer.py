import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn.train import set_model_flags, run_with_seed
from nn.att_layers import init_longformer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_model_flags(parser, set_required=False)
    args = vars(parser.parse_args())
    print(f"Received arguments: {args}")
    run_with_seed(args, run_fn=init_longformer)
