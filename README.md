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

See [this document](./docs/data_generation.md) for details.

### Train model

After all data is prepared, you can train a model.

```bash
python nn/train.py \
  --graph_dir $DATA_DIR/generated/d2v_data/graph \ # See ./docs/data_generation.md to generate this
  --tp_cov_dir $DATA_DIR/generated/d2v_data/tp_cov \ # See ./docs/data_generation.md to generate this
  --batch_size 256 \ # Adjust this if you run into an OOM error.
  --split_ratio $SPLIT \ # Comma separated numbers that add up to 1 (e.g. 0.7,0.2,0.1)
  --seed 123 \ # Random seed for reproducibility
  --append_seed_to_ckpt_dir \ # This will add seed information to the checkpoint directory name
  --ckpt_dir $DATA/ckpts/split_$SPLIT/ \  # Where to save model checkpoints
  -tf_data_dir $DATA_DIR/nn_dataset \
  --generate_data # Previously generated data is processed to fit NN data pipeline, only required once
```