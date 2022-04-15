import tensorflow as tf
import numpy as np
import sys
import os
import json
import yaml
import argparse
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.utils import TestParameterVocab, BranchVocab
from nn.models import Design2VecBase
from data.cdfg_datagen import GraphHandler
from cdfg.constructor import DesignGraph, Module
from nn.train import convert_to_tf_dataset, get_d2v_model

def output_yaml(vocab, tp, cp_idx):
  de_normalized_tp = vocab.de_normalize(tp)
  adjusted_test_name = f"adjusted_test_{cp_idx}"
  tp_str = f"description: {adjusted_test_name}\n gcc_opts: -mno-strict-align \n"
  tp_str += "\n".join(f"+{k}={v}" for k, v in de_normalized_tp.items())
  gen_opts_str = [" " * 4 + l for l in tp_str.split("\n")]
  gen_opts_str = "\n".join(["  gen_opts: >"] + gen_opts_str)
  gen_opts_str += f"""
  gen_test: riscv_ml_test
  iterations: 1
  no_iss: true
  no_post_compare: true
  rtl_test: core_ibex_reset_test
  template_name: base.20220228.yaml
  test: {adjusted_test_name}
  """
  gen_opts_str = (f"test: {adjusted_test_name} \n") + gen_opts_str
  with open(f"{adjusted_test_name}.yaml", "w+") as f:
    f.write(gen_opts_str)

def adjust_tp(graph_dir, ckpt_dir, tp_cov_dir, cp_idx):
  vocab_files = glob(os.path.join(tp_cov_dir, "vocab.*.yaml"))
  vocab = TestParameterVocab(vocab_filepath=vocab_files[1])
  tp_size = len(vocab.tokens)

  tp = np.random.rand(tp_size)
  design_graph_filepath = os.path.join(graph_dir, "design_graph.pkl")
  graph_handler = GraphHandler(design_graph_filepath, output_dir=graph_dir)
  graphs = graph_handler.get_dataset()
  bvocab = BranchVocab(os.path.join(tp_cov_dir, "vocab.branches.yaml"))

  model_config_file = os.path.join(ckpt_dir, "meta.json")
  model_config_json = open(model_config_file)
  model_config = json.load(model_config_json)
  model_config = model_config["model_config"]
  model = get_d2v_model(**model_config)
  model.load_weights(os.path.join(ckpt_dir, "model.ckpt"))

  is_hit = 0
  lr = 1
  step_threshold = 50
  for step in range(step_threshold):
    if step == step_threshold - 1:
      while(is_hit < .9):
        x = update_tp(tp, cp_idx, graphs, bvocab, tp_size)
        with tf.GradientTape() as g:
          g.watch(x)
          y = model.call(x)
          loss = y[[0]] - 1
        is_hit = y[[0]]
        dy_dx = g.gradient(loss, x)["test_parameters"]
        tp += dy_dx * lr
    else:
      x = update_tp(tp, cp_idx, graphs, bvocab, tp_size)
      with tf.GradientTape() as g:
        g.watch(x)
        y = model.call(x)
        loss = y[[0]] - 1
      is_hit = y[[0]]
      dy_dx = g.gradient(loss, x)["test_parameters"]
      tp += dy_dx * lr
  tp = [1 if i > 1 else 0 if i < 0 else i for i in tp.numpy()[0]]
  output_yaml(vocab, tp, cp_idx)
def update_tp(tp, cp_idx, graphs, bvocab, tp_size):
    midx = bvocab.get_module_index(cp_idx)
    mask_length = graphs[0].n_nodes
    mask = bvocab.get_mask(cp_idx, midx, mask_length=mask_length)
    mask = np.ma.make_mask(mask)
    is_hit = np.array([1])
    gidx = np.array([midx])
    cp = np.array([cp_idx])
    if isinstance(tp, tf.Tensor):
      tp = [1 if i > 1 else 0 if i < 0 else i for i in tp.numpy()[0]]
    ret = {
      "test_parameters": np.array([tp]).reshape(1, tp_size),
      "is_hits": is_hit.reshape(1, 1),
      "coverpoint_mask": mask.reshape(1, mask_length),
      "graph": gidx.reshape(1,),
      "coverpoint": cp.reshape(1,)
    }

    #formating from dictionary of numpy arrays to dictionary of tensors
    tf_ret = ret
    for k, v in tf_ret.items():
      tf_ret[k] = tf.constant(v, dtype=v.dtype)
    
    return tf_ret
if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("-gd", "--graph_dir", type=str, required=True,
                      help="Directory of the graph dataset.")
    parser.add_argument("-cd", "--ckpt_dir", type=str, required=True,
                      help="Directory to save the checkpoints.")       
    parser.add_argument("-td", "--tp_cov_dir", type=str,
                      help="Directory of the test parameter coverage dataset.")
    parser.add_argument("-cp", "--cp_idx", type=int,
                      help="Coverpoint index.")
    args = parser.parse_args()
    adjust_tp(**vars(args))

