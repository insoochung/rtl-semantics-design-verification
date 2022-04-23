import os
import sys
import argparse

from tqdm import tqdm
import numpy as np
import tensorflow as tf


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.cdfg_datagen import GraphHandler
from nn.models import Design2VecBase
from nn.train import set_model_flags, run_with_seed, get_optimizer
from nn.datagen import get_fake_inputs
from nn.utils import warmstart_pretrained_weights, save_model


def accumulate_gradients_and_loss(grads_list, losses, accum_steps, train_vars):
  # Accumulate gradients
  for g, v in zip(grads_list[0], train_vars):
    if g is None:
      assert 0, f"No gradient for {v.name}"
  grads = [tf.zeros_like(g) for g in grads_list[0]]
  for g in grads_list:
    for idx, _ in enumerate(g):
      grads[idx] += g[idx]
  for idx, _ in enumerate(grads):
    grads[idx] /= accum_steps
  loss = sum(losses) / accum_steps
  return grads, loss


def load_or_warmstart_model(save_path, params):
  if os.path.exists(os.path.dirname(save_path)):
    raise NotImplementedError(
        "Checkpoint directory already exists, and load from save not "
        "supported yet, use 'warmstart_dir' instead.")
  if not os.path.exists(os.path.dirname(save_path)):
    model = Design2VecBase(params)
    model(get_fake_inputs(params["tf_data_dir"]))
    if params["warmstart_dir"] is not None:
      warmstart_path = os.path.join(
          params["warmstart_dir"], params["ckpt_name"], "cdfg_reader")
      assert os.path.exists(os.path.exists(os.path.dirname(warmstart_path))), (
          f"{warmstart_path} not found")
      warmstart_pretrained_weights(model, warmstart_path)
    return model


def pretrain(params):
  graph_dir = params["graph_dir"]
  graph_handler = GraphHandler(output_dir=graph_dir)
  accum_steps = params["accum_steps"]
  save_steps = params["save_steps"]

  # Load the dataset
  cdfgs = graph_handler.get_dataset()
  params["graphs"] = cdfgs
  # Pretraining is for the attention model only.
  assert params["use_attention"]
  save_path = os.path.join(
      params["ckpt_dir"], params["ckpt_name"])

  model = load_or_warmstart_model(save_path, params)
  optimizer = get_optimizer(params)
  loss_obj = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=params["label_smoothing"])
  train_vars = []
  for v in model.cdfg_reader.trainable_variables:
    if "pooler" in v.name or "word_embeddings" in v.name:
      continue
    if v.name.startswith("design2_vec_base/cdfg_reader/feed_forward/"):
      continue
    train_vars.append(v)

  batch_size = params["batch_size"] / accum_steps
  assert batch_size - int(batch_size) == 0
  batch_size = int(batch_size)
  train_progress = tqdm(range(params["train_steps"]), initial=0,)
  last_saved = -1

  print(f"Pretraining (save_path: \"{save_path}\")")
  pretrain_loss = []
  best_loss = 1e10
  for step in train_progress:
    grads_list = []
    losses = []
    for inner_step in range(accum_steps):
      with tf.GradientTape() as tape:
        y, y_labels = model.cdfg_reader.pretrain_forward(
            batch_size=batch_size)
        loss = loss_obj(y_labels, y)
        losses.append(loss)
        grads_list.append(tape.gradient(loss, train_vars))
    grads, loss = accumulate_gradients_and_loss(grads_list, losses,
                                                accum_steps, train_vars)
    pretrain_loss.append(loss)
    train_progress.set_description(
        f"loss: {loss:.2f}, "
        f"(last_saved: step={last_saved}, loss={best_loss:.2f}) -> ")
    # Apply gradients
    optimizer.apply_gradients(zip(grads, train_vars))
    del grads_list, grads
    if step > 0 and step % save_steps == 0:
      loss = np.mean(pretrain_loss[:save_steps])
      if best_loss < loss:
        continue  # Only save if this loss is better than the best loss
      print(f"Saving model @ (step: {step}, "
            f"mean_loss (last {save_steps} steps): {loss:.2f})")
      best_loss = loss
      last_saved = step
      save_model(model, save_path)
  save_model(model, save_path)

  return {"pretrain_loss": pretrain_loss, "last_saved_step": last_saved}


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  set_model_flags(parser, set_required=False)
  parser.add_argument("--accum_steps", type=int, default=1)
  parser.add_argument("--train_steps", type=int, default=1000000)
  parser.add_argument("--save_steps", type=int, default=500)
  parser.add_argument("--label_smoothing", type=float, default=0.1)
  args = parser.parse_args()
  print(f"Received arguments: {args}")
  run_with_seed(vars(args), run_fn=pretrain)
