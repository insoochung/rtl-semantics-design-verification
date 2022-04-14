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


def accumulate_gradients_and_loss(grads_list, losses, accum_steps):
  # Accumulate gradients
  grads = [tf.zeros_like(g) for g in grads_list[0]]
  for g in grads_list:
    for idx, _ in enumerate(g):
      grads[idx] += g[idx]
  for idx, _ in enumerate(grads):
    grads[idx] /= accum_steps
  loss = sum(losses) / accum_steps
  return grads, loss


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

  model = Design2VecBase(params)
  cdfg_reader = model.cdfg_reader
  save_path = os.path.join(
      params["ckpt_dir"], params["ckpt_name"], "cdfg_reader")

  optimizer = get_optimizer(params)
  loss_obj = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=params["label_smoothing"])
  train_vars = []
  for v in cdfg_reader.trainable_variables:
    if "pooler" in v.name or "word_embeddings" in v.name:
      continue
    train_vars.append(v)

  batch_size = params["batch_size"] / accum_steps
  assert batch_size - int(batch_size) == 0
  batch_size = int(batch_size)
  train_progress = tqdm(range(params["train_steps"]))
  last_saved = 0

  print(f"Pretraining (save_path: \"{save_path}\")")
  pretrain_loss = []
  best_loss = 1e10
  for step in train_progress:
    grads_list = []
    losses = []
    for inner_step in range(accum_steps):
      with tf.GradientTape() as tape:
        y, y_labels = cdfg_reader.pretrain_forward(
            batch_size=batch_size)
        loss = loss_obj(y_labels, y)
        losses.append(loss)
        grads_list.append(tape.gradient(loss, train_vars))
    grads, loss = accumulate_gradients_and_loss(grads_list, losses,
                                                accum_steps)
    pretrain_loss.append(loss)
    train_progress.set_description(
        f"loss: {loss:.2f}, last_saved: {last_saved} -> ")
    # Apply gradients
    optimizer.apply_gradients(zip(grads, train_vars))
    del grads_list, grads
    if step > 0 and step % save_steps == 0:
      loss = np.mean(pretrain_loss[:save_steps])
      if best_loss < loss:
        continue  # Only save if this loss is better than the best loss
      print(f"Saving cdfg_reader (step: {step}, "
            f"mean_loss (last {save_steps} steps): {loss:.2f})")
      best_loss = loss
      last_saved = step
      model.save_weights(save_path)
  model.save_weights(save_path)

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
