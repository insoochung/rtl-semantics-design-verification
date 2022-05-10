import os
import math
import json
import tensorflow as tf

with open(os.path.join(os.path.dirname(__file__),
                       "var_maps/pretrained_to_trained.json"), "r") as f:
  CKPT_VAR_NAME_SWAP = json.load(f)


class WarmUpThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, initial_learning_rate, decay_rate, decay_steps,
               warmup_steps):
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.decay_rate = decay_rate
    self.decay_steps = decay_steps

  def __call__(self, step):
    decay_factor = tf.constant(
        math.pow(self.decay_rate, 1 / self.decay_steps), dtype=tf.float32)
    warmup_steps_tf = tf.cast(self.warmup_steps, dtype=tf.float32)
    step_tf = tf.cast(step, dtype=tf.float32)
    lr = tf.cast(self.initial_learning_rate, dtype=tf.float32)

    cond = step <= self.warmup_steps
    warmup = lr * (step_tf / warmup_steps_tf)
    decay = lr * tf.math.pow(decay_factor, step_tf - warmup_steps_tf)
    return tf.cond(cond, lambda: warmup, lambda: decay)

  def get_config(self):
    config = {"initial_learning_rate": self.initial_learning_rate,
              "warmup_steps": self.warmup_steps,
              "decay_rate": self.decay_rate,
              "decay_steps": self.decay_steps}
    return config


def get_lr_schedule(
        lr, lr_scheme, decay_rate=0.90, decay_steps=500, warmup_steps=1000):
  if lr_scheme == "warmup_then_decay":
    lr_schedule = WarmUpThenDecay(lr, decay_rate, decay_steps, warmup_steps)
  elif lr_scheme == "constant":
    lr_schedule = tf.constant(lr, tf.float32)
  elif lr_scheme == "exp_decay":
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps, decay_rate)
  elif lr_scheme == "linear_decay":
    lr_schedule = tf.keras.optimizers.schedules.LinearDecay(
        lr, decay_steps, decay_rate)
  else:
    raise ValueError(f"Unknown lr_scheme: {lr_scheme}")

  return lr_schedule


def warmstart_pretrained_weights(model, ckpt_path):
  print(f"Warmstart pretrained weights from: {ckpt_path}")
  from tensorflow.python.training import py_checkpoint_reader
  reader = py_checkpoint_reader.NewCheckpointReader(ckpt_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  tensor_dict = {}

  for key in var_to_shape_map:
    if key in CKPT_VAR_NAME_SWAP:
      new_key = CKPT_VAR_NAME_SWAP[key]
      tensor_dict[new_key] = reader.get_tensor(key)

  for var in model.trainable_variables:
    if var.name in tensor_dict:
      var.assign(tensor_dict[var.name])
      print(f"Variable loaded: {var.name}")
    else:
      print("Variable not found:", var.name)


def save_model(model, save_path):
  print(f"Saving model to {save_path}")
  model.save(save_path)
  model.save_weights(f"{save_path}/cdfg_reader")
