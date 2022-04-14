import math
import tensorflow as tf


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


def get_lr_schedule(
        lr, lr_scheme, decay_rate=0.90, decay_steps=500, warmup_steps=1000):
  print(f"lr_scheme: {lr_scheme} is ignored. "
        "For now, only warm up then exp decay is used.")
  lr_schedule = WarmUpThenDecay(lr, decay_rate, decay_steps, warmup_steps)
  return lr_schedule
