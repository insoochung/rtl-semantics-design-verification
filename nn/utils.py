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
    warmup_steps = self.warmup_steps
    decay_factor = math.pow(self.decay_rate, 1 / self.decay_steps)
    lr = self.initial_learning_rate
    return tf.cond(step <= warmup_steps,
      lambda: lr * (step / warmup_steps), # Before warmup steps is reached
      lambda: lr * tf.math.pow(tf.constant(decay_factor, dtype=tf.float32),
                               step - warmup_steps)) # After

def get_lr_schedule(
  lr, lr_scheme, decay_rate=0.90, decay_steps=500, warmup_steps=1000):
  print(f"lr_scheme: {lr_scheme} is ignored. "
    "For now, only warm up then exp decay is used.")
  lr_schedule = WarmUpThenDecay(lr, decay_rate, decay_steps, warmup_steps)
  return lr_schedule