import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn import layers


class Design2VecBase(Model):
  def __init__(self, cdfgs, n_hidden, n_labels=1, n_gcn_layers=4,
               n_mlp_hidden=256, dropout=0.1, cov_point_aggregate="mean",
               dtype=tf.float32):
    super().__init__()

    self.n_hidden = n_hidden
    self.n_labels = n_labels
    self.n_gcn_layers = n_gcn_layers
    self.n_mlp_hidden = n_mlp_hidden
    self.dropout = dropout
    self.dtype = dtype

    # Prepare left-hand side of the model (design reader side)
    self.cdfg_reader = layers.CdfgReader(
        cdfgs=cdfgs, n_hidden=n_hidden, n_gcn_layers=n_gcn_layers,
        dropout=dropout, activation="relu", final_activation="softmax",
        aggregate=cov_point_aggregate, dtype=self.dtype)
    # Prepare right-hand side of the model (test parameter side)
    self.tp_reader = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_labels=n_mlp_hidden, dropout=dropout,
        activation="relu", final_activation="softmax", dropout_at_end=True,
        dtype=self.dtype)
    # Prepare top that produces final output
    self.top = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_labels=n_labels, dropout=dropout,
        activation="relu", final_activation="sigmoid", dropout_at_end=False,
        dtype=self.dtype)

  def call(self, inputs):
    """Call the model."""
    # cov_point_embed: (batch_size, n_hidden)
    cov_point_embed = self.cdfg_reader(inputs)
    # tp_embed: (batch_size, n_mlp_hidden)
    tp_embed = self.tp_reader(inputs["test_parameters"])
    # is_hit: (batch_size, 1)
    is_hit = self.top(tf.concat((cov_point_embed, tp_embed), axis=1))
    return is_hit
