import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn import layers


class Design2VecBase(Model):
  def __init__(self, cdfgs, n_hidden, n_mlp_hidden=256,
               n_lstm_hidden=None, n_labels=1,
               n_gnn_layers=4, n_mlp_layers=2, n_lstm_layers=2,
               dropout=0.1, cov_point_aggregate="mean", use_attention=True,
               dtype=tf.float32):
    super().__init__()

    self.n_hidden = n_hidden
    self.n_labels = n_labels
    self.n_gnn_layers = n_gnn_layers
    self.n_mlp_hidden = n_mlp_hidden
    self.dropout = dropout

    # Prepare left-hand side of the model (design reader side)
    self.cdfg_reader = layers.CdfgReader(
        cdfgs=cdfgs, n_hidden=n_hidden, n_gnn_layers=n_gnn_layers,
        dropout=dropout, activation="relu", final_activation="tanh",
        aggregate=cov_point_aggregate, use_attention=use_attention,
        n_lstm_hidden=n_lstm_hidden, n_lstm_layers=n_lstm_layers, dtype=dtype)
    # Prepare right-hand side of the model (test parameter side)
    self.tp_reader = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_out=n_mlp_hidden, n_layers=n_mlp_layers,
        activation="relu", final_activation="tanh", dropout_at_end=True,
        dropout=dropout, dtype=dtype)
    # Prepare top that produces final output
    self.top = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_out=n_labels, n_layers=n_mlp_layers,
        activation="relu", final_activation="sigmoid", dropout_at_end=False,
        dropout=dropout, dtype=dtype)

  def call(self, inputs):
    """Call the model."""
    # cov_point_embed: (batch_size, n_hidden)
    cov_point_embed = self.cdfg_reader(inputs)
    # tp_embed: (batch_size, n_mlp_hidden)
    tp_embed = self.tp_reader(inputs["test_parameters"])
    # is_hit: (batch_size, 1)
    is_hit = self.top(tf.concat((cov_point_embed, tp_embed), axis=1))
    return is_hit
