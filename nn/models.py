import os
import sys
import copy

import tensorflow as tf
from tensorflow.keras.models import Model

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nn import layers


class Design2VecBase(Model):
  def __init__(self, params):
    super().__init__()
    cdfgs = params["graphs"]
    n_hidden = params["n_hidden"]
    n_labels = params["n_labels"]
    n_gnn_layers = params["n_gnn_layers"]
    n_mlp_hidden = params["n_mlp_hidden"]
    n_mlp_layers = params["n_mlp_layers"]
    dropout = params["dropout"]
    aggregate = params["aggregate"]
    use_attention = params["use_attention"]
    n_lstm_hidden = params["n_lstm_hidden"]
    n_lstm_layers = params["n_lstm_layers"]
    self.params = copy.deepcopy(params)

    # Prepare left-hand side of the model (design reader side)
    self.cdfg_reader = layers.CdfgReader(
        cdfgs=cdfgs, n_hidden=n_hidden, n_gnn_layers=n_gnn_layers,
        dropout=dropout, activation="relu", final_activation="tanh",
        aggregate=aggregate, use_attention=use_attention,
        n_lstm_hidden=n_lstm_hidden, n_lstm_layers=n_lstm_layers,
        params=params)
    # Prepare right-hand side of the model (test parameter side)
    self.tp_reader = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_out=n_mlp_hidden, n_layers=n_mlp_layers,
        activation="relu", final_activation="tanh", dropout_at_end=True,
        dropout=dropout, params=params)
    # Prepare top that produces final output
    self.top = layers.FeedForward(
        n_hidden=n_mlp_hidden, n_out=n_labels, n_layers=n_mlp_layers,
        activation="relu", final_activation="sigmoid", dropout_at_end=False,
        dropout=dropout, params=params)

  def call(self, inputs):
    """Call the model."""
    # cov_point_embed: (batch_size, n_hidden)
    cov_point_embed = self.cdfg_reader(inputs)
    # tp_embed: (batch_size, n_mlp_hidden)
    tp_embed = self.tp_reader(inputs["test_parameters"])
    # is_hit: (batch_size, 1)
    is_hit = self.top(tf.concat((cov_point_embed, tp_embed), axis=1))
    return is_hit
