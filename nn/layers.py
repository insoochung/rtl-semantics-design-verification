import os
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def preprocess_cdfgs(cdfgs):
  """Preprocess CDFG batches for easy fetching during execution."""
  # CDFGs are constant untrained input to GNNs
  cdfg_xs = []
  cdfg_as = []
  for cdfg in cdfgs:
    cdfg_xs.append(cdfg.x)
    cdfg_as.append(cdfg.a)
  cdfg_xs = tf.stack(cdfg_xs)
  cdfg_as = tf.stack(cdfg_as)
  return cdfg_xs, cdfg_as


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, n_layers, n_hidden, n_out, dropout, dropout_at_end=False,
               activation="relu", final_activation="softmax",
               dtype=tf.float32):
    super(FeedForward, self).__init__()
    self.n_layers = n_layers
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.dropout_at_end = dropout_at_end

    self.dense_layers = []
    self.dropout_layers = []
    for _ in range(n_layers - 1):
      self.dense_layers.append(
          Dense(n_hidden, activation=activation, dtype=dtype))
      self.dropout_layers.append(Dropout(dropout))
    self.dense_layers.append(
        Dense(n_out, activation=final_activation, dtype=dtype))
    if dropout_at_end:
      self.dropout_layers.append(Dropout(dropout))

  def call(self, input):
    x = input
    for i in range(self.n_layers - 1):
      x = self.dense_layers[i](x)
      x = self.dropout_layers[i](x)
    x = self.dense_layers[-1](x)
    if self.dropout_at_end:
      x = self.dropout_layers[-1](x)
    return x


class CdfgReader(tf.keras.layers.Layer):
  def __init__(self, cdfgs, n_hidden, n_gnn_layers, dropout, activation="relu",
               final_activation="softmax", aggregate="mean", dtype=tf.float32):
    assert aggregate in ["mean", "lstm"]

    super(CdfgReader, self).__init__()
    self.n_gnn_layers = n_gnn_layers
    self.aggregate = aggregate

    self.cdfg_xs, self.cdfg_as = preprocess_cdfgs(cdfgs)

    self.gnn_input_layer = Dense(n_hidden, activation=activation,
                                 dtype=dtype)
    self.gnn_layers = []
    self.gnn_dropouts = []
    self.gnn_input_dropout = Dropout(dropout)
    for _ in range(n_gnn_layers - 1):
      self.gnn_layers.append(
          GCNConv(n_hidden, activation=activation, dtype=dtype))
      self.gnn_dropouts.append(Dropout(dropout))
    self.gnn_layers.append(
        GCNConv(n_hidden, activation=final_activation, dtype=dtype))
    self.gnn_dropouts.append(Dropout(dropout))
    if aggregate == "lstm":
      self.aggregate_layer = tf.keras.layers.LSTM(
          n_hidden, dropout=dropout, dtype=dtype)

  def call(self, inputs):
    # cdfg_xs (batch_size, num_nodes, num_features)
    # cdfg_as: (batch_size, num_nodes, num_nodes)
    if len(inputs["graph"].shape) == 1:
      inputs["graph"] = tf.expand_dims(inputs["graph"], axis=-1)
    if len(inputs["coverpoint"].shape) == 1:
      inputs["coverpoint"] = tf.expand_dims(inputs["coverpoint"], axis=-1)
    cdfg_xs = tf.gather_nd(self.cdfg_xs, inputs["graph"])
    cdfg_as = tf.gather_nd(self.cdfg_as, inputs["graph"])

    # GNN
    # input layer -> n - 1 GNN layers (relu) -> final GNN layer (softmax)
    # (batch_size, num_nodes, n_hidden)
    x = self.gnn_input_layer(cdfg_xs)
    x = self.gnn_input_dropout(x)
    to_add = x  # Save for residual connection
    for i in range(self.n_gnn_layers):
      x = self.gnn_layers[i]((x, cdfg_as))
      x = self.gnn_dropouts[i](x)
    x += to_add  # Residual connection, (batch_size, num_nodes, n_hidden)

    # Aggregate final output
    cp_masks = inputs["coverpoint_mask"]  # (batch_size, num_nodes)
    # (batch_size, num_nodes_in_cp, n_hidden)
    x = tf.ragged.boolean_mask(x, cp_masks)
    if self.aggregate == "mean":
      x = tf.reduce_mean(x, axis=1)  # (batch_size, n_hidden)
    elif self.aggregate == "lstm":
      x = self.aggregate_layer(x)  # (batch_size, n_hidden)
    else:
      assert False, f"Unsupported aggregate type {self.aggregate}."

    return x
