import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv


class Design2VecBase(Model):
  def __init__(self, n_hidden, n_labels=1, n_gcn_layers=4, n_mlp_hidden=256, dropout=0.1):
    # GCN activation="relu", output_activation="softmax"
    super().__init__()
    self.n_hidden = n_hidden
    self.n_labels = n_labels
    self.n_gcn_layers = n_gcn_layers
    self.n_mlp_hidden = n_mlp_hidden
    self.dropout = dropout

    self.gcn_input_layer = Dense(n_hidden, activation="relu")
    self.gcn_layers = []
    self.gcn_dropouts = []
    for i in range(n_gcn_layers - 1):
      self.gcn_layers.append(
          GCNConv(n_hidden, activation="relu"))
    self.gcn_layers.append(GCNConv(n_hidden, activation="softmax"))
    self.gcn_input_dropout = Dropout(dropout)
    for i in range(n_gcn_layers):
      self.gcn_dropouts.append(Dropout(dropout))

    self.tp_mlp_1 = Dense(n_mlp_hidden, activation="relu")
    self.tp_mlp_2 = Dense(n_mlp_hidden, activation="softmax")
    self.tp_input_dropout = Dropout(dropout)
    self.tp_mlp_1_dropout = Dropout(dropout)
    self.tp_mlp_2_dropout = Dropout(dropout)

    self.final_mlp_1 = Dense(n_mlp_hidden, activation="relu")
    self.final_mlp_2 = Dense(n_labels, activation="sigmoid")
    self.final_input_dropout = Dropout(dropout)
    self.final_mlp_1_dropout = Dropout(dropout)

  def call(self, inputs):
    tps = inputs["test_parameters"]  # (batch_size, n_mlp_hidden)
    graphs = inputs["graph"]  # (batch_size, num_nodes, num_features)
    cp_masks = inputs["coverpoint_mask"]  # (batch_size, num_nodes)

    # GCN
    # input layer -> n - 1 GCN layers (relu) -> final GCN layer (softmax)
    # (batch_size, num_nodes, n_hidden)
    x = self.gcn_input_layer(graphs)
    x = self.gcn_input_dropout(x)
    to_add = x  # Save for residual connection
    for i in range(self.n_gcn_layers):
      x = self.gcn_layers[i](x)
      x = self.gcn_dropouts[i](x)
    x += to_add  # Residual connection, (batch_size, num_nodes, n_hidden)
    # (batch_size, num_nodes_in_cp, n_hidden)
    x = tf.ragged.boolean_mask(x, cp_masks)
    # TODO: Add an option to support LSTM later on.
    cov_point_embed = tf.reduce_mean(x, axis=1)  # (batch_size, n_hidden)

    # MLP
    x = tps  # (batch_size, n_mlp_hidden)
    x = self.tp_input_dropout(x)
    x = self.tp_mlp_1(x)
    x = self.tp_mlp_1_dropout(x)
    x = self.tp_mlp_2(x)
    tp_embed = self.tp_mlp_2_dropout(x)  # (batch_size, n_mlp_hidden)

    # Concatenate, x: (batch_size, n_hidden + n_mlp_hidden)
    x = tf.concat(cov_point_embed, tp_embed, axis=1)
    x = self.final_input_dropout(x)
    x = self.final_mlp_1(x)  # x: (batch_size, n_mlp_hidden)
    x = self.final_mlp_1_dropout(x)
    x = self.final_mlp_2(x)  # x: (batch_size)
    return x
