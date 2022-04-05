import os
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv

sys.path.append(os.path.join(
    os.path.dirname(__file__), "../third_party/bigbird"))
from bigbird.core import flags as bigbird_flags
from bigbird.core.decoder import DecoderStack as BigBirdDecoderStack


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


def convert_batch_to_single_mode(cdfg_xs, cdfg_as):
  # Check rank of given input
  assert len(cdfg_xs.shape) == 3 and len(cdfg_as.shape) == 3

  # Create mask 1 where node is non-zero (i.e. not a padding), 0 otherwise
  mask = tf.reduce_sum(tf.abs(cdfg_xs), axis=-1)
  mask = tf.cast(mask > 0, dtype=cdfg_xs.dtype)

  # Convert Xs (i.e. graph node arrays) to single mode
  mask_1d = tf.reshape(mask, [-1])
  cdfg_xs = tf.reshape(cdfg_xs, [-1, cdfg_xs.shape[-1]])
  cdfg_xs = tf.boolean_mask(cdfg_xs, tf.cast(mask_1d, dtype=tf.bool))

  # Convert As (i.e. adjacent matrices) to single mode
  cdfg_lens = [int(l) for l in tf.reduce_sum(mask, axis=-1).numpy()]
  cdfg_as_aggr = tf.zeros(
      (cdfg_xs.shape[0], cdfg_xs.shape[0]), dtype=cdfg_as.dtype)
  offsets = [0]
  for i, cdfg_len in enumerate(cdfg_lens):
    offset = offsets[-1]
    next_offset = offset + cdfg_len
    cdfg_a = tf.sparse.from_dense(cdfg_as[i])
    cdfg_a_indices = cdfg_a.indices + [offset, offset]
    cdfg_as_aggr = tf.tensor_scatter_nd_update(
        cdfg_as_aggr, cdfg_a_indices, cdfg_a.values)
    lhs = cdfg_as_aggr[offset:next_offset, offset:next_offset]
    rhs = cdfg_as[i][:cdfg_len, :cdfg_len]
    assert tf.math.reduce_all(
        lhs == rhs), f"Update failed on {i}th case: {lhs} != {rhs}"
    offsets.append(offset + cdfg_len)
  offsets = offsets[:-1]  # Remove last offset
  cdfg_as = tf.sparse.from_dense(cdfg_as_aggr)

  return cdfg_xs, cdfg_as, cdfg_lens, offsets


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, n_layers, n_hidden, n_out, dropout, dropout_at_end=False,
               activation="relu", final_activation="tanh",
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


class AttentionModule(tf.keras.layers.Layer):
  def __init__(self, n_hidden, max_n_nodes=4096, dropout=0.1,
               dtype=tf.float32):
    super().__init__()

    var_init = tf.keras.initializers.glorot_uniform()
    self.cls_tok = tf.Variable(
        var_init(shape=[n_hidden]), trainable=True, dtype=dtype)
    self.sep_tok = tf.Variable(
        var_init(shape=[n_hidden]), trainable=True, dtype=dtype)
    self.pos_embed = tf.Variable(
        var_init(shape=[max_n_nodes, n_hidden]), trainable=True, dtype=dtype)

    params = bigbird_flags.FLAGS.flag_values_dict()  # Default params
    params["use_gradient_checkpointing"] = False
    params["attention_probs_dropout_prob"] = dropout
    params["hidden_dropout_prob"] = dropout
    params["num_attention_heads"] = 4
    params["num_hidden_layers"] = 6
    params["hidden_size"] = n_hidden
    params["intermediate_size"] = n_hidden * 4
    params["max_encoder_length"] = max_n_nodes
    # To name bigbird module to encoder
    params["couple_encoder_decoder"] = False
    self.att_block = BigBirdDecoderStack(params)
    self.self_attention_mask = tf.ones(shape=(1, 1, 1, 1), dtype=dtype)
    self.encoder_mask = tf.ones(shape=(1, max_n_nodes), dtype=dtype)

  def call(self, inputs):
    x = inputs["x"]
    y = inputs["y"]
    x += tf.expand_dims(self.pos_embed[:x.shape[1], :], 0)
    encoder_mask = self.encoder_mask[:, :x.shape[1]]
    x = self.att_block(encoder_outputs=x, decoder_inputs=y,
                       self_attention_mask=self.self_attention_mask,
                       encoder_mask=encoder_mask,
                       training=bool(tf.keras.backend.learning_phase()))
    return x


class CdfgReader(tf.keras.layers.Layer):
  def __init__(self, cdfgs, n_hidden, n_gnn_layers, dropout, activation="relu",
               final_activation="tanh", aggregate="mean", n_lstm_hidden=256,
               n_lstm_layers=2, use_attention=False, max_n_nodes=4096,
               dtype=tf.float32):
    super().__init__()
    assert use_attention or aggregate in ["mean", "lstm"]
    self.n_hidden = n_hidden
    self.n_gnn_layers = n_gnn_layers
    self.aggregate = aggregate
    self.use_attention = use_attention

    self.cdfg_xs, self.cdfg_as = preprocess_cdfgs(cdfgs)
    if self.use_attention:
      ret = convert_batch_to_single_mode(self.cdfg_xs, self.cdfg_as)
      self.cdfg_xs, self.cdfg_as, self.cdfg_lens, self.cdfg_offsets = ret

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

    if self.use_attention:
      self.sent_vec_bottleneck = FeedForward(
          1, n_hidden, n_out=n_hidden, dropout=dropout, dropout_at_end=True,
          final_activation=final_activation)
      self.att_module = AttentionModule(n_hidden, max_n_nodes, dropout, dtype)
      self.att_pooler = FeedForward(1, n_hidden, n_out=n_hidden,
                                    dropout=dropout, dropout_at_end=True,
                                    final_activation=final_activation)
    else:  # When not using attention, we use a single layer for aggregating
      if aggregate == "lstm":
        cells = [
            tf.keras.layers.LSTMCell(
                n_lstm_hidden, dropout=dropout, dtype=dtype)
            for _ in range(n_lstm_layers)]
        self.aggregate_layer = tf.keras.layers.RNN(
            tf.keras.layers.StackedRNNCells(cells))

  def place_special_tokens(self, x):
    # CLS is prepended, and SEP is added in between modules.
    sep_vec = tf.expand_dims(self.att_module.sep_tok, 0)
    cls_vec = tf.expand_dims(self.att_module.cls_tok, 0)

    x_per_graph = [cls_vec]  # Add CLS token at position 0
    for i, offset in enumerate(self.cdfg_offsets):
      start = offset
      end = offset + self.cdfg_lens[i]
      x_per_graph.append(x[start:end])
      x_per_graph.append(sep_vec)  # Add separator token in between
    x_per_graph = x_per_graph[:-1]  # Remove last separator token

    # x: (num_nodes + num_special tokens, n_hidden)
    x = tf.concat(x_per_graph, axis=0)
    return x

  def gnn_stack_forward(self, cdfg_xs, cdfg_as):
    # GNN: input layer -> n - 1 GNN layers (relu) -> final GNN layer (tanh)
    # x: (batch_size, num_nodes, n_hidden)
    x = self.gnn_input_layer(cdfg_xs)
    x = self.gnn_input_dropout(x)
    to_add = x  # Save for residual connection
    for i in range(self.n_gnn_layers):
      x = self.gnn_layers[i]((x, cdfg_as))
      x = self.gnn_dropouts[i](x)
    x += to_add  # Residual connection, (batch_size, num_nodes, n_hidden)
    return x

  def aggregate_forward(self, inputs):
    # Prepare inputs
    # cdfg_xs (batch_size, num_nodes, num_features)
    # cdfg_as: (batch_size, num_nodes, num_nodes)
    cdfg_xs = tf.gather_nd(self.cdfg_xs, inputs["graph"])
    cdfg_as = tf.gather_nd(self.cdfg_as, inputs["graph"])

    # x: (batch_size, num_nodes, n_hidden)
    x = self.gnn_stack_forward(cdfg_xs, cdfg_as)

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

  def attention_forward(self, inputs):
    # NOTE: whole graph is fed to the GNN regardless of the input.
    # cdfg_xs: (num_nodes, num_features)
    # cdfg_as: (num_nodes, num_nodes)
    # x: (num_nodes, n_hidden) <= all node embeddings
    x = self.gnn_stack_forward(self.cdfg_xs, self.cdfg_as)
    # CLS is prepended, and SEP is added in between modules.
    # x: (num_nodes + num_special_toks, n_hidden)
    x = self.place_special_tokens(x)

    # y: (batch_size, sent_vec_hidden) <= CP information is fed
    y = inputs["cp_sent_vecs"]
    y = tf.expand_dims(y, axis=1)  # (batch_size, 1, sent_vec_hidden)
    y = self.sent_vec_bottleneck(y)  # (batch_size, 1, n_hidden)

    # Add dimension to x to match size with y
    # x: (batch_size, num_nodes + num_special_toks, n_hidden)
    x = tf.expand_dims(x, axis=0)
    x = tf.tile(x, multiples=[tf.shape(y)[0], 1, 1])

    cp_embed = self.att_module({"x": x, "y": y})  # (batch_size, 1, n_hidden)
    cp_embed = cp_embed[:, 0, :]  # (batch_size, n_hidden)
    cp_embed = self.att_pooler(cp_embed)  # (batch_size, n_hidden)
    return cp_embed

  def call(self, inputs):
    if self.use_attention:
      return self.attention_forward(inputs)
    else:
      return self.aggregate_forward(inputs)
