import os
import sys
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv

sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/bigbird"))

from nn.att_layers import AttentionModule


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
    return (tf.cast(cdfg_xs, dtype=tf.float32), tf.cast(cdfg_as, dtype=tf.float32))


def add_cls_tok_to_cdfgs(cdfg_xs, cdfg_as, cls_tok):
    if len(cls_tok.shape) != 2:
        assert len(cls_tok.shape) == 1
        cls_tok = tf.expand_dims(cls_tok, axis=0)
    assert len(cdfg_xs.shape) == 2 and len(cdfg_as.shape) == 2, (
        f"CDFGs must be 2D or in single mode, provided shapes are, "
        f"XS: {cdfg_xs.shape}, AS: {cdfg_as.shape}"
    )
    _xs = tf.concat([cls_tok, cdfg_xs], axis=0)
    _as = tf.sparse.to_dense(cdfg_as)
    # Update adjancent matrix for CLS token.
    # Note: CLS token acts as an isolated node in the graph.
    _as = tf.concat([tf.zeros(shape=(1, _as.shape[1])), _as], axis=0)  # Add row
    _as = tf.concat([tf.zeros(shape=(_as.shape[0], 1)), _as], axis=1)  # Add col

    return _xs, _as


def add_padding(tensor, min_size, rank=2, axis=0):
    # Increment the dimension of an axis by 1 by padding with zeros
    assert len(tensor.shape) == rank
    pad_size = min_size - tf.shape(tensor)[axis]
    paddings = [[0, 0] for _ in range(rank)]
    paddings[axis][1] = pad_size
    return tf.pad(tensor, paddings)


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
    cdfg_as_aggr = tf.zeros((cdfg_xs.shape[0], cdfg_xs.shape[0]), dtype=cdfg_as.dtype)
    offsets = [0]
    for i, cdfg_len in enumerate(cdfg_lens):
        offset = offsets[-1]
        next_offset = offset + cdfg_len
        cdfg_a = tf.sparse.from_dense(cdfg_as[i])
        cdfg_a_indices = cdfg_a.indices + [offset, offset]
        cdfg_as_aggr = tf.tensor_scatter_nd_update(
            cdfg_as_aggr, cdfg_a_indices, cdfg_a.values
        )
        lhs = cdfg_as_aggr[offset:next_offset, offset:next_offset]
        rhs = cdfg_as[i][:cdfg_len, :cdfg_len]
        assert tf.math.reduce_all(
            lhs == rhs
        ), f"Update failed on {i}th case: {lhs} != {rhs}"
        offsets.append(offset + cdfg_len)
    offsets = offsets[:-1]  # Remove last offset
    cdfg_as = tf.sparse.from_dense(cdfg_as_aggr)

    return cdfg_xs, cdfg_as, cdfg_lens, offsets


class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self,
        n_layers,
        n_hidden,
        n_out,
        dropout,
        dropout_at_end=False,
        activation="relu",
        final_activation="tanh",
        params=None,
    ):
        super(FeedForward, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.dropout_at_end = dropout_at_end

        self.dense_layers = []
        self.dropout_layers = []
        for _ in range(n_layers - 1):
            self.dense_layers.append(Dense(n_hidden, activation=activation))
            self.dropout_layers.append(Dropout(dropout))
        self.dense_layers.append(Dense(n_out, activation=final_activation))
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
    def __init__(
        self,
        cdfgs,
        n_hidden,
        n_gnn_layers,
        dropout,
        activation="relu",
        final_activation="tanh",
        aggregate="mean",
        n_lstm_hidden=256,
        n_lstm_layers=2,
        use_attention=False,
        params=None,
    ):
        super().__init__()
        assert use_attention or aggregate in ["mean", "lstm"]
        self.n_hidden = n_hidden
        self.n_gnn_layers = n_gnn_layers
        self.aggregate = aggregate
        self.use_attention = use_attention
        self.cdfg_cutoff = params["cdfg_cutoff"]

        self.batch_xs, self.batch_as = preprocess_cdfgs(cdfgs)
        if self.use_attention:
            (
                self.single_xs,
                self.single_as,
                self.cdfg_lens,
                self.cdfg_offsets,
            ) = convert_batch_to_single_mode(self.batch_xs, self.batch_as)

        if self.cdfg_cutoff:
            print(f"NOTE: Cutoffing cdfgs in the CdfgReader...")
            max_n_nodes = params["max_n_nodes"]
            self.cutoff_embedding = tf.keras.layers.Embedding(max_n_nodes, n_hidden)
            (_, _, self.cdfg_lens, self.cdfg_offsets) = convert_batch_to_single_mode(
                self.batch_xs, self.batch_as
            )
            self.cdfg_lens_const = tf.constant(self.cdfg_lens, dtype=tf.int32)
            self.cdfg_offsets_const = tf.constant(self.cdfg_offsets, dtype=tf.int32)
        else:
            self.gnn_input_layer = Dense(n_hidden, activation=activation)
            self.gnn_layers = []
            self.gnn_dropouts = []
            self.gnn_input_dropout = Dropout(dropout)
            for _ in range(n_gnn_layers - 1):
                self.gnn_layers.append(GCNConv(n_hidden, activation=activation))
                self.gnn_dropouts.append(Dropout(dropout))
            self.gnn_layers.append(GCNConv(n_hidden, activation=final_activation))
            self.gnn_dropouts.append(Dropout(dropout))

        if self.use_attention:
            n_att_hidden = params["n_att_hidden"]
            n_att_layers = params["n_att_layers"]
            max_n_nodes = params["max_n_nodes"]
            pretrain_dir = params["pretrain_dir"]
            self.sent_vec_bottleneck = FeedForward(
                1,
                n_att_hidden,
                n_out=n_att_hidden,
                dropout=dropout,
                dropout_at_end=True,
                final_activation=final_activation,
                params=params,
            )
            self.node_embed_bottleneck = FeedForward(
                1,
                n_hidden,
                n_out=n_att_hidden,
                dropout=dropout,
                dropout_at_end=True,
                final_activation=final_activation,
                params=params,
            )
            self.att_module = AttentionModule(
                n_att_hidden, n_att_layers, max_n_nodes, pretrain_dir, params
            )
            self.att_pooler = FeedForward(
                1,
                n_att_hidden,
                n_out=n_hidden,
                dropout=dropout,
                dropout_at_end=True,
                final_activation=final_activation,
                params=params,
            )
        else:  # When not using attention, we use a single layer for aggregating
            if aggregate == "lstm":
                cells = [
                    tf.keras.layers.LSTMCell(n_lstm_hidden, dropout=dropout)
                    for _ in range(n_lstm_layers)
                ]
                self.aggregate_layer = tf.keras.layers.RNN(
                    tf.keras.layers.StackedRNNCells(cells)
                )

    def gnn_stack_forward(self, cdfg_xs, cdfg_as):
        assert not self.cdfg_cutoff
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

    def get_cutoff_embeddings(self, inputs):
        offsets = tf.gather(self.cdfg_offsets_const, inputs["graph"])  # [batch_size, 1]
        indices = tf.range(tf.shape(self.batch_xs)[1])  # [max_graph_len]
        indices = tf.tile(  # [batch_size, max_graph_len]
            tf.expand_dims(indices, 0), [tf.shape(inputs["graph"])[0], 1]
        )
        indices += offsets  # [batch_size, max_graph_len]
        graph_len = tf.gather(self.cdfg_lens_const, inputs["graph"])
        # [batch_size, max_graph_len, n_hidden]
        res = self.cutoff_embedding(indices)
        mask = tf.sequence_mask(
            tf.reshape(graph_len, shape=[-1]),
            maxlen=tf.shape(self.batch_xs)[1],
            dtype=tf.float32,
        )
        mask = tf.expand_dims(mask, axis=-1)  # [batch_size, max_graph_len, 1]
        res *= mask  # [batch_size, max_graph_len, n_hidden]
        return res

    def aggregate_forward(self, inputs):
        # Prepare inputs
        # cdfg_xs (batch_size, num_nodes, num_features)
        # cdfg_as: (batch_size, num_nodes, num_nodes)
        if not self.cdfg_cutoff:
            graph = inputs["graph"]
            if len(graph.shape) == 1:
                graph = tf.expand_dims(graph, axis=-1)
            cdfg_xs = tf.gather_nd(self.batch_xs, inputs["graph"])
            cdfg_as = tf.gather_nd(self.batch_as, inputs["graph"])

            # x: (batch_size, num_nodes, n_hidden)
            x = self.gnn_stack_forward(cdfg_xs, cdfg_as)
        else:  # Cut off cdfg information
            x = self.get_cutoff_embeddings(inputs)

        # Add batch dimension if x is not a batch but a single example
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=0)

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
        cdfg_xs, cdfg_as = self.single_xs, self.single_as
        x = self.gnn_stack_forward(cdfg_xs, cdfg_as)
        x = self.node_embed_bottleneck(x)  # (batch_size, num_nodes, n_att_hidden)

        # y: (batch_size, sent_vec_hidden) <= CP information is fed
        y = inputs["cp_sent_vecs"]
        y = tf.expand_dims(y, axis=1)  # (batch_size, 1, sent_vec_hidden)
        y = self.sent_vec_bottleneck(y)  # (batch_size, 1, n_att_module)

        # Add dimension to x to match size with y
        x = tf.expand_dims(x, axis=0)  # x: (1, num_nodes, n_att_module)
        # x: (batch_size, num_nodes, n_att_module)
        x = tf.tile(x, multiples=[tf.shape(y)[0], 1, 1])
        # Split output per graph lengths
        # x_list = [(batch_size, ?, n_att_module), ...]
        x_list = tf.split(x, self.cdfg_lens, axis=1)
        # cp_embed: (batch_size, num_nodes, n_att_module)
        cp_embed = self.att_module(
            {"x": x_list, "y": y, "cp_idx": inputs["coverpoint"]}
        )
        cp_embed = cp_embed[:, 0, :]  # (batch_size, n_att_module)
        # cp_embed = tf.concat([cp_embed, y], axis=-1)  # (batch_size, n_hidden)
        cp_embed = self.att_pooler(cp_embed)  # (batch_size, n_hidden)

        return cp_embed

    def pretrain_forward(self, batch_size=16, cdfgs=None, percent_graphs=0.85):
        if cdfgs is not None:
            cdfg_xs, cdfg_as, cdfg_lens = cdfgs
        else:
            cdfg_xs, cdfg_as = self.single_xs, self.single_as
            cdfg_lens = self.cdfg_lens
        x = self.gnn_stack_forward(cdfg_xs, cdfg_as)
        x = tf.expand_dims(x, axis=0)  # x: (1, num_nodes, n_att_module)
        x_list = tf.split(x, cdfg_lens, axis=1)

        # Pick a random subset of graphs
        indices = list(range(len(x_list)))
        random.shuffle(indices)
        indices = indices[: int(len(x_list) * percent_graphs)]
        x_list = [x_list[i] for i in indices]
        cdfg_lens = [cdfg_lens[i] for i in indices]
        x = tf.concat(x_list, axis=1)  # (1, num_nodes, n_att_module)
        x = self.node_embed_bottleneck(x)  # (1, num_nodes, n_att_hidden)
        # x: (batch_size, num_nodes, n_att_module)
        x = tf.tile(x, multiples=[batch_size, 1, 1])
        # Split output per graph lengths
        # x_list = [(batch_size, ?, n_att_module), ...]
        x_list = tf.split(x, cdfg_lens, axis=1)
        y, y_labels = self.att_module.pretrain_forward(x_list)
        return y, y_labels

    def call(self, inputs):
        # Make this work with generated dataset with legacy datagen code.
        if self.use_attention:
            return self.attention_forward(inputs)
        else:
            return self.aggregate_forward(inputs)
