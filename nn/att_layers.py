import os
import sys
import h5py
import tensorflow as tf
import numpy as np

from transformers import AutoConfig
from transformers import TFAutoModel

sys.path.append(os.path.join(os.path.dirname(__file__), "../third_party/bigbird"))

from nn.transformer import CrossModalModule


def tile_token_for_batch(tok, batch_size):
    assert (
        len(tok.shape) == 2
    ), f"Token shape must be [1, n_hidden], instead got {tok.shape}"
    tok = tf.expand_dims(tok, axis=0)
    tok = tf.tile(tok, [batch_size, 1, 1])
    return tok


def init_longformer(params):
    pretrain_dir = params["pretrain_dir"]
    model_id = params["huggingface_model_id"]
    attention_window = params["attention_window"]
    max_pos = params["max_n_nodes"] + 2
    num_attention_heads = params["num_attention_heads"]
    n_att_hidden = params["n_att_hidden"]
    n_att_layers = params["n_att_layers"]

    # Initialize and save model
    config = AutoConfig.from_pretrained(model_id)
    config.gradient_checkpointing = True
    config.attention_window = attention_window
    config.max_position_embeddings = max_pos
    config.hidden_size = n_att_hidden
    config.intermediate_size = n_att_hidden * 4
    config.num_attention_heads = num_attention_heads
    config.num_hidden_layers = n_att_layers
    init_model = TFAutoModel.from_config(config)
    # Run fake input to build the init_model
    init_model(tf.keras.Input(shape=[None], dtype=tf.int32))
    embed_name = init_model.longformer.embeddings.position_embeddings.name
    init_model.save_pretrained(pretrain_dir)
    tf.keras.backend.clear_session()

    # Load parent to get the embeddings
    parent = TFAutoModel.from_pretrained(model_id)
    # Run fake input to build the parent model
    parent(tf.keras.Input(shape=[None], dtype=tf.int32))
    parent_pos_emb = parent.longformer.embeddings.position_embeddings
    parent_pos_emb = parent_pos_emb[:max_pos].numpy()
    tf.keras.backend.clear_session()

    pool_step_size = parent_pos_emb.shape[-1] / n_att_hidden
    assert pool_step_size - int(pool_step_size) == 0, (
        f"Attention model's hidden size should be divisible by the parent's "
        f"hidden size. Got {n_att_hidden} and {parent_pos_emb.shape[-1]}."
    )
    pool_step_size = int(pool_step_size)
    new_embed = np.zeros((max_pos, n_att_hidden))
    for i in range(max_pos):
        for j in range(n_att_hidden):
            new_embed[i][j] = parent_pos_emb[i][j * pool_step_size]

    # Replace positional embedding from pretrained
    with h5py.File(os.path.join(pretrain_dir, "tf_model.h5"), "r+") as f:
        key = f"longformer/{embed_name}"
        f[key][:] = new_embed


def get_transformer(
    pretrain_dir,
    model_id="allenai/longformer-base-4096",
    n_hidden=768,
    n_layers=12,
    num_attention_heads=12,
    attention_window=256,
    max_pos=4098,
    from_scratch=False,
    params=None,
):

    if from_scratch:
        init_longformer(params)  # Saves initial transformer to pretrain_dir
    else:
        ignored_values = {
            "max_position_embeddings": max_pos,
            "hidden_size": n_hidden,
            "intermediate_size": n_hidden * 4,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": n_layers,
        }
        print(f"These values are ignored {ignored_values}")
        print(
            "Please use make sure that you are reading from correct pretrained "
            "model."
        )

    if os.path.isdir(pretrain_dir):
        # If pretrain_dir exists, load the model from there.
        print(f"Loading Longformer from {pretrain_dir}...")
        config = AutoConfig.from_pretrained(pretrain_dir)
        config.gradient_checkpointing = True
        config.attention_window = attention_window
        model = TFAutoModel.from_pretrained(pretrain_dir, config=config)
    else:
        # If no model exists, download the model
        print(f"Downloading Longformer: {model_id}")
        config = AutoConfig.from_pretrained(model_id)
        config.gradient_checkpointing = True
        config.attention_window = attention_window
        model = TFAutoModel.from_pretrained(model_id, config=config)

    print(f"Config used: {config}")
    model.save_pretrained(pretrain_dir)

    return model


class AttentionModule(tf.keras.layers.Layer):
    def __init__(
        self,
        n_hidden,
        n_layers,
        max_n_nodes=4096,
        pretrain_dir="pretrain/longformer-base-4096",
        params=None,
    ):
        super().__init__()
        assert params["use_att_encoder"] or params["use_att_decoder"]
        self.max_n_nodes = max_n_nodes
        self.mask_dtype = tf.int32
        self.att_encoder = None
        self.att_decoder = None

        model_id = params["huggingface_model_id"]
        num_attention_heads = params["num_attention_heads"]

        if params["use_att_encoder"]:
            self.att_encoder = get_transformer(
                pretrain_dir,
                model_id=model_id,
                max_pos=max_n_nodes + 2,
                n_hidden=n_hidden,
                n_layers=n_layers,
                attention_window=params["attention_window"],
                from_scratch=params["use_custom_attention_hparams"],
                num_attention_heads=num_attention_heads,
                params=params,
            )
            # To ignore final softmax layer
            self.att_encoder.set_output_embeddings(lambda x: x)
            if params["freeze_att_encoder"]:
                for layer in self.att_encoder.layers:
                    layer.trainable = False
                    print(f"Freezing layer {layer.name}")

        if params["use_att_decoder"]:
            self.att_decoder = CrossModalModule(
                num_layers=n_layers,
                d_model=n_hidden,
                rate=params["dropout"],
                num_heads=params["num_attention_heads"],
                dff=n_hidden * 4,
            )
            self.cp_embedding = tf.keras.layers.Embedding(
                params["n_max_coverpoints"], n_hidden
            )

        self.n_hidden = n_hidden
        self.one = tf.ones(shape=(1))
        self.zero = tf.zeros(shape=(1))
        var_init = tf.keras.initializers.glorot_uniform()
        self.cls_node = tf.Variable(
            var_init(shape=(1, self.n_hidden)), trainable=True, name="cls_node"
        )
        self.sep_node = tf.Variable(
            var_init(shape=(1, self.n_hidden)), trainable=True, name="sep_node"
        )
        self.mask_node = tf.Variable(
            var_init(shape=(1, self.n_hidden)), trainable=True, name="mask_node"
        )

    def get_input_information(self, inputs):
        mask_dtype = self.mask_dtype
        batch_size = tf.shape(inputs[0])[0]
        sep_node = tile_token_for_batch(self.sep_node, batch_size)
        cls_node = tile_token_for_batch(self.cls_node, batch_size)
        sep_batch = tf.cast(sep_node, dtype=inputs[0].dtype)
        cls_batch = tf.cast(cls_node, dtype=inputs[0].dtype)
        ones_batch = tf.ones(shape=(batch_size, 1), dtype=mask_dtype)
        return batch_size, sep_batch, cls_batch, ones_batch

    def prepare_att_input(self, inputs):
        mask_dtype = self.mask_dtype
        (batch_size, sep_batch, cls_batch, ones_batch) = self.get_input_information(
            inputs
        )
        _x_list = []
        global_att_mask = []
        for i, x in enumerate(inputs):
            _x_list.append(x)
            _x_list.append(sep_batch)
            global_att_mask.append(tf.zeros(shape=tf.shape(x)[:2], dtype=mask_dtype))
            global_att_mask.append(ones_batch)

        x = tf.concat([cls_batch] + _x_list, axis=1)
        global_att_mask = tf.concat([ones_batch] + global_att_mask, axis=1)
        return x, global_att_mask, batch_size, mask_dtype

    def call(self, inputs):
        x, y, cp_idx = inputs["x"], inputs["y"], inputs["cp_idx"]
        use_encoder = self.att_encoder is not None
        use_decoder = self.att_decoder is not None
        if use_decoder:
            x.append(y)  # If decoder isn't used, add y to input.

        x, global_att_mask, batch_size, mask_dtype = self.prepare_att_input(x)

        if not use_decoder:
            query_len = tf.shape(x[-1])[1] + 1
        else:
            query_len = 0
        token_type_ids = tf.concat(
            [
                tf.zeros(
                    shape=(batch_size, tf.shape(x)[1] - query_len), dtype=mask_dtype
                ),
                tf.ones(shape=(batch_size, query_len), dtype=mask_dtype),
            ],
            axis=1,
        )

        if use_encoder and use_decoder:
            # In this case as all rows in x are the same (no query in x, only cdfg
            # nodes) we can just use the first row to save memory.
            x = x[:1, ...]
            global_att_mask = global_att_mask[:1, ...]
            token_type_ids = token_type_ids[:1, ...]

        if use_encoder:
            x = self.att_encoder(
                input_ids=None,
                inputs_embeds=x,
                token_type_ids=token_type_ids,
                global_attention_mask=global_att_mask,
            )
            if use_decoder:
                # Encoder output is returned if no decoder is used.
                return tf.expand_dims(x.pooler_output, axis=1)
            else:
                x = x.last_hidden_state  # This will be used as input to decoder

        if use_encoder and use_decoder:
            # Revert batch size dimension
            x = tf.tile(x, multiples=[batch_size, 1, 1])

        if self.att_decoder is not None:
            encoder_output = x  # [batch_size, num_nodes, n_hidden]
            query = y + self.cp_embedding(cp_idx)  # [batch_size, 1, n_hidden]
            # x: [batch_size, 1, n_hidden]
            x = self.att_decoder(x=query, enc_output=encoder_output)

        return x

    def pretrain_forward(self, inputs, mask_ratio=0.15):
        (x, global_att_mask, batch_size, mask_dtype) = self.prepare_att_input(inputs)
        embed = x[0]  # Later transposed and matmul to output to get likelihoods

        # mask_mask: 1 where the node will be replaced with mask_node
        mask_mask = tf.cast(
            tf.random.uniform(global_att_mask.shape, minval=0, maxval=1) < mask_ratio,
            dtype=mask_dtype,
        )
        mask_mask *= 1 - global_att_mask  # zero out <cls> and <sep> location
        mask_location = tf.where(tf.cast(mask_mask, tf.float32))
        mask_batch = tf.tile(self.mask_node, [tf.shape(mask_location)[0], 1])
        # Update mask location with <mask>
        x = tf.tensor_scatter_nd_update(x, mask_location, mask_batch)
        token_type_ids = tf.zeros(shape=(batch_size, tf.shape(x)[1]), dtype=mask_dtype)
        x = self.att_encoder(
            input_ids=None,
            inputs_embeds=x,
            token_type_ids=token_type_ids,
            global_attention_mask=global_att_mask,
        )
        x = x.last_hidden_state

        # Transpose embedding to get likelihoods for each mask location
        x = tf.gather_nd(x, mask_location)
        y = tf.matmul(x, embed, transpose_b=True)
        y = tf.nn.softmax(y, axis=-1)
        # Get index labels
        labels = tf.range(start=0, limit=tf.shape(embed)[0], dtype=mask_dtype)
        labels = tf.expand_dims(labels, axis=0)
        labels = tf.tile(labels, [batch_size, 1])
        labels = tf.gather_nd(labels, mask_location)
        labels = tf.one_hot(labels, depth=tf.shape(embed)[0])

        return y, labels
