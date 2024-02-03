import bert
import tensorflow as tf

import math


# note: Instead of using register buffers, I am just using normal variables, but 
# this might cause issues when exporting models

# stuff to work on:
""" stuff to work on:
- testing everything
- check with actiavtion function is actually used (check in colab)
- fix configs
- replace all linear layers bert.dense_v2
- THE config in PhiForTokenClassication --> if we change flash-atten to atten,
will the output of the running model be good
    -that does not neccesaeryily mean we should not eventually flash-attn
    
"""

def Flatten(x):
    elements = 1
    for d in x.shape:
        elements *= d
    return tf.reshape(x, (elements))

def MY_get_unpad_data(attention_mask):
    assert(attention_mask.dtype == tf.int32)
    seqlens_in_batch = tf.cast(tf.math.reduce_sum(attention_mask, axis=-1), dtype=tf.int32)
    zero = tf.constant(0, dtype=tf.int32)
    indices = Flatten(tf.where(tf.not_equal(Flatten(attention_mask), zero)))
    max_seqlen_in_batch = tf.math.reduce_max(seqlens_in_batch).numpy().item()
    cu_seqlens = tf.pad(tf.cast(tf.math.cumsum(seqlens_in_batch, axis=0), tf.int32), [[0,0],[1,1]])
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class PhiRotaryEmbedding(tf.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, name=None):
        super().__init__(name)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (tf.cast((tf.range(0, self.dim, 2)), dtype=tf.float32) / self.dim))
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.experimental.numpy.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)
    def __call__(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)
        return (
            tf.cast(self.cos_cached[:seq_len], dtype=x.dtype),
            tf.cast(self.cos_cached[:seq_len], dtype=x.dtype),
        )

class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = tf.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)

class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (base ** (tf.cast(tf.range(0, self.dim, 2), dtype=tf.float32) / self.dim))

        t = tf.range(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = tf.outer(t, self.inv_freq)
        emb = tf.concat((freqs, freqs), axis=-1)
        self.cos_cached = tf.cast(tf.cos(emb), dtype=dtype)
        self.sin_cached = tf.cast(tf.sin(emb), dtype=dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tf.concat((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """See orginal source for better description of what code does"""
    cos = tf.expand_dims(tf.gather(cos, position_ids), axis=unsqueeze_dim)
    sin = tf.expand_dims(tf.gather(sin, position_ids), axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PhiMLP_Config():
    def __init__(self):
        self.hidden_act
        self.hidden_size
        self.intermediate_size

class NewGELU(tf.Module):
    def __init__(self, name=None):
        super().__init__(name)
    def __call__(self, input):
        return 0.5 * input * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * tf.pow(input, 3.0))))

# need to come back to this
class PhiMLP(tf.Module):
    def __init__(self, config:PhiMLP_Config, params, name=None):
        super().__init__(name)
        self.config = config
        self.activation_fn = NewGELU() # check this matches the actual model
        self.fc1 = bert.Dense_v2(config.hidden_size, config.intermediate_size, params)
        self.fc2 = bert.Dense_v2(config.intermediate_size, config.hidden_size, params)
    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states) # also the function is stateless (consider changing)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

def repeat_kv(hidden_states, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states    
    hidden_states = tf.broadcast_to(hidden_states[:, :, None, :, :], (batch, num_key_value_heads, n_rep, slen, head_dim))
    return tf.reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))


class PhiConfig():
    def __init__(self) -> None:
        self.attention_dropout
        self.hidden_size
        self.num_attention_heads
        self.num_key_value_heads
        self.max_position_embeddings
        self.rope_theta
        self.partial_rotary_factor
        self.layer_norm_eps
        self.rope_scaling # appears to be a dictionary of sort


class PhiAttention(tf.Module):
    def __init__(self, config: PhiConfig, layer_idx:int = None, name=None):
        super().__init__(name)
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = bert.Dense_v2.(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj =bert.Dense_v2(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = bert.Dense_v2(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.dense = bert.Dense_v2(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = bert.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = bert.LayerNorm(
                config.hidden_size // self.num_heads, eps=config.layer_norm_eps, elementwise_affine=True
            )
        self._init_rope()
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states =  tf.reshape(query_states,[bsz, q_len, self.num_heads,           self.head_dim])
        key_states =    tf.reshape(key_states,  [bsz, q_len, self.num_key_value_heads, self.head_dim])
        value_states =  tf.reshape(value_states,[bsz, q_len, self.num_key_value_heads, self.head_dim])
        query_states =  tf.transpose(query_states,  perm=[1, 2])
        key_states =    tf.transpose(key_states,    perm=[1, 2])
        value_states =  tf.transpose(value_states,  perm=[1, 2])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            print("Error!") # for now
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = tf.concat((query_rot, query_pass), axis=-1)
        key_states = tf.concat((key_rot, key_pass), axis=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = tf.matmul(
            tf.cast(query_states, dtype=tf.float32), 
            tf.transpose(tf.cast(key_states, dtype=tf.float32), perm=(2,3))
        ) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = tf.cast(tf.nn.softmax(tf.cast(attn_weights, dtype=tf.float32), axis=-1), dtype=value_states.dtype)

        attn_output = tf.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = tf.transpose(attn_output, perm=(1,2))
        attn_output = tf.reshape(attn_output, (bsz, q_len, self.hidden_size))
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# for now, lets try without flash attention


