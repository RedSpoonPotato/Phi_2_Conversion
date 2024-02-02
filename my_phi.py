import bert
import tensorflow as tf

import math

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

