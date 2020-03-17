import numpy as np
import tensorflow as tf


def uniform(dim, batch_size):
    ones = np.ones(shape=dim, dtype=np.float32)
    noise = tf.distributions.Uniform(low=0 * ones, high=ones).sample(sample_shape=[batch_size])
    return noise
