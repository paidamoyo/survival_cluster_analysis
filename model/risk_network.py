import tensorflow as tf

from utils.distributions import uniform
from utils.tf_helpers import create_nn_weights, mlp_neuron, dropout_normalised_mlp


def pt_given_z(z, hidden_dim, batch_size, is_training, batch_norm, keep_prob,
               scope='generate_t_given_z',
               reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # Variables
        input_shape = z.get_shape().as_list()[1]

        w_hi, b_hi = create_nn_weights('h_z_3', 'decoder', [input_shape, hidden_dim[-1]])
        hidden_z = dropout_normalised_mlp(layer_input=z, weights=w_hi, biases=b_hi,
                                          is_training=is_training,
                                          batch_norm=batch_norm, keep_prob=keep_prob,
                                          layer='h_z_decoder')

        noise = uniform(dim=hidden_dim[-1], batch_size=batch_size)
        hidden_z_plus_noise = tf.concat([hidden_z, noise], axis=1)
        input_shape = hidden_z_plus_noise.get_shape().as_list()[1]

        w_t, b_t = create_nn_weights('t', 'encoder', [input_shape, 1])
        t_mu = mlp_neuron(hidden_z_plus_noise, w_t, b_t, activation=False)
        return tf.exp(t_mu), z


def pz_given_x(x, hidden_dim, is_training, batch_norm, keep_prob=0.9,
               scope='generate_z_given_x', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        input_shape = x.get_shape().as_list()[1]

        w_hi, b_hi = create_nn_weights('h_z_1', 'decoder', [input_shape, hidden_dim[0]])
        hidden_x = dropout_normalised_mlp(layer_input=x, weights=w_hi, biases=b_hi,
                                          is_training=is_training,
                                          batch_norm=batch_norm, keep_prob=keep_prob,
                                          layer='h_x_decoder')

        input_shape = hidden_x.get_shape().as_list()[1]

        w_hi, b_hi = create_nn_weights('h_z_2', 'decoder', [input_shape, hidden_dim[1]])

        hidden_z = dropout_normalised_mlp(layer_input=hidden_x, weights=w_hi, biases=b_hi,
                                          is_training=is_training,
                                          batch_norm=batch_norm, keep_prob=keep_prob,
                                          layer='h_z_decoder')

        return hidden_z
