import tensorflow as tf


def accuracy_loss(e, predicted, batch_size, empirical):
    """Computes accuracy loss"""
    total_cens_loss = tf.constant(0.0, shape=())
    total_obs_loss = tf.constant(0.0, shape=())
    predicted = tf.squeeze(predicted)
    observed = tf.reduce_sum(e)
    censored = tf.subtract(tf.cast(batch_size, dtype=tf.float32), observed)

    def condition(i, recon_loss, obs_recon_loss):
        return i < batch_size

    def body(i, cens_recon_loss, obs_recon_loss):
        # get edges for observation i
        pred_t_i = tf.gather(predicted, i)
        emp_t_i = tf.gather(empirical, i)
        e_i = tf.gather(e, i)
        censored = tf.equal(e_i, 0)
        # calculate partial likelihood

        # Censored generated t loss
        diff_time = tf.subtract(pred_t_i, emp_t_i)
        hinge = tf.nn.relu(1.0 - diff_time)
        censored_loss_i = tf.cond(censored, lambda: hinge, lambda: tf.constant(0.0))
        # L1 recon
        observed_loss_i = tf.cond(censored, lambda: tf.constant(0.0),
                                  lambda: tf.losses.absolute_difference(labels=emp_t_i, predictions=pred_t_i))
        # add observation risk to total risk
        cum_cens_loss = tf.add(cens_recon_loss, censored_loss_i)
        cum_obs_loss = tf.add(obs_recon_loss, observed_loss_i)
        return [i + 1, tf.reshape(cum_cens_loss, shape=()), tf.reshape(cum_obs_loss, shape=())]

    # Relevant Functions
    idx = tf.constant(0, shape=())
    _, batch_cens_loss, batch_obs_loss = tf.while_loop(condition, body,
                                                       loop_vars=[idx,
                                                                  total_cens_loss,
                                                                  total_obs_loss],
                                                       shape_invariants=[
                                                           idx.get_shape(),
                                                           total_cens_loss.get_shape(),
                                                           total_obs_loss.get_shape()])

    def normarlize_loss(cost, size):
        return tf.div(cost, tf.cast(size, dtype=tf.float32))

    total_recon_loss = tf.add(normarlize_loss(batch_cens_loss, size=censored),
                              normarlize_loss(batch_obs_loss, size=observed))
    return total_recon_loss
