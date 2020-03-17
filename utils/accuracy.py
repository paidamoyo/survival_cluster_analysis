import tensorflow as tf


def accuracy_loss(e, predicted, batch_size, empirical):
    """Computes accuracy loss"""
    indices_lab = tf.where(tf.equal(tf.constant(1.0, dtype=tf.float32), e))
    predicted = tf.squeeze(predicted)

    batch_obs_loss = tf.losses.absolute_difference(labels=tf.gather(empirical, indices=indices_lab),
                                                   predictions=tf.gather(predicted, indices_lab))

    indices_cens = tf.where(tf.equal(tf.constant(0.0, dtype=tf.float32), e))
    diff_time_cens = tf.subtract(tf.gather(predicted, indices=indices_cens), tf.gather(empirical, indices=indices_cens))
    batch_cens_loss = tf.nn.relu(1.0 - diff_time_cens)

    observed = tf.reduce_sum(e)
    censored = tf.subtract(tf.cast(batch_size, dtype=tf.float32), observed)

    def normarlize_loss(cost, size):
        return tf.div(tf.reduce_sum(cost), tf.cast(size, dtype=tf.float32))

    total_recon_loss = tf.add(normarlize_loss(batch_cens_loss, size=censored),
                              normarlize_loss(batch_obs_loss, size=observed))
    return total_recon_loss

