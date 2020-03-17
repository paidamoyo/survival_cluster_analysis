import tensorflow as tf


def km_estimator(predicted, empirical, t_range, batch_size, e, t_range_size):
    """Computes the calibration loss from SFM"""

    def cond(idx, pred_surv, emp_surv, emd_los):
        return tf.less(idx, t_range_size)

    t_range = tf.cast(t_range, dtype=tf.float32)
    predicted = tf.cast(predicted, dtype=tf.float32)
    empirical = tf.cast(empirical, dtype=tf.float32)
    batch_size = tf.cast(batch_size, dtype=tf.float32)

    def step_approx(a):
        return 0.5 + 0.5 * tf.sign(a)

    def body(idx, pred_surv, emp_surv, emd_loss):
        high = tf.gather(t_range, idx)
        low = tf.cond(tf.equal(idx, 0), lambda: tf.constant(0.0), lambda: tf.gather(t_range, idx - 1))

        ## death = t < r and t >= low and e == 1
        # past t< low ; at_risk =  len(empirical) - np.sum(past)

        ## pred compute
        ## step = - (step_approx(z-high) - step_approx(z -low))
        predicted_high = predicted - high
        low_predicted = predicted - low
        pred_in_range = tf.subtract(step_approx(low_predicted), step_approx(predicted_high))
        pred_death = tf.reduce_sum(tf.multiply(e, pred_in_range))
        pred_risk = batch_size - tf.reduce_sum(step_approx(-low_predicted))

        ## emp compute
        empirical_high = empirical - high
        low_empirical = empirical - low
        emp_in_range = tf.subtract(step_approx(low_empirical), step_approx(empirical_high))
        emp_death = tf.reduce_sum(tf.multiply(e, emp_in_range))
        emp_risk = batch_size - tf.reduce_sum(step_approx(-low_empirical))

        ## compute pred
        pred_hazard = tf.div(tf.cast(pred_death, dtype=tf.float32),
                             tf.cast(pred_risk, dtype=tf.float32) + tf.constant(1e-8))
        pred_int_surv = tf.subtract(tf.constant(1.0), pred_hazard)
        pred_surv = tf.multiply(pred_surv, pred_int_surv)

        ## compute emp
        emp_hazard = tf.div(tf.cast(emp_death, dtype=tf.float32),
                            tf.cast(emp_risk, dtype=tf.float32) + tf.constant(1e-8))
        emp_int_surv = tf.subtract(tf.constant(1.0), emp_hazard)
        emp_surv = tf.multiply(emp_surv, emp_int_surv)

        emd_loss_int = tf.losses.absolute_difference(labels=emp_surv, predictions=pred_surv)
        emd_loss = tf.add(emd_loss, emd_loss_int)

        return [idx + 1, pred_surv, emp_surv, emd_loss]

    init_pred_surv = tf.constant(1.0)
    init_emp_surv = tf.constant(1.0)
    init_emd_loss = tf.constant(0.0)
    init_idx = tf.constant(0)
    _, total_pred_surv, total_emp_surv, total_emd_loss = tf.while_loop(cond, body,
                                                                       loop_vars=[init_idx, init_pred_surv,
                                                                                  init_emp_surv, init_emd_loss],
                                                                       shape_invariants=[init_idx.get_shape(),
                                                                                         init_pred_surv.get_shape(),
                                                                                         init_emp_surv.get_shape(),
                                                                                         init_emd_loss.get_shape()])
    return total_pred_surv, total_emp_surv, total_emd_loss
