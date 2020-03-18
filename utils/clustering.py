import tensorflow as tf
import numpy as np


def t_dist_distance(a, b):
    expanded_vectors = tf.expand_dims(a, 0)
    expanded_centroids = tf.expand_dims(b, 1)
    # T-distributed with \nu = 1
    distances = 1 + tf.reduce_sum(tf.square(
        tf.subtract(expanded_vectors, expanded_centroids)), 2)

    return distances


def run_k_means(centroids, features, n_clusters, num_iter):
    updated_centroid_value = None
    for _ in np.arange(num_iter):
        nearest_indices = assign_to_nearest(samples=features, centroids=centroids)
        new_centroids = update_centroids(samples=features, nearest_indices=nearest_indices,
                                         n_clusters=n_clusters, old_centroids=centroids)
        updated_centroid_value = new_centroids
    return updated_centroid_value


def assign_to_nearest(samples, centroids):
    distances = t_dist_distance(a=samples, b=centroids)
    mins = tf.argmin(distances, axis=0)
    nearest_indices = mins
    return nearest_indices


def update_centroids(samples, nearest_indices, n_clusters, old_centroids):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    new_centroids = tf.where(condition=tf.is_nan(new_centroids), x=old_centroids, y=new_centroids)
    return new_centroids


def cluster_assignment(centroids, embed_z, n_clusters, batch_size, gamma_0, pop_pi):
    distance = tf.transpose(t_dist_distance(a=embed_z, b=centroids))

    p_assign, likelihood, kl_loss, curr_post, curr_prior = compute_prob(dist=distance, n_clusters=n_clusters,
                                                                        batch_size=batch_size, gamma_0=gamma_0,
                                                                        curr_pop_nk=pop_pi)
    nearest_indices = tf.argmax(p_assign, axis=1)
    curr_batch_nk = tf.reduce_sum(likelihood, axis=0) / (tf.cast(batch_size, dtype=tf.float32))

    return nearest_indices, kl_loss, curr_batch_nk, curr_post, curr_prior


def compute_prob(dist, n_clusters, batch_size, gamma_0, curr_pop_nk):
    """Computes KL loss """
    likelihood = 1.0 / dist
    likelihood = likelihood / tf.reduce_sum(likelihood, axis=1, keepdims=True)

    prior_weights = compute_pi(curr_pop_pi_k=curr_pop_nk, n_clusters=n_clusters, batch_size=batch_size, gamma_0=gamma_0)

    p_mle = likelihood * curr_pop_nk
    p_mle = p_mle / tf.reduce_sum(p_mle, axis=1, keepdims=True)

    alpha_c = likelihood * curr_pop_nk
    alpha_c = tf.contrib.framework.sort(alpha_c, direction='DESCENDING')  ## sort in a descending order
    alpha_0 = tf.reduce_sum(alpha_c, axis=1, keepdims=True)

    beta_c = likelihood * prior_weights
    beta_c = tf.contrib.framework.sort(beta_c, direction='DESCENDING')  ## sort in a descending order
    beta_0 = tf.reduce_sum(beta_c, axis=1, keepdims=True)

    digamma_diff = tf.digamma(alpha_c) - tf.digamma(alpha_0)
    geometric_mean = tf.reduce_sum((alpha_c - beta_c) * digamma_diff, axis=1)

    conc_diff = tf.log(tf.lgamma(alpha_0)) - tf.log(tf.lgamma(beta_0))

    mean_diff = tf.reduce_sum(tf.lgamma(beta_c), axis=1) - tf.reduce_sum(tf.lgamma(alpha_c), axis=1)

    kl_loss = tf.reduce_mean(conc_diff + mean_diff + geometric_mean)
    idx = 0

    return p_mle, likelihood, kl_loss, tf.contrib.framework.sort(p_mle[idx])[
                                       n_clusters - 2:n_clusters], tf.contrib.framework.sort(
        likelihood[idx])[n_clusters - 2:n_clusters]


def compute_pi(curr_pop_pi_k, n_clusters, batch_size, gamma_0):
    n_k = curr_pop_pi_k * tf.cast(batch_size, dtype=tf.float32)
    b_k = tf.ones_like(n_k) * (1 / (1 + gamma_0))

    prod = 1
    stick_prob = tf.zeros_like(b_k)
    for i in np.arange(n_clusters):
        if i == n_clusters - 1:
            p_k = 1 - tf.reduce_sum(stick_prob)
            p_k = tf.cond(p_k < 0, lambda: 0.0, lambda: p_k)
        else:
            b_i = tf.gather(b_k, indices=i)
            p_k = b_i * prod
            prod = prod * (1 - b_i)
        if prod == 0:
            break
        stick_prob = stick_prob + tf.one_hot(indices=i, depth=n_clusters) * p_k

    return stick_prob


def update_pop_pi(batch_pop_pi, pop_pi, is_training):
    alpha = 0.9  # use numbers closer to 1 if you have more data

    def batch_curr():
        train_pop_pi = tf.assign(pop_pi, pop_pi * alpha + batch_pop_pi * (1 - alpha))
        return train_pop_pi

    def pop_final():
        return pop_pi

    return tf.cond(is_training, batch_curr, pop_final)
