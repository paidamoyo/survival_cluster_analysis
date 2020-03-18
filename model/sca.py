import logging
import os
import threading
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
from scipy.stats.stats import spearmanr

from model.risk_network import pz_given_x, pt_given_z
from utils.accuracy import accuracy_loss
from utils.metrics import plot_cost, l2_loss, l1_loss
from utils.clustering import cluster_assignment, update_pop_pi, run_k_means
from utils.calibration import km_estimator
import math
from utils.tf_helpers import create_centroids, create_pop_pi, show_all_variables


class SCA(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 hidden_dim,
                 input_dim,
                 num_examples,
                 keep_prob,
                 train_data,
                 valid_data,
                 test_data,
                 sample_size,
                 l2_reg,
                 max_epochs,
                 gamma_0,
                 n_clusters,
                 path_large_data=""
                 ):

        self.max_epochs = max_epochs
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.path_large_data = path_large_data
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.l2_reg = l2_reg
        self.log_file = 'model.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = True
        self.sample_size = sample_size

        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        # Load Data
        self.train_x, self.train_t, self.train_e = train_data['x'], train_data['t'], train_data['e']
        self.valid_x, self.valid_t, self.valid_e = valid_data['x'], valid_data['t'], valid_data['e']

        self.test_x, self.test_t, self.test_e = test_data['x'], test_data['t'], test_data['e']
        self.keep_prob = keep_prob
        self.input_dim = input_dim
        self.num_examples = num_examples
        self.n_clusters = n_clusters
        self.gamma_0 = gamma_0
        self.train_k = []
        self.turn_clust = False

        self.model = 'SCA'

        self._build_graph()
        self.train_cost, self.train_ci, self.train_calibration, self.train_accuracy, self.train_clustering, \
            = [], [], [], [], []
        self.valid_cost, self.valid_ci, self.valid_calibration, self.valid_accuracy, self.valid_clustering, \
            = [], [], [], [], []

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.e = tf.placeholder(tf.float32, shape=[None], name='e')
            self.t = tf.placeholder(tf.float32, shape=[None], name='t')
            # are used to feed data into our queue
            self.batch_size_tensor = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.batch_size_tensor_lab = tf.placeholder(tf.int32, shape=[], name='batch_size_tensor_lab')
            self.is_training = tf.placeholder(tf.bool)
            self.cluster_alpha = tf.placeholder(tf.bool)
            self.t_range = tf.placeholder(tf.float32, shape=[None], name='t_range')
            self.t_range_size = tf.placeholder(tf.int32, shape=[], name='t_range_size')

            self._objective()
            self.session = tf.Session(config=self.config)

            self.capacity = 1400
            self.coord = tf.train.Coordinator()
            enqueue_thread = threading.Thread(target=self.enqueue)
            self.queue = tf.RandomShuffleQueue(capacity=self.capacity,
                                               dtypes=[tf.float32, tf.float32, tf.float32],
                                               shapes=[[self.input_dim], [], []],
                                               min_after_dequeue=self.batch_size)

            self.enqueue_op = self.queue.enqueue_many([self.x, self.t, self.e])
            enqueue_thread.start()
            dequeue_op = self.queue.dequeue()
            self.x_batch, self.t_batch, self.e_batch = tf.train.batch(
                dequeue_op, batch_size=self.batch_size,
                capacity=self.capacity)
            self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.session)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.current_dir = os.getcwd()
            self.save_path = "summaries/{0}_model".format(self.model)
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)

    def _objective(self):
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self._build_model()
        self.reg_loss = l2_loss(self.l2_reg) + l1_loss(self.l2_reg)
        clust_lambda = tf.cond(self.cluster_alpha, lambda: 1.0, lambda: 0.0)
        self.cost = self.accuracy_loss + self.calibration_loss + clust_lambda * self.cluster_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def _build_model(self):
        self._cluster_data()
        self._risk_model()
        self._risk_surv_fun_match()
        self._initialize_centroids()

    def _initialize_centroids(self):
        self.k_means_cent = run_k_means(centroids=self.centroids, features=self.z, num_iter=100,
                                        n_clusters=self.n_clusters)

    def _cluster_data(self):
        self.centroids_dim = self.hidden_dim[len(self.hidden_dim) - 2]
        self.z = pz_given_x(x=self.x, is_training=self.is_training, batch_norm=self.batch_norm,
                            hidden_dim=self.hidden_dim,
                            scope="x", keep_prob=self.keep_prob)

        self.centroids = create_centroids(centers=self.n_clusters, centoids_dim=self.centroids_dim)
        self.pop_pi_k = create_pop_pi(n_clusters=self.n_clusters)
        self.nearest_indices, kl_loss, batch_pop_pi, self.curr_post, self.curr_lik = cluster_assignment(
            centroids=self.centroids,
            pop_pi=self.pop_pi_k,
            embed_z=self.z,
            n_clusters=self.n_clusters,
            batch_size=self.batch_size_tensor, gamma_0=self.gamma_0)

        self.pop_pi_k = tf.cond(self.cluster_alpha, lambda: update_pop_pi(batch_pop_pi=batch_pop_pi,
                                                                          pop_pi=self.pop_pi_k,
                                                                          is_training=self.is_training),
                                lambda: self.pop_pi_k)

        self.cluster_loss = kl_loss

    def _risk_model(self):
        indices_lab = tf.where(tf.equal(tf.constant(1.0, dtype=tf.float32), self.e))
        print("indices_lab: ", indices_lab.shape)

        t_gen, _ = pt_given_z(z=self.z, hidden_dim=self.hidden_dim, is_training=self.is_training,
                              batch_norm=self.batch_norm, keep_prob=self.keep_prob,
                              batch_size=self.batch_size_tensor,
                              reuse=True)

        self.predicted_time = tf.squeeze(t_gen)
        print("predicted_time: ", self.predicted_time.shape)

        self.accuracy_loss = accuracy_loss(e=self.e,
                                           predicted=self.predicted_time,
                                           batch_size=self.batch_size_tensor,
                                           empirical=self.t)

    def _risk_surv_fun_match(self):

        pop_pred_surv, pop_emp_surv, pop_km_loss = km_estimator(predicted=self.predicted_time,
                                                                t_range=self.t_range, e=self.e,
                                                                batch_size=self.batch_size_tensor,
                                                                t_range_size=self.t_range_size,
                                                                empirical=self.t)

        self.calibration_loss = pop_km_loss
        self.pred_surv = pop_pred_surv
        self.emp_surv = pop_emp_surv

    def train_neural_network(self):
        train_print = "Training {0} Model:".format(self.model)
        params_print = "Parameters:, l2_reg:{}, learning_rate:{}," \
                       " momentum: beta1={} beta2={}, batch_size:{}, batch_norm:{}," \
                       " hidden_dim:{}, num_of_batches:{}, keep_prob:{}, n_clusters:{}, gamma_0:{}" \
            .format(self.l2_reg, self.learning_rate, self.beta1, self.beta2, self.batch_size,
                    self.batch_norm, self.hidden_dim, self.num_batches, self.keep_prob, self.n_clusters, self.gamma_0)
        print(train_print)
        print(params_print)
        logging.debug(train_print)
        logging.debug(params_print)
        self.session.run(tf.global_variables_initializer())

        best_calibration = np.inf
        best_ci = 0
        best_valid_epoch = 0
        last_improvement = 0

        start_time = time.time()
        epochs = 0
        show_all_variables()
        j = 0

        for i in range(self.num_iterations):
            # Batch Training
            run_options = tf.RunOptions(timeout_in_ms=4000)
            x_batch, t_batch, e_batch = self.session.run(
                [self.x_batch, self.t_batch, self.e_batch],
                options=run_options)
            batch_size = len(t_batch)
            unique_t = self.sorted_unique_t(t_batch)
            # TODO simplify batch processing
            feed_dict_train = {self.x: x_batch,
                               self.t: t_batch,
                               self.e: e_batch,
                               self.batch_size_tensor: batch_size,
                               self.batch_size_tensor_lab: np.sum(e_batch),
                               self.is_training: True,
                               self.t_range: unique_t,
                               self.cluster_alpha: self.turn_clust,
                               self.t_range_size: len(unique_t)
                               }

            summary, train_time, train_cost, train_clustering, train_reg, train_accuracy, train_calibration, \
            train_k, _ = self.session.run(
                [self.merged, self.predicted_time, self.cost, self.cluster_loss,
                 self.reg_loss, self.accuracy_loss, self.calibration_loss,
                 self.nearest_indices, self.optimizer], feed_dict=feed_dict_train)

            train_pred_surv, train_emp_surv, train_km_loss, curr_pop_nk, curr_post, curr_lik = self.session.run(
                [self.pred_surv, self.emp_surv, self.calibration_loss, self.pop_pi_k, self.curr_post, self.curr_lik],
                feed_dict=feed_dict_train)

            try:
                train_ci = concordance_index(event_times=t_batch,
                                             predicted_event_times=train_time.reshape(t_batch.shape),
                                             event_observed=e_batch)
            except IndexError:
                train_ci = 0.0
                print("C-Index IndexError")

            tf.verify_tensor_all_finite(train_cost, "Training Cost has Nan or Infinite")
            if j >= self.num_examples:
                epochs += 1
                is_epoch = True
                j = 0
            else:
                j += self.batch_size
                is_epoch = False

            if i % 100 == 0:
                train_print = "ITER:{}, Train CI:{}, Clustering:{}," \
                              "Reg:{}, Accuracy:{}, Calibration:{}, Cost:{}".format(
                    i, train_ci, train_clustering, train_reg, train_accuracy, train_calibration, train_cost)
                print(train_print)
                logging.debug(train_print)

            clust_epoch = int(self.max_epochs * 0.1)
            if epochs == clust_epoch:
                k_means_cent, centroids_before = self.session.run([self.k_means_cent, self.centroids],
                                                                  feed_dict=feed_dict_train)
                print("running k_means: ", epochs)
                alpha = 0.2
                self.centroids = tf.assign(self.centroids, value=self.centroids * alpha + k_means_cent * (1 - alpha))
                self.turn_clust = True

            if is_epoch or (i == (self.num_iterations - 1)):
                improved_str = ''
                # Calculate  Vaid CI the CI
                self.train_ci.append(train_ci)
                self.train_cost.append(train_cost)
                self.train_calibration.append(train_calibration)
                self.train_accuracy.append(train_accuracy)
                self.train_clustering.append(train_clustering)
                self.train_k.append(train_k)

                self.train_writer.add_summary(summary, i)
                valid_ci, valid_cost, valid_clustering, valid_reg, valid_calibration, valid_accuracy, \
                    = self.predict_concordance_index(
                    x=self.valid_x,
                    e=self.valid_e,
                    t=self.valid_t, num_samples=50)

                self.valid_ci.append(valid_ci)
                self.valid_cost.append(valid_cost)
                self.valid_calibration.append(valid_calibration)
                self.valid_accuracy.append(valid_accuracy)
                self.valid_clustering.append(valid_clustering)
                tf.verify_tensor_all_finite(valid_cost, "Validation Cost has Nan or Infinite")

                if valid_ci >= best_ci and (
                        valid_calibration <= best_calibration or valid_calibration - best_calibration <= 1.0):
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    print_example = " Final curr_post:{}, curr_lik:{}, K:{}".format(curr_post, curr_lik,
                                                                                    len(np.unique(train_k)))
                    print(print_example)
                    logging.debug(print_example)

                    best_valid_epoch = epochs
                    if valid_calibration <= best_calibration:
                        best_calibration = valid_calibration
                    best_ci = valid_ci
                    last_improvement = i
                    improved_str = '*'
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                # update best validation accuracy
                surv_print = "pred_surv:{},  emp_surv:{},  km_loss:{}, len_unique_t:{}, " \
                             "curr_pop_nk, min:{}, max:{}, sum:{}, curr_post:{}  " \
                             "curr_lik:{}, train_k:{}, \n".format(train_pred_surv, train_emp_surv,
                                                                  np.sum(train_km_loss), len(unique_t),
                                                                  np.min(curr_pop_nk), np.max(curr_pop_nk),
                                                                  np.sum(curr_pop_nk), curr_post, curr_lik,
                                                                  len(np.unique(train_k)))

                logging.debug(surv_print)
                print(surv_print)

                perf_sum_print = "valid_calibration:{}, best_calibration:{}, valid_ci:{},best_ci:{}, best_epoch:{} \n ".format(
                    valid_calibration, best_calibration,
                    valid_ci, best_ci, best_valid_epoch)
                print(perf_sum_print)
                logging.debug(perf_sum_print)

                optimization_print = "Iteration: {} epochs:{}, Train Loss: {}, " \
                                     "Reg:{}, Calibration:{}, Accuracy:{},  CI:{}, Clustering:{}" \
                                     "Valid Loss:{}, Reg:{}, Calibration:{}, " \
                                     "Accuracy:{}, CI:{}, Clustering:{}, {} \n" \
                    .format(i + 1, epochs, train_cost, train_reg, train_calibration,
                            train_accuracy, train_ci, train_clustering, valid_cost, valid_reg,
                            valid_calibration, valid_accuracy, valid_ci, valid_clustering, improved_str)

                print(optimization_print)
                logging.debug(optimization_print)

                if i - last_improvement > self.require_improvement or math.isnan(
                        train_cost) or epochs >= self.max_epochs:
                    print("No improvement found in a while, stopping optimization.")
                    # Break out from the for-loop.
                    break
        # Ending time.

        end_time = time.time()
        time_dif = end_time - start_time
        time_dif_print = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(time_dif_print)
        logging.debug(time_dif_print)
        # shutdown everything to avoid zombies
        self.session.run(self.queue.close(cancel_pending_enqueues=True))
        self.coord.request_stop()
        self.coord.join(self.threads)
        return best_valid_epoch, epochs

    def get_dict(self, x, t, e):
        unique_t = self.sorted_unique_t(t)

        feed_dict = {self.x: x,
                     self.t: t,
                     self.e: e,
                     self.batch_size_tensor: len(t),
                     self.batch_size_tensor_lab: np.sum(e),
                     self.is_training: False,
                     self.cluster_alpha: True,
                     self.t_range: unique_t,
                     self.t_range_size: len(unique_t)
                     }
        return feed_dict

    def train_test(self, train=True):

        session_dict = {
            'Test': self.get_dict(x=self.test_x, t=self.test_t, e=self.test_e),
            'Train': self.get_dict(x=self.train_x, t=self.train_t, e=self.train_e),
            'Valid': self.get_dict(x=self.valid_x, t=self.valid_t, e=self.valid_e)}

        if train:
            best_epoch, epochs = self.train_neural_network()
            centroids, pop_nk, curr_post, curr_lik = self.session.run(
                [self.centroids, self.pop_pi_k, self.curr_post, self.curr_lik],
                feed_dict=
                self.get_dict(x=self.test_x, t=self.test_t, e=self.test_e))

            np.save("matrix/Test_centroids", centroids)
            np.save("matrix/Test_pop_nk", pop_nk)
            np.save("matrix/learned_K", self.train_k)

            self.time_related_metrics(best_epoch, epochs, session_dict=session_dict)

        else:
            self.generate_statistics(data_x=self.test_x, data_e=self.test_e, data_t=self.test_t, name='Test',
                                     data_dict=session_dict)

        self.session.close()

    def save_generated_h2(self, feed_dict, name):
        z = self.session.run(self.z, feed_dict=feed_dict)
        np.save("matrix/{}_z".format(name), z)

    def time_related_metrics(self, best_epoch, epochs, session_dict):
        plot_cost(training=self.train_cost, validation=self.valid_cost, model=self.model, name="Cost",
                  epochs=epochs, best_epoch=best_epoch)

        plot_cost(training=self.train_ci, validation=self.valid_ci, model=self.model, name="CI",
                  epochs=epochs, best_epoch=best_epoch)

        plot_cost(training=self.train_clustering, validation=self.valid_clustering, model=self.model, name="Clustering",
                  epochs=epochs, best_epoch=best_epoch)

        plot_cost(training=self.train_calibration, validation=self.valid_calibration, model=self.model,
                  name="Calibration", epochs=epochs, best_epoch=best_epoch)

        plot_cost(training=self.train_accuracy, validation=self.valid_accuracy, model=self.model, name="Accuracy",
                  epochs=epochs, best_epoch=best_epoch)

        # TEST
        self.generate_statistics(data_x=self.test_x, data_e=self.test_e, data_t=self.test_t, name='Test',
                                 data_dict=session_dict['Test'])
        self.save_generated_h2(feed_dict=session_dict['Test'], name='Test')

        # VALID
        self.generate_statistics(data_x=self.valid_x, data_e=self.valid_e, data_t=self.valid_t, name='Valid',
                                 data_dict=session_dict['Valid'])
        # TRAIN
        self.generate_statistics(data_x=self.train_x, data_e=self.train_e, data_t=self.train_t, name='Train',
                                 data_dict=session_dict['Train'])

    def generate_statistics(self, data_x, data_e, data_t, name, data_dict, save=True):
        self.saver.restore(sess=self.session, save_path=self.save_path)
        # valid_ci, valid_cost, valid_clustering, valid_reg, valid_emd, valid_t_reg
        ci, cost, clustering, reg, earth_movers, t_reg = \
            self.predict_concordance_index(x=data_x,
                                           e=data_e,
                                           t=data_t)

        observed_idx = self.extract_observed_death(name=name, observed_e=data_e, observed_t=data_t, save=save)

        median_predicted_time = self.median_predict_time(data_dict)
        print("data_dict keys: ", data_dict.keys())

        if name == 'Test':
            predicted_samples = self.generate_time_samples(data_dict=data_dict)
            np.save('matrix/{}_predicted_t_median'.format(name), median_predicted_time)
            np.save('matrix/{}_empirical_t'.format(name), data_t)
            np.save('matrix/{}_empirical_e'.format(name), data_e)
            np.transpose(np.save('matrix/{}_predicted_t_samples'.format(name), predicted_samples))

        observed_empirical = data_t[observed_idx]
        observed_predicted = median_predicted_time[observed_idx]
        observed_ci = concordance_index(event_times=observed_empirical, predicted_event_times=observed_predicted,
                                        event_observed=data_e[observed_idx])

        corr = spearmanr(observed_empirical, observed_predicted)
        results = ":{}, Loss:{}, Reg:{}, Clustering {}, Accuracy:{}, Calibration:{}," \
                  " CI:{}, Observed: CI:{}, Correlation:{}".format(name, cost, reg, clustering, t_reg, earth_movers, ci,
                                                                   observed_ci, corr)
        logging.debug(results)
        print(results)

    def predict_concordance_index(self, x, t, e, num_samples=None):
        if not num_samples:
            num_samples = self.sample_size
        input_size = x.shape[0]
        i = 0
        num_batches = input_size / self.batch_size
        predicted_time = np.zeros(shape=input_size, dtype=np.int)
        total_clustering = 0.0
        total_cost = 0.0
        total_t_reg_loss = 0.0
        total_reg = 0.0
        total_emd = 0.0
        while i < input_size:
            # The ending index for the next batch is denoted j.
            j = min(i + self.batch_size, input_size)
            feed_dict = self.batch_feed_dict(e=e, i=i, j=j, t=t, x=x)

            temp_pred_time = []
            temp_km_loss = []
            temp_cost = []
            temp_clustering = []
            temp_reg = []
            temp_t_reg_loss = []

            for p in range(num_samples):
                gen_time, t_emd, cost, clustering, reg, t_reg_loss = self.session.run(
                    [self.predicted_time, self.calibration_loss, self.cost, self.cluster_loss,
                     self.reg_loss, self.accuracy_loss], feed_dict=feed_dict)
                temp_pred_time.append(gen_time)
                temp_km_loss.append(t_emd)
                temp_cost.append(cost)
                temp_clustering.append(clustering)
                temp_reg.append(reg)
                temp_t_reg_loss.append(t_reg_loss)

            temp_pred_time = np.array(temp_pred_time)
            temp_km_loss = np.array(temp_km_loss)
            mean_km_loss = np.mean(temp_km_loss, axis=0)
            print("valid batch_i: ", i, "km_loss: ", mean_km_loss)
            # print("temp_pred_time:{}".format(temp_pred_time.shape))
            predicted_time[i:j] = np.median(temp_pred_time, axis=0)

            total_clustering += np.mean(temp_clustering, axis=0)
            total_cost += np.mean(temp_cost, axis=0)
            total_reg += np.mean(temp_reg, axis=0)
            total_t_reg_loss += np.mean(temp_t_reg_loss, axis=0)
            total_emd += mean_km_loss
            i = j

        predicted_event_times = predicted_time.reshape(input_size)
        ci_index = concordance_index(event_times=t, predicted_event_times=predicted_event_times.tolist(),
                                     event_observed=e)

        def batch_average(total):
            return total / num_batches

        # valid_ci, valid_cost, valid_clustering, valid_reg, valid_km, valid_t_reg
        print("total emd: ", batch_average(total_emd))
        return ci_index, batch_average(total_cost), batch_average(total_clustering), batch_average(
            total_reg), batch_average(total_emd), batch_average(total_t_reg_loss)

    def batch_feed_dict(self, e, i, j, t, x):
        batch_x = x[i:j, :]
        batch_t = t[i:j]
        batch_e = e[i:j]
        unique_t = self.sorted_unique_t(batch_t)
        feed_dict = {self.x: batch_x,
                     self.t: batch_t,
                     self.e: batch_e,
                     self.batch_size_tensor: len(batch_t),
                     self.batch_size_tensor_lab: np.sum(batch_e),
                     self.is_training: False,
                     self.t_range: unique_t,
                     self.t_range_size: len(unique_t),
                     self.cluster_alpha: True
                     }
        return feed_dict

    def median_predict_time(self, data_dict):
        predicted_time = []
        for p in range(self.sample_size):
            gen_time = self.session.run(self.predicted_time, feed_dict=data_dict)
            predicted_time.append(gen_time)
        predicted_time = np.array(predicted_time)
        # print("predicted_time_shape:{}".format(predicted_time.shape))
        return np.median(predicted_time, axis=0)

    def generate_time_samples(self, data_dict):
        # observed = e == 1
        predicted_time = []
        for p in range(self.sample_size):
            gen_time = self.session.run(self.predicted_time, feed_dict=data_dict)
            predicted_time.append(gen_time)
        predicted_time = np.array(predicted_time)
        return predicted_time

    def enqueue(self):
        """ Iterates over our data puts small junks into our queue."""
        # TensorFlow Input Pipelines for Large Data Sets
        # ischlag.github.io
        # http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
        # http://web.stanford.edu/class/cs20si/lectures/slides_09.pdf
        under = 0
        max = len(self.train_x)
        try:
            while not self.coord.should_stop():
                # print("starting to write into queue")
                upper = under + self.capacity
                # print("try to enqueue ", under, " to ", upper)
                if upper <= max:
                    curr_x = self.train_x[under:upper]
                    curr_t = self.train_t[under:upper]
                    curr_e = self.train_e[under:upper]
                    under = upper
                else:
                    rest = upper - max
                    curr_x = np.concatenate((self.train_x[under:max], self.train_x[0:rest]))
                    curr_t = np.concatenate((self.train_t[under:max], self.train_t[0:rest]))
                    curr_e = np.concatenate((self.train_e[under:max], self.train_e[0:rest]))
                    under = rest

                self.session.run(self.enqueue_op,
                                 feed_dict={self.x: curr_x, self.t: curr_t, self.e: curr_e})
        except tf.errors.CancelledError:
            print("finished enqueueing")

    @staticmethod
    def extract_observed_death(name, observed_e, observed_t, save=False):
        idx_observed = observed_e == 1
        observed_death = observed_t[idx_observed]
        if save:
            death_observed_print = "{} observed_death:{}, percentage:{}".format(name, observed_death.shape, float(
                len(observed_death) / len(observed_t)))
            logging.debug(death_observed_print)
            print(death_observed_print)
        return idx_observed

    @staticmethod
    def sorted_unique_t(batch_t):
        unique = np.unique(np.sort(np.append(batch_t, [0])))
        return unique
