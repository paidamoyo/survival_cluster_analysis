import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

seed = 31415
np.random.seed(seed)

fontsize = 18
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

plt.rc('font', **font)
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

sns.set_style('white')
sns.set_context('paper')
sns.set()
title_fontsize = 18
label_fontsize = 18


def plot_cost(training, validation, name, model, epochs, best_epoch):
    x = np.arange(start=0, stop=len(training), step=1).tolist()
    constant = 1e-10
    plt.figure()
    plt.xlim(min(x), max(x))
    plt.ylim(min(min(training), min(validation), 0) - constant, max(max(training), max(validation)) + constant)
    plt.plot(x, training, color='blue', linestyle='-', label='training')
    plt.plot(x, validation, color='green', linestyle='-', label='validation')
    plt.axvline(x=best_epoch, color='red')
    title = 'Training {} {}: epochs={}, best epoch={} '.format(model, name, epochs, best_epoch)
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend(loc='best', fontsize=10)
    plt.savefig('plots/{}_{}'.format(model, name))


def l2_loss(scale):
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return l2 * scale


def l1_loss(scale):
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=scale, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    l1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    return l1
