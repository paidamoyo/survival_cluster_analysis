params = {
    'num_iterations': 40000,
    'batch_size': 500,
    'seed': 31415,
    'require_improvement': 10000,  # num of iterations before early stopping
    'learning_rate': 3e-4,
    'beta1': 0.9,  # Adam  optimizer beta 1
    'beta2': 0.999,  # Adam optimizer beta 1
    'hidden_dim': [50, 50, 50],
    'l2_reg': 0.001,  # l2 regularization weight multiplier (just for debugging not optimization)
    'l1_reg': 0.001,  # l1 regularization weight multiplier (just for debugging not optimization)
    'keep_prob': 0.8,  # keep prob for weights implementation in layers
    'sample_size': 200  # number of samples of generated time for p(T|x)
}
