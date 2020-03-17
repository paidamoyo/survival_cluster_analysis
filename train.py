import os
import argparse

import data.flchain.flchain_data as flchain_data
import data.support.support_data as support_data
import data.seer.seer_data as seer_data
import data.sleep.sleep_data as sleep_data
import importlib
from model.sca import SCA


def init_config():
    parser = argparse.ArgumentParser(description='Survival Cluster Analysis')

    # model hyperparameters
    parser.add_argument('--dataset', type=str, default='support', help='dataset in [support, flchain, seer, sleep]')
    parser.add_argument('--GPUID', type=str, default='1', help='GPU ID')
    parser.add_argument('--n_clusters', type=int, default=25, help='K upper bound of number of clusters')
    parser.add_argument('--gamma_0', type=int, default=2,
                        help='concentration parameter in Dirichlet Process, selected from {2, 3,4,8}')

    args = parser.parse_args()

    # load config file into args
    config_file = "configs"
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
    parser.print_help()
    return args


if __name__ == '__main__':
    args = init_config()
    GPUID = args.GPUID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    flchain = {"path": '', "preprocess": flchain_data, "epochs": 600}
    support = {"path": '', "preprocess": support_data, "epochs": 400}
    seer = {"path": '', "preprocess": seer_data, "epochs": 40}  # download data from SEER website
    sleep = {"path": '', "preprocess": sleep_data, "epochs": 300}  # download data from SLEEP website
    all_datasets = {'support': support, 'flchain': flchain, 'seer': seer, 'sleep': sleep}

    data = all_datasets[args.dataset]

    data_set = data['preprocess'].generate_data()
    train_data, valid_data, test_data = data_set['train'], data_set['valid'], data_set['test']

    perfomance_record = []

    non_par = SCA(batch_size=args.batch_size,
                  learning_rate=args.learning_rate,
                  beta1=args.beta1,
                  beta2=args.beta2,
                  require_improvement=args.require_improvement,
                  num_iterations=args.num_iterations, seed=args.seed,
                  l2_reg=args.l2_reg,
                  hidden_dim=args.hidden_dim,
                  train_data=train_data, test_data=test_data, valid_data=valid_data,
                  input_dim=train_data['x'].shape[1],
                  num_examples=train_data['x'].shape[0], keep_prob=args.keep_prob, sample_size=args.sample_size,
                  max_epochs=data['epochs'], gamma_0=args.gamma_0, n_clusters=args.n_clusters)

    with non_par.session:
        non_par.train_test()
