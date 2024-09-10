from __future__ import print_function
import sys
sys.path.append("../functions")
from utils import *
import csv
import argparse
import os
import torch
import time
from model import GNNNet, train_model, test_model, get_graph_data_loader, get_test_fc_adj
import torch.optim as optim
import pickle
from sklearn.model_selection import KFold, train_test_split


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 8, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 8}
    if use_cuda:
        cuda_kwargs = {'pin_memory': True, 'num_workers': 4}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    time_start = time.time()
    dataset_type = args.dataset
    print(dataset_type, " dataset")


    if dataset_type == 'HCPYA':
        with open('/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/HCP_SC_FC_reSC_info_297.pickle', 'rb') as out_data:
            (SC, FC, re_SC, sub_list) = pickle.load(out_data)

        SC_train, SC_test, FC_train, FC_test, re_SC_train, re_SC_test, = train_test_split(SC, FC, re_SC, test_size=0.5, random_state=args.seed)


    else:
        with open(
                '/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/preprocessed_data/final/HCPD_SC_FC_reSC_info_499.pickle',
                'rb') as out_data:
            (SC, FC, re_SC, info) = pickle.load(out_data)

        SC_train, SC_test, FC_train, FC_test, re_SC_train, re_SC_test = train_test_split(SC, FC, re_SC, test_size=0.5,
                                                                                          random_state=args.seed)


    train_data_x = SC_train
    test_data_x = SC_test
    train_data_y = FC_train
    test_data_y = FC_test

    # rewire the SC_train
    if args.use_rewired:
        train_data_x = re_SC_train

    elif args.rewired>0:
        re_SC_train = rewired_graph_parallel(SC_train, args.rewired, 10)
        train_data_x = re_SC_train
        time_end = time.time()
        print('rewire process end', time_end - time_start, 's')


    for times in range(1):

        if args.if_kfold:
            kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

            X = SC_train
            Y = FC_train

            for train, test in kfold.split(X, Y):
                train_data_x = X[train, :, :]
                test_data_x = X[test, :, :]
                train_data_y = Y[train, :, :]
                test_data_y = Y[test, :, :]

                time_end = time.time()
                print('begin', time_end - time_start, 's')

                train_loader = get_graph_data_loader(train_data_x, train_data_y, train_kwargs)
                test_loader = get_graph_data_loader(test_data_x, test_data_y, test_kwargs)

                time_end = time.time()
                print('build loader', time_end - time_start, 's')

                model = GNNNet(layer_num=args.layer_num, feature_num=train_data_x.shape[1], conv_dim=args.conv_dim).to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.lr)

                for epoch in range(1, args.epochs + 1):
                    time_end = time.time()
                    print('train_epoch', time_end - time_start, 's')
                    train_model(args, model, device, train_loader, optimizer, epoch)
                    if epoch % args.checkperiod == 0 or epoch == args.epochs:
                        temp_corr_ = test_model(model, device, test_loader)

                    if epoch == args.epochs:
                        temp_corr = test_model(model, device, test_loader)

                    time_end = time.time()
                    print('test_epoch', time_end - time_start, 's')


        else:
            time_end = time.time()
            print('begin', time_end - time_start, 's')

            # use one-hot to represent each node, shared for all the graph
            train_loader = get_graph_data_loader(train_data_x, train_data_y, train_kwargs)
            test_loader = get_graph_data_loader(test_data_x, test_data_y, test_kwargs)

            time_end = time.time()
            print('build loader', time_end - time_start, 's')

            model = GNNNet(layer_num=args.layer_num, conv_dim=args.conv_dim, feature_num=train_data_x.shape[1]).to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                train_model(args, model, device, train_loader, optimizer, epoch)
                time_end = time.time()
                print('train_epoch：', epoch, time_end - time_start, 's')
                time_end = time.time()
                print('test_epoch', time_end - time_start, 's')

            if args.save_model:
                save_model_path = "/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_model/" + args.dataset
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                if args.rewired==0 and not args.use_rewired:
                        torch.save(model.state_dict(),
                               save_model_path + "/GNN_GCN_dnn_dim" + str(
                                   args.conv_dim) + 'lr' + str(args.lr) + 'b' + str(args.batch_size) + 'ep' + str(args.epochs)+ 'ln' + str(args.layer_num)+"prelu.pt")
                else:
                    save_model_path += '/rewired'
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                    torch.save(model.state_dict(),
                               save_model_path + "/rewired_GNN_GCN_dim" + str(
                                   args.conv_dim) + 'lr' + str(args.lr) + 'b' + str(args.batch_size) + 'ep' + str(
                                   args.epochs) + 'reg' + str(args.reg)+ "prelu.pt")


            if args.get_result:
                FC_train_pred = get_test_fc_adj(model, train_data_x, train_data_y, device)
                FC_test_pred = get_test_fc_adj(model, test_data_x, test_data_y, device)
                save_result_path = "/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/result_out/" + args.dataset + "/"
                if not os.path.exists(save_result_path):
                    os.makedirs(save_result_path)
                if args.rewired>0 or args.use_rewired:
                    save_result_path += "rewired_"
                    FC_test_pred_rewTest, _ = get_test_fc_adj(model, re_SC_test, test_data_y, device)
                    save_result_path += 'GCN_dim' + str(
                                   args.conv_dim) + 'lr' + str(args.lr) + 'b' + str(args.batch_size) + 'ep' + str(
                                   args.epochs) + 'reg' + str(args.reg)+ 'prelu' +'.pickle'
                    with open(save_result_path, 'wb') as in_data:
                        pickle.dump((FC_train_pred, FC_test_pred, FC_test_pred_rewTest), in_data, pickle.HIGHEST_PROTOCOL)

                else:
                    save_result_path += 'GCN_dim' + str(
                                   args.conv_dim) + 'lr' + str(args.lr) + 'b' + str(args.batch_size) + 'ep' + str(
                                   args.epochs) + 'reg' + str(args.reg)+ 'prelu' +'.pickle'
                    with open(save_result_path, 'wb') as in_data:
                        pickle.dump((FC_train_pred, FC_test_pred), in_data, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    print("if cuda available:", torch.cuda.is_available())
    # parser settings
    parser = argparse.ArgumentParser(description='GNN SC FC')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--layer-num', type=int, default=2, metavar='N',
                        help='the layer number of GNN')
    parser.add_argument('--conv-dim', type=int, default=256, metavar='N',
                        help='the conv dim of GNN')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=0.0001, metavar='M',
                        help='regularization parameter')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('--checkepoch', type=int, default=50, metavar='N',
                        help='how many epochs to wait before logging training result')
    parser.add_argument('--checkperiod', type=int, default=20, metavar='N',
                        help='check period')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--if-kfold', default=False,
                        help='5 fold on training set search the optimal hyperparameter')
    parser.add_argument('--rewired', type=int, default=0, metavar='N',
                        help='the rewired parameter, 0 represents no rewired')
    parser.add_argument('--use-rewired', action='store_true', default=False, help='whether use rewired SC')
    parser.add_argument('--dataset', default='HCPYA', help='choose a dataset')
    parser.add_argument('--get-result', action='store_true', default=False, help='if get result')


    args = parser.parse_args()

    main(args)
