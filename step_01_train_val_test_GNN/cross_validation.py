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


    if args.fold == 1:
        temp_SC = SC_train.copy()
        temp_FC = FC_train.copy()
        SC_train = SC_test
        FC_train = FC_test
        SC_test = temp_SC
        FC_test = temp_FC


    train_data_x = SC_train
    test_data_x = SC_test
    train_data_y = FC_train
    test_data_y = FC_test



    for times in range(1):
        train_loader = get_graph_data_loader(train_data_x, train_data_y, train_kwargs)
        model = GNNNet(layer_num=args.layer_num, conv_dim=args.conv_dim, feature_num=train_data_x.shape[1]).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train_model(args, model, device, train_loader, optimizer, epoch)

        if args.save_model:
            save_path = "/home/cuizaixu_lab/chenpeiyu/DATA_C/project/SC_FC_Pred/cross_val/" + args.dataset + "/" + str(
                args.seed) + "_" + str(args.fold) + "_GCN_" + str(
                args.conv_dim) + 'lr' + str(args.lr) + 'b' + str(args.batch_size) + 'ep' + str(
                args.epochs) + 'reg' + str(args.reg) + "prelu_" + args.atlas + ".pt"
            torch.save(model.state_dict(),
                       save_path)



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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed 1-100')
    parser.add_argument('--fold', type=int, default=0, metavar='S',
                        help='fold of cross val: 0/1')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--if-kfold', default=False,
                        help='5 fold on training set search the optimal hyperparameter')
    parser.add_argument('--rewired', type=int, default=0, metavar='N',
                        help='the rewired parameter, 0 represents no rewired')
    parser.add_argument('--use-rewired', action='store_true', default=False, help='whether use rewired SC')
    parser.add_argument('--dataset', default='HCPYA', help='choose a dataset')


    args = parser.parse_args()
    main(args)





