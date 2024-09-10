import sys
sys.path.append("../functions")
from utils import *
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset, DataListLoader
from scipy.sparse import coo_matrix



class GNNNet(nn.Module):
    def __init__(self, layer_num=2, conv_dim=64, feature_num=400, dnn_dim=64, GNN=GCNConv):
        super(GNNNet, self).__init__()

        self.prelu = nn.PReLU()
        self.convs = nn.ModuleList()
        self.prelus = nn.ModuleList()
        self.convs.append(GNN(feature_num, conv_dim))
        for i in range(layer_num-1):
            self.convs.append(GNN(conv_dim, conv_dim))
        for i in range(layer_num):
            self.prelus.append(nn.PReLU())
        self.dnn = nn.Sequential(
            nn.Linear(conv_dim*2, dnn_dim),
            nn.PReLU(),
            nn.Linear(dnn_dim, 1))

    def forward(self, x, edge_index, edge_weight, label_edge_index):
        for (conv, prelu) in zip(self.convs, self.prelus):
            x = prelu(conv(x, edge_index, edge_weight))
        x1 = x[label_edge_index[0]]
        x2 = x[label_edge_index[1]]

        x = torch.cat((x1, x2), dim=-1)
        x = self.dnn(x)
        return x



def train_model(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_loss = 0
    train_output = []
    train_target = []
    loss_func = nn.MSELoss(reduction='mean')
    for batch_idx, data in enumerate(train_loader):
        sc, fc = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(sc.x, sc.edge_index, sc.edge_attr, fc.edge_index).view(-1)
        edge_label = fc.edge_attr
        loss = loss_func(output, edge_label)
        regularization_loss = torch.norm(model.dnn[0].weight, 2) + torch.norm(model.dnn[2].weight, 2)
        if args.reg>0:
            loss += args.reg * regularization_loss #  0.0001
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if epoch > args.checkepoch and epoch%args.checkperiod==0:
            train_output = np.append(train_output, np.array(output.detach().cpu().numpy()).reshape(-1))
            train_target = np.append(train_target, np.array(edge_label.detach().cpu().numpy()).reshape(-1))

    if epoch > args.checkepoch and epoch % args.checkperiod==0:
        corr = np.corrcoef(train_target, train_output)[0, 1]
        train_loss = (np.square(train_target - train_output)).mean()
        print('Train Epoch: {} [Train Average:  \tloss: {:.4f}, \tCorr: {:.6f}]'.format(epoch, train_loss, corr))


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    corr_list = []
    test_output = []
    test_target = []
    with torch.no_grad():
        for data in test_loader:
            sc, fc = data[0].to(device), data[1].to(device)
            output = model(sc.x, sc.edge_index, sc.edge_attr, fc.edge_index).view(-1)
            edge_label = fc.edge_attr
            test_loss += F.mse_loss(output, edge_label, reduction="mean").item()
            test_output.append(np.array(output.cpu().numpy()).reshape(-1))
            test_target.append(np.array(edge_label.cpu().numpy()).reshape(-1))
            corr_list.append(np.corrcoef(np.array(edge_label.cpu().numpy()).reshape(-1),
                                    np.array(output.cpu().numpy()).reshape(-1))[0,1])
    test_output = np.array(test_output)
    test_target = np.array(test_target)
    test_loss = (np.square(test_target - test_output)).mean()
    print('\nTest set: Average loss: {:.4f}, all_Corr: {:.4f}'.format(
        test_loss, np.mean(corr_list)))

    return np.mean(corr_list)




def get_test_fc_adj(model, x, y, device):
    '''
    :param model: a GNN model
    :param sc: symmetric matrix list
    :param fc: symmetric matrix list
    :param device: device
    :return: return the predicted FC list
    '''
    sc = x.copy()
    fc = y.copy()
    kwargs = {'batch_size': 1, 'num_workers': 4, 'pin_memory': True}
    loader = get_graph_data_loader(sc, fc, kwargs)
    model.eval()
    predicted_fc_list = []
    with torch.no_grad():
        for data in loader:
            sc, fc = data[0].to(device), data[1].to(device)
            output = model(sc.x, sc.edge_index, sc.edge_attr, fc.edge_index).view(-1)
            fc.edge_attr = output
            rows = np.array(fc.edge_index[0].cpu().numpy()).reshape(-1)
            cols = np.array(fc.edge_index[1].cpu().numpy()).reshape(-1)
            values = np.array(output.cpu().numpy()).reshape(-1)
            sparse_temp_graph = coo_matrix((values, (rows, cols)))
            dense_temp_graph = np.array(sparse_temp_graph.todense())
            for j in range(dense_temp_graph.shape[1]):
                dense_temp_graph[j,j] = 0
            dense_temp_graph = np.array([dense_temp_graph])
            if len(predicted_fc_list)==0:
                predicted_fc_list = dense_temp_graph
            else:
                predicted_fc_list = np.append(predicted_fc_list, dense_temp_graph, axis=0)
    return predicted_fc_list



class GetLoader(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)


def get_graph_data_loader(x, y, kwargs_):
    '''
    x,y must have same number at dimension 0
    :param x: SC, input, symmetric adjacent matrix list
    :param y: FC, label, symmetric adjacent matrix list
    :return: dataloader (dataset_x, dataset_y, graph_fea)
    '''
    dataset_x = []
    dataset_y = []
    graph_fea = []
    # take position as the node feature
    pos = np.ones(x.shape[1])
    pos = np.diag(pos)
    node_fea = torch.tensor(pos, dtype=torch.float)

    for i in range(x.shape[0]):
        temp_graph = x[i, :, :]
        sc_lookup = temp_graph
        sc_lookup = torch.tensor(sc_lookup, dtype=torch.float)
        graph_fea.append(sc_lookup)

        coo_temp_x = coo_matrix(temp_graph)
        edge_index = torch.tensor(np.vstack((coo_temp_x.row, coo_temp_x.col)), dtype=torch.long)
        values = torch.tensor(coo_temp_x.data, dtype=torch.float)
        temp_graph_tensor = Data(x=node_fea, edge_index=edge_index, edge_attr=values)
        dataset_x.append(temp_graph_tensor)

        temp_graph = y[i, :, :]
        # make zero edge become nonzeros since we need predict these edges also.
        temp_graph += 100
        for i in range(x.shape[1]):
            temp_graph[i,i] = 0
        coo_temp_x = coo_matrix(temp_graph)
        edge_index = torch.tensor(np.vstack((coo_temp_x.row, coo_temp_x.col)), dtype=torch.long)
        values = torch.tensor(coo_temp_x.data, dtype=torch.float)
        values -= 100
        temp_graph_tensor = Data(edge_index=edge_index, edge_attr=values)
        dataset_y.append(temp_graph_tensor)

    dataset = GetLoader(dataset_x, dataset_y)

    loader = DataLoader(dataset, **kwargs_)

    return loader