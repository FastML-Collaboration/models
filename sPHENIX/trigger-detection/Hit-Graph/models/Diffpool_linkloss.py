from builtins import print
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.vp_GNN import GNN
from models.encoders import SoftPoolingGcnEncoder
from .utils import make_mlp

class Diffpool_linkloss(nn.Module):
    def __init__(self, hidden_dim, hidden_activation='Tanh', layer_norm=True, GNN_config={}, learning_rate=0.001, lr_scheduler_decrease_rate=0.95, diff_pool_config={}):
        """
        SetToGraph model.
        :param in_features: input set's number of features per data point
        :param out_features: number of output features.
        :param set_fn_feats: list of number of features for the output of each deepsets layer
        :param method: transformer method - quad, lin2 or lin5
        :param hidden_mlp: list[int], number of features in hidden layers mlp.
        :param predict_diagonal: Bool. True to predict the diagonal (diagonal needs a separate psi function).
        :param attention: Bool. Use attention in DeepSets
        :param cfg: configurations of using second bias in DeepSetLayer, normalization method and aggregation for lin5.
        """
        super(Diffpool_linkloss, self).__init__()

        self.name = 'Diffpool_linkloss'

        #input model
        self.input_network = make_mlp(4, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)

        self.loss_func = nn.BCELoss()

        # ip pred diffpool model
        self.ip_pred_diffpool = SoftPoolingGcnEncoder(input_dim=hidden_dim, **diff_pool_config)

        # optimizer init
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        # lr_scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=lr_scheduler_decrease_rate)
        
                
    def forward(self, x, edge_index, batch, batch_size, e=None, is_e=False):
        node_output = self.input_network(x)
        # print(node_output)
        n_hits = x.shape[0] // batch_size
        nodes = node_output.view(batch_size.item(), n_hits, node_output.shape[1])
        A = torch.cuda.FloatTensor(batch_size, n_hits, n_hits).fill_(0)
        A[:, np.arange(n_hits), np.arange(n_hits)] = 1
        start, end = edge_index
        if is_e:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = e.reshape(-1)
        else:
            A[batch[start], (start-n_hits*batch[start]), (end-n_hits*batch[start])] = 1
        ip_pred = self.ip_pred_diffpool(nodes, A, batch_num_nodes=None)
        return ip_pred, A

    def train_model(self, ip_pred, ip_true, A):
        self.optimizer.zero_grad()
        # ip_pred = ip_pred.view(-1, 1)
        # ip_true = ip_true.view(-1, 1)
        loss = self.get_loss(ip_pred, ip_true, A)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_loss(self, ip_pred, ip_true, A):
        sigmoid = nn.Sigmoid()
        return  self.ip_pred_diffpool.loss(pred=ip_pred, label=ip_true, adj=A, type='binary')
        # return self.loss_func(sigmoid(ip_pred), ip_true)