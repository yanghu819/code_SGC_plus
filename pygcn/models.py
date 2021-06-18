import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_nf
import torch
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module




class a_L(nn.Module):
    def __init__(self, order, args):
        super(a_L, self).__init__()
        self.a_data = torch.nn.Parameter(torch.ones(order)) 
        self.mask = torch.nn.Parameter(torch.ones(order)) 
        self.args = args
        self.tmp = torch.nn.Parameter(torch.ones([1])) 

    def forward(self, G_list):

        mask_softmax = F.softmax(self.mask * self.tmp, dim = 0)


        a_data_mask = mask_softmax * (self.a_data)

        
        G_all = a_data_mask[0] * G_list[0]
        for i in range(1, len(G_list)):
            G_all = G_all + a_data_mask[i] * G_list[i]


        return G_all




class GraphConvolution_ac(Module):

    def __init__(self, n_in, n_out, adj, args, order = 64, bias=True):
        super(GraphConvolution_ac, self).__init__()

        self.args = args
        self.order = order
        self.adj = adj
        self.a_L1 = a_L(order=self.order, args=self.args)
        self.gc = GraphConvolution_nf(n_in, n_out, bias=True)

    def forward(self, G_list):

        if self.args.plus:
            G_list = [self.gc(G_list[i]) for i in range(self.order)]
            G_all = self.a_L1(G_list)
        else:
            G_all = self.gc(G_list[-1])

        return G_all










class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj, args,  features, order = 4):
        super(GCN, self).__init__()
        
        self.args = args
        self.order = order
        self.adj = adj
        self.gc1 = GraphConvolution_ac(n_in=nfeat, n_out=nclass, adj=self.adj, order=self.order, bias=True, args=self.args)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.adj.shape[1],1)

        G_list = [features]
        for i in range(self.order - 1):
            features = torch.spmm(self.adj, features)
            G_list.append(features)
        self.G_list  = G_list


    def forward(self, x):

        # x = self.gc1(x)   
        x = self.gc1(self.G_list)  
    
        return F.log_softmax(x, dim=1)

