from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import SGC_plus

import random

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
            
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='reddit',
                    help='dataset to be used')

parser.add_argument('--order', type=int, default=4,
                    help='num of hops we use')


parser.add_argument('--descending', type=float, default=2.0,
                    help='To control descending rate')

parser.add_argument('--tmp', type=float, default=0.5,
                    help='tmp')

parser.add_argument('--neibor_ratio', type=float, default=1.0,
                    help='neibor_ratio')

parser.add_argument('--plus', type=int, default=1,
                    help='use plus or not')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


from utils import load_citation, sgc_precompute, set_seed, load_reddit_data
adj, train_adj, features, labels, idx_train, idx_val, idx_test = load_reddit_data('AugNormAdj', cuda=args.cuda)

from sklearn.metrics import f1_score
def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro


# Model and optimizer
model = SGC_plus(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            adj=adj,
            order=args.order,
            features=features,
            args= args,
            )




optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train_setting(model):
    for name, param in model.named_parameters():
        if "mask" in name and param.is_leaf:
            param.requires_grad = False
        else:
            param.requires_grad = True



## for val
def val_setting_1(model):
    for name, param in model.named_parameters():
        if "mask" in name and param.is_leaf:
            param.requires_grad = True
        else:
            param.requires_grad = False




def train():
    
    model.train()
    train_setting(model)
    optimizer.zero_grad()
    output= model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    return 

def val():
    model.train()
    val_setting_1(model)
    optimizer.zero_grad()
    output= model(features)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # loss_train = F.nll_loss(output[idx_val], labels[idx_val])
    loss_train.backward()
    optimizer.step()
    return 




def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = f1(output[idx_test], labels[idx_test])
    acc_val = f1(output[idx_val], labels[idx_val])

    return acc_test, acc_val


t_total = time.time()


passes = 1
test_acc_pass = 0

time_train = 0
time_test = 0

for i in range(passes):
    best_val_acc = test_acc = 0
    best_accu = 0
    for epoch in range(args.epochs - 1):


        if args.plus:
            val()
        time0 = time.time()
        train()
        time_train += time.time()-time0

        time0 = time.time()
        tmp_test_acc, val_acc = test()
        time_test += time.time() - time0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            epoch_best = epoch


    test_acc_pass += test_acc/passes



file = open(r"log_best.txt", encoding="utf-8",mode="a+")  
with open("log_best.txt", encoding="utf-8",mode="a+") as data:  
    print(args.lr, args.weight_decay, args.order, args.data, args.epochs, args.tmp, args.plus, test_acc_pass.item(), time_train, time_test, file=data)


