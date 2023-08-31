import os
import sys
import argparse
import pickle
from copy import deepcopy
import csv
from tqdm import tqdm

import random
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics.functional import accuracy, mean_squared_error as mse

import utils
import dataset
import models
import trainer as t

def get_filter(net, layer):
    layer_name = list(net.state_dict().keys())[layer]
    weights = net.state_dict()[layer_name].clone().cpu() # (C_out, C_in, q, q)
    return weights

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y 

    def __len__(self):
            return len(self.y)

    def __getitem__(self, idx):
            X_mb = self.X[idx]
            y_mb = self.y[idx]
            return (X_mb, y_mb)

def get_classes(X_full, y_full, c1, c2):
    y_full_ = torch.argmax(y_full,dim=1)
    c1_idx = (y_full_ == c1)
    c2_idx = (y_full_ == c2)

    X = torch.concat((X_full[c1_idx], X_full[c2_idx]))
    y = torch.concat((y_full_[c1_idx], y_full_[c2_idx]))
    y[y==c1] = 1
    y[y==c2] = 0

    return X, y.unsqueeze(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_train',type=int, default=1000)
    parser.add_argument('-n_test',type=int, default=1000)
    parser.add_argument('-width',type=int, default=256)
    parser.add_argument('-epochs',type=int, default=100)
    parser.add_argument('-depth',type=int, default=1)
    parser.add_argument('-seed',type=int, default=0)
    parser.add_argument('-lr',type=float, default=5e-4)
    parser.add_argument('-opt',default="adam")
    parser.add_argument('-dataset',default="cifar")
    parser.add_argument('-arch',default="vanilla")
    parser.add_argument('-padding',action='store_true')
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    num_colors = 3
    in_shape = (32, 32)

    if args.dataset=="gtsrb":
        train_X_full, test_X_full, train_y_full, test_y_full = dataset.get_gtsrb(n_train=1e6, n_test=1e6)
        num_classes=43
    elif args.dataset=="flowers":
        train_X_full, test_X_full, train_y_full, test_y_full = dataset.get_flowers(get_all=True)
        num_classes=20
    elif args.dataset=="DTD":
        train_X_full, test_X_full, train_y_full, test_y_full = dataset.get_dtd(get_all=True)
        num_classes=20
    elif args.dataset=="food":
        train_X_full, test_X_full, train_y_full, test_y_full = dataset.get_food(n_train=1e6, n_test=1e6)
        num_classes=20
    
    train_X_full = train_X_full.float()
    train_y_full = train_y_full.float()
    test_X_full = test_X_full.float()
    test_y_full= test_y_full.float()

    for c1 in range(num_classes-1):
        for c2 in range(c1+1, num_classes):
            
            train_X, train_y = get_classes(train_X_full, train_y_full, c1, c2)
            test_X, test_y = get_classes(test_X_full, test_y_full, c1, c2)

            train_loader = torch.utils.data.DataLoader(MyDataset(train_X,train_y), batch_size=128, shuffle=False)
            test_loader = torch.utils.data.DataLoader(MyDataset(test_X,test_y), batch_size=128, shuffle=False)

            model = models.Vanilla(1, num_colors, args.width, in_shape=in_shape, depth=args.depth, padding=args.padding)
            model.to(train_X.dtype)
            model.cuda()
            model.eval()
            model, acc, mse = t.train_network(model, train_loader, test_loader, num_epochs=args.epochs, lr=args.lr, opt=args.opt, binary=True)
            
            with open(f'results/{args.dataset}_depth_{args.depth}_seed_{args.seed}_padding_{args.padding}.csv', 'a') as f:
                write = csv.writer(f)
                write.writerow(["class",c1, c2])
                write.writerow([acc])
                write.writerow([mse])
