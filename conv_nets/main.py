import os
import sys
import argparse
import pickle
import csv
from copy import deepcopy
from tqdm import tqdm
import random

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n_train',type=int, default=1000)
    parser.add_argument('-n_test',type=int, default=1000)
    parser.add_argument('-width',type=int, default=256)
    parser.add_argument('-epochs',type=int, default=100)
    parser.add_argument('-class1',type=int, default=-1)
    parser.add_argument('-class2',type=int, default=-1)
    parser.add_argument('-depth',type=int, default=-1)
    parser.add_argument('-lr',type=float, default=5e-4)
    parser.add_argument('-opt',default="adam")
    parser.add_argument('-sigma',type=float, default=0.)
    parser.add_argument('-dataset',default="cifar")
    parser.add_argument('-arch',default="vanilla")
    parser.add_argument('-padding',action='store_true')
    parser.add_argument('-seed',type=int, default=0)
    args = parser.parse_args()

    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    num_colors = 3
    in_shape = (32, 32)
    if args.dataset=="cifar":
        train_X, test_X, train_y, test_y = dataset.get_cifar(args.n_train, args.n_test)
        num_classes=10
    elif args.dataset=="cifar100":
        train_X, test_X, train_y, test_y = dataset.get_cifar100(args.n_train, args.n_test)
        num_classes=100
    elif args.dataset=="svhn":
        train_X, test_X, train_y, test_y = dataset.get_svhn(args.n_train, args.n_test)
        num_classes=10
    elif args.dataset=="gtsrb":
        train_X, test_X, train_y, test_y = dataset.get_gtsrb(args.n_train, args.n_test)
        num_classes=43
    elif args.dataset=="gtsrb_binary":
        train_X, test_X, train_y, test_y = dataset.get_gtsrb_binary(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="flowers":
        train_X, test_X, train_y, test_y = dataset.get_flowers(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="celeba":
        train_X, test_X, train_y, test_y = dataset.get_celeba(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="stl10":
        train_X, test_X, train_y, test_y = dataset.get_stl10(args.n_train, args.n_test)
        num_classes=10
    elif args.dataset=="stl10_binary":
        train_X, test_X, train_y, test_y = dataset.get_stl10_binary(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="DTD":
        train_X, test_X, train_y, test_y = dataset.get_dtd(args.class1, args.class2)
        num_classes=2
    elif args.dataset == "PCAM":
        train_X, test_X, train_y, test_y = dataset.get_PCAM(args.n_train, args.n_test)
        num_classes = 2
    elif args.dataset == "FGVCAircraft":
        train_X, test_X, train_y, test_y = dataset.get_FGVCAircraft(args.n_train, args.n_test)
        num_classes = 30
    elif args.dataset == "QMNIST":
        train_X, test_X, train_y, test_y = dataset.get_QMNIST(args.n_train, args.n_test)
        num_classes = 10
    elif args.dataset == "FashionMNIST":
        train_X, test_X, train_y, test_y = dataset.get_FashionMNIST(args.n_train, args.n_test)
        num_classes = 10
    elif args.dataset == "EMNIST":
        train_X, test_X, train_y, test_y = dataset.get_EMNIST(args.n_train, args.n_test)
        num_classes = 27
        num_colors=1
        in_shape = (28, 28)
    elif args.dataset == "EMNIST_binary":
        train_X, test_X, train_y, test_y = dataset.get_EMNIST_binary(args.class1, args.class2)
        num_classes = 2
    elif args.dataset == "Food101":
        train_X, test_X, train_y, test_y = dataset.get_Food101(args.n_train, args.n_test)
        num_classes = 101
    elif args.dataset == "USPS":
        train_X, test_X, train_y, test_y = dataset.get_USPS(args.n_train, args.n_test)
        num_classes = 10
    elif args.dataset == "Caltech101":
        train_X, test_X, train_y, test_y = dataset.get_Caltech101(args.n_train, args.n_test)
        num_classes = 101
    elif args.dataset == "StanfordCars":
        train_X, test_X, train_y, test_y = dataset.get_StanfordCars(args.n_train, args.n_test)
        num_classes = 196
    elif args.dataset == "toy":
        train_X, test_X, train_y, test_y = dataset.get_toy(args.n_train, args.n_test, args.sigma)
        num_classes = 2
        num_colors=1
        in_shape=(16,32)
    elif args.dataset == "toy_mnist":
        train_X, test_X, train_y, test_y = dataset.get_toy_mnist(args.n_train, args.n_test, args.sigma)
        num_classes = 10
        num_colors=1
        in_shape=(42,42)

    #num_classes = 2
    #train_X, train_y = utils.get_cats_dogs(train_X, train_y)
    #test_X, test_y = utils.get_cats_dogs(test_X, test_y)

    print("Train shape",train_X.shape, train_y.shape)
    print("Test shape",test_X.shape, test_y.shape)

    train_X = train_X.float()
    test_X = test_X.to(train_X.dtype)
    train_y = train_y.to(train_X.dtype)
    test_y = test_y.to(train_X.dtype)

    if args.arch == "myrtle10":
        model = models.Myrtle10(args.width)
    elif args.arch == "myrtle5":
        model = models.Myrtle5(args.width)
    elif args.arch == "gap":
        model = models.CNNGAP(args.width)
    elif args.arch == "myrtleMax":
        model = models.MyrtleMax(args.width)
    elif args.arch == "vgg":
        model = models.VGG(args.width)
    elif args.arch == "simplenet":
        model = models.SimpleNet(args.width)
    elif args.arch == "vanilla":
        model = models.Vanilla(num_classes, num_colors, args.width, in_shape=in_shape, depth=args.depth, padding=args.padding)
    elif args.arch == "mlp":
        model = models.MLP(args.width)

    model.to(train_X.dtype).cuda()
    model.eval()
    
    train_loader = torch.utils.data.DataLoader(MyDataset(train_X,train_y), batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(MyDataset(test_X,test_y), batch_size=128, shuffle=False)

    for layer in range(args.depth):
        w = get_filter(model, layer)
        print(w.shape)
        
        with open(f'init_filters/{args.dataset}_classes_{args.class1}_{args.class2}_{args.arch}_depth_{args.depth}_width_{args.width}_layer_{layer}_seed_{args.seed}_sigma_{args.sigma}_padding_{args.padding}_opt_{args.opt}.pkl', "wb") as f:
            pickle.dump(w, f)

    model, acc, mse = t.train_network(model, train_loader, test_loader, num_epochs=args.epochs, lr=args.lr, opt=args.opt)

    print("Saving filters...")
    for layer in range(args.depth):
        w = get_filter(model, layer)
        print(w.shape)
        
        with open(f'trained_filters/{args.dataset}_classes_{args.class1}_{args.class2}_{args.arch}_depth_{args.depth}_width_{args.width}_layer_{layer}_seed_{args.seed}_sigma_{args.sigma}_padding_{args.padding}_opt_{args.opt}.pkl', "wb") as f:
            pickle.dump(w, f)
        
    with open(f'results/{args.dataset}_depth_{args.depth}_seed_{args.seed}_padding_{args.padding}.csv', 'a') as f:
            write = csv.writer(f)
            write.writerow(["width",args.width,"lr",args.lr,"opt",args.opt,"epochs",args.epochs])
            write.writerow(["class",args.class1, args.class2])
            write.writerow(["sigma",args.sigma])
            write.writerow([acc])
            write.writerow([mse])


