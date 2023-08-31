import os
import sys
import argparse
import pickle
import csv
from copy import deepcopy
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import models
import dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-width',type=int, default=64)
    parser.add_argument('-seed',type=int, default=0)
    parser.add_argument('-dataset',default="svhn")
    parser.add_argument('-opt',default="sgd")
    parser.add_argument('-padding',action='store_true')
    args = parser.parse_args() 
    
    n=10
    if args.dataset=="svhn":
        sigma = 0.0
        n=40
        X, _, _, _ = dataset.get_svhn(n, 1)
    elif args.dataset=="toy_mnist":
        sigma = 2.0
        X, _, _, _ = dataset.get_toy_mnist(20000, 10000, sigma)
    elif args.dataset=="toy":
        sigma = 0.1
        X, _, _, _ = dataset.get_toy(500, 500, sigma)
    elif args.dataset=="gtsrb":
        sigma = 0.0
        X, _, _, _ = dataset.get_gtsrb(10, 1)
    elif args.dataset=="EMNIST":
        sigma = 0.0
        X, _, _, _ = dataset.get_EMNIST(10, 1)
    elif args.dataset=="cifar":
        sigma = 0.0
        X, _, _, _ = dataset.get_cifar(10, 1)
    
    X = X[:n]
    depth = 3
    for layer in range(depth):
        print("layer:",layer)
        print("X:",X.shape)

        emb_path = os.path.join(os.environ["DATA_PATH"], "nn_embeddings", args.dataset, f'X_nn_depth_{layer}_sigma_{sigma}_padding_{args.padding}.pt') 
        torch.save(X, emb_path)

        weight_path = os.path.join("trained_filters", 
                        f'{args.dataset}_classes_-1_-1_vanilla_depth_3_width_{args.width}_layer_{layer}_seed_{args.seed}_sigma_{sigma}_padding_{args.padding}_opt_{args.opt}.pkl')
        
        with open(weight_path, "rb") as f:
            w = pickle.load(f)
        k, c, q, _ = w.shape
        conv_layer = nn.Conv2d(in_channels=c, out_channels=args.width, kernel_size=3)
        conv_layer.weight = nn.Parameter(w)
        conv_layer = nn.Sequential(conv_layer, nn.ReLU())
        X = conv_layer(X).detach()
        


