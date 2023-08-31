import os
import sys
import argparse
import pickle as p
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.multiprocessing as mp

import utils
import trainer
import nfa
from datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_train',type=int, default=2000)
    parser.add_argument('-n_test',type=int, default=2000)
    parser.add_argument('-epochs',type=int, default=1000)
    parser.add_argument('-patch_size',type=int, default=3)
    parser.add_argument('-k',type=int, default=64)
    parser.add_argument('-lr',type=float, default=1e-3)
    parser.add_argument('-init',type=float, default=5e-3)
    parser.add_argument('-dataset',default="cifar10")
    parser.add_argument('-opt',default="adam")
    parser.add_argument('-arch',default="vanilla")
    parser.add_argument('-pad_type',default="zeros")
    parser.add_argument('-warm_init', action='store_true')
    parser.add_argument('-raw_M', action='store_true')
    parser.add_argument('-pre_trained_M', action='store_true')
    parser.add_argument('-whiten_patches', action='store_true')
    parser.add_argument('-whiten_image', action='store_true')
    parser.add_argument('-full_train',action='store_true')
    parser.add_argument('-full_test',action='store_true')
    args = parser.parse_args()
    print(sys.argv)

    if args.dataset == "cifar10":
        trainset, testset = get_cifar_data(args.n_train, args.n_test, args.full_train, args.full_test, args.whiten_patches)
        num_classes = 10
    elif args.dataset == "ImageNet":
        trainset, testset = get_ImageNet_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 1000
    elif args.dataset == "PCAM":
        trainset, testset = get_PCAM_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 2
    elif args.dataset == "FGVCAircraft":
        trainset, testset = get_FGVCAircraft_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 30
    elif args.dataset == "bars":
        trainset, testset = get_toy_data(args.n_train, args.n_test)
        num_classes = 2
    elif args.dataset == "stl10":
        trainset, testset = get_stl10_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "gtsrb":
        trainset, testset = get_gtsrb_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 43
    elif args.dataset == "DTD":
        trainset, testset = get_dtd_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 47
    elif args.dataset == "svhn":
        trainset, testset = get_svhn_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "MNIST":
        trainset, testset = get_MNIST_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "EMNIST":
        trainset, testset = get_EMNIST_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 27
    elif args.dataset == "QMNIST":
        trainset, testset = get_QMNIST_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "Food101":
        trainset, testset = get_Food101_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 101
    elif args.dataset == "Caltech101":
        trainset, testset = get_Caltech101_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 101
    elif args.dataset == "StanfordCars":
        trainset, testset = get_StanfordCars_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 196
    elif args.dataset == "EuroSAT":
        trainset, testset = get_EuroSAT_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "USPS":
        trainset, testset = get_USPS_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "SEMEION":
        trainset, testset = get_SEMEION_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 10
    elif args.dataset == "RenderedSST2":
        trainset, testset = get_RenderedSST2_data(args.n_train, args.n_test, args.full_train, args.full_test)
        num_classes = 2
    
    print(f'Train len: {len(trainset)}, shape: {trainset[0][0].shape}')
    print(f'Test len: {len(testset)}, shape: {testset[0][0].shape}')

    trained_corrs, init_corrs, losses, accs = nfa.verify_ansatz(trainset, testset, args.dataset, args.arch,
                                                args.patch_size, args.pad_type, args.k, args.epochs, 
                                                args.lr, args.init, args.opt, num_classes)

    print("Trained correlations by layer:",trained_corrs)
    print("Init correlations by layer:",init_corrs)
    
    import csv
    with open(f'correlations/init_{args.dataset}_{args.arch}_{args.patch_size}_epochs_{args.epochs}.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(trained_corrs)
        write.writerow(init_corrs)
        write.writerow(["train_loss","test_loss"])
        write.writerow(losses)
        write.writerow(["train_acc","test_acc"])
        write.writerow(accs)

    print("Write successful.")

