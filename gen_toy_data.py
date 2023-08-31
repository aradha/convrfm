import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import random
import os

from math import log, sqrt

def one_hot_data(dataset, num_classes, num_samples):
    Xs = []
    ys = []

    for ix in range(min(len(dataset),num_samples)):
        X,y = dataset[ix]
        Xs.append(X)

        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        ys.append(ohe_y)

    return torch.stack(Xs), torch.stack(ys)

def gen_toy_mnist_backgrounds(n_train, n_test):

    NUM_CLASSES = 10
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "MNIST/"

    trainset = torchvision.datasets.MNIST(root=path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.MNIST(root=path,
                                        train = False,
                                        transform=transform,
                                        download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    def get_noise_frame(X):
        n, c, p, q = X.shape
        P, Q = 42, 42

        s1s = torch.randint(high=(P-p), size=(n,))
        s2s = torch.randint(high=(Q-q), size=(n,))

        X_frame = torch.randn(size=(n,c,P,Q))

        return X_frame, s1s, s2s
    
    train_frame, train_s1s, train_s2s = get_noise_frame(train_X)
    test_frame, test_s1s, test_s2s = get_noise_frame(test_X)

    path = os.environ["DATA_PATH"]

    torch.save(train_X, os.path.join(path, f'toy_mnist/train_X_n_{n_train}.pt'))
    torch.save(train_y, os.path.join(path, f'toy_mnist/train_y_n_{n_train}.pt'))
    torch.save(train_frame, os.path.join(path, f'toy_mnist/train_frames_n_{n_train}.pt'))
    torch.save(train_s1s, os.path.join(path, f'toy_mnist/train_frame_s1s_n_{n_train}.pt'))
    torch.save(train_s2s, os.path.join(path, f'toy_mnist/train_frame_s2s_n_{n_train}.pt'))

    torch.save(test_X, os.path.join(path, f'toy_mnist/test_X_n_{n_test}.pt'))
    torch.save(test_y, os.path.join(path, f'toy_mnist/test_y_n_{n_test}.pt'))
    torch.save(test_frame, os.path.join(path, f'toy_mnist/test_frames_n_{n_test}.pt'))
    torch.save(test_s1s, os.path.join(path, f'toy_mnist/test_frame_s1s_n_{n_test}.pt'))
    torch.save(test_s2s, os.path.join(path, f'toy_mnist/test_frame_s2s_n_{n_test}.pt'))

    return 


def gen_toy_backgrounds(n_train, n_test):
    d = 32
    k = 16
    wstar = torch.ones((d,))
    wstar /= torch.linalg.norm(wstar)

    train_X = torch.randn(size=(n_train, 1, k, d))
    test_X = torch.randn(size=(n_test, 1, k, d))
    train_y = torch.randint(0,2, size=(n_train,))*2 - 1
    test_y = torch.randint(0,2, size=(n_test,))*2 - 1

    wstar_train = torch.randint(low=0,high=k,size=(n_train,))
    wstar_test = torch.randint(low=0,high=k,size=(n_test,))

    path = os.environ["DATA_PATH"]
    torch.save(train_X, os.path.join(path, f'toy/train_frames_n_{n_train}.pt'))
    torch.save(train_y, os.path.join(path, f'toy/train_y_n_{n_train}.pt'))
    torch.save(test_X, os.path.join(path, f'toy/test_frames_n_{n_test}.pt'))
    torch.save(test_y, os.path.join(path, f'toy/test_y_n_{n_test}.pt'))
    torch.save(wstar_train, os.path.join(path, f'toy/train_wstar_n_{n_train}.pt'))
    torch.save(wstar_test, os.path.join(path, f'toy/test_wstar_n_{n_test}.pt'))

    return

if __name__=="__main__":

    #gen_toy_backgrounds(500, 500)
    gen_toy_mnist_backgrounds(2000, 2000)

