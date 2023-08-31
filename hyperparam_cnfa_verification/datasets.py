import os
import sys
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.feature_extraction import image

from math import sqrt
import pickle
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def whiten_images(X_, verbose=True, patch_size=3):
    '''X_: images, shape (num_images, num_channels, h, w).'''
    X = torch.moveaxis(X_, 1, -1).numpy()
    h, w, c = X.shape[1:]
    for idx in range(X.shape[0]):
        if verbose and idx % 1000 == 0:
            print(idx)
        im = X[idx]
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        if p.ndim < 4:
            p = p[:,:,:,None]
        p -= p.mean((1,2))[:,None,None,:]
        im = image.reconstruct_from_patches_2d(p, (h, w, c))
        p = image.extract_patches_2d(im, (patch_size, patch_size))
        p = p.reshape(p.shape[0], -1)

        cov = p.T.dot(p)
        s, U = np.linalg.eigh(cov)
        s[s <= 0] = 0
        s = np.sqrt(s)
        ind = s < 1e-8 * s.max()
        s[ind == False] = 1. / np.sqrt(s[ind == False])
        s[ind] = 0

        p = p.dot(U.dot(np.diag(s)).dot(U.T))
        p = p.reshape(p.shape[0], patch_size, patch_size, -1)
        X[idx] = image.reconstruct_from_patches_2d(p, (h, w, c))
    return torch.moveaxis(torch.from_numpy(X), -1, 1)

def pre_process(torchset, n_samples, full=False, num_classes=10, channels=3):
    n = len(torchset)
    indices = list(np.random.choice(n, min(n,n_samples)))

    if full:
        indices = np.arange(n)

    trainset = []
    print(f'{channels} expected')
    for ix in tqdm(indices):
        x,y = torchset[ix]

        if x.shape[0] != channels:
            continue

        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        trainset.append((x,ohe_y))
    return trainset

def process_for_kernels(torchset):
    Xs=[]
    ys=[]
    for x,y in torchset:
        Xs.append(x)
        ys.append(y)
    return torch.stack(Xs), torch.stack(ys)

def get_toy_data(N_train=200, N_test=200, small_blocks=True):
    print("Getting toy bars dataset")
    stride=1
    dim=16
    c=1
    w = len(range(0,dim,stride))
    num_classes = 2
    N = (N_train + N_test)//2
    if small_blocks:
        Xs = []
        ys = []
        block_dims = [(3,7),(7,3)]
        for y0, (w, h) in enumerate(block_dims):
            X = torch.zeros((N,c,dim,dim))
            center_i = torch.randint(0,dim-w+1,size=(N,))
            center_j = torch.randint(0,dim-h+1,size=(N,))
            
            X += np.random.normal(scale=1.5,size=(N,c,dim,dim))
            
            for z, (i,j) in enumerate(list(zip(center_i,center_j))):
                X[z,:,i:i+w,j:j+h] = 1
                
            y = torch.zeros(N,num_classes)
            y[:,y0] = 1
            
            Xs.append(X)
            ys.append(y)
            
        X = torch.concatenate(Xs)
        y = torch.concatenate(ys)
    else: 
        X = torch.zeros((2*N,1,dim,dim))
        y = torch.zeros(2*N,num_classes)
        X += np.random.normal(scale=3,size=(2*N,1,dim,dim))
        
        center_i = torch.randint(0,dim,size=(2*N,))
        for n, i in enumerate(center_i):
            if n < N:
                X[n,:,i,:] = 1
                y[:,0] = 1 
            else:
                X[n,:,:,i] = 1
                y[:,1] = 1 

    idx = np.random.permutation(2*N)
    train_idx = idx[:N_train]
    test_idx = idx[N_train:]

    train_X = X[train_idx]
    train_y = y[train_idx]
    test_X = X[test_idx]
    test_y = y[test_idx]

    return train_X, test_X, train_y, test_y

def get_dtd_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting DTD dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'dtd/'
    trainset0 = torchvision.datasets.DTD(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.DTD(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=47)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=47)

    return trainset, testset

def get_cifar_data(n_train=1000, n_test=1000, full_train=False, full_test=False, whitened=False, patch_size=3):
    print("Getting CIFAR dataset")

    data_path = os.environ["DATA_PATH"] + 'cifar10/'
    if whitened:
        train_X = torch.load(data_path + f'cifar10_white{patch_size}_xtrain.pt')
        test_X = torch.load(data_path + f'cifar10_white{patch_size}_xtest.pt')
        train_y = torch.load(data_path + f'train_y.pt')
        test_y = torch.load(data_path + f'test_y.pt')
        
        if not full_train:
            indices = list(np.random.choice(len(train_X), n_train))
            train_X = train_X[indices]
            train_y = train_y[indices]
        if not full_test:
            indices = list(np.random.choice(len(test_X), n_test))
            test_X = test_X[indices]
            test_y = test_y[indices]

    else:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transform = transforms.Compose(
            [ transforms.ToTensor(),
             transforms.Normalize(mean, std)])

        trainset0 = torchvision.datasets.CIFAR10(root=data_path,
                                            train = True,
                                            transform=transform,
                                            download=True)
        testset0 = torchvision.datasets.CIFAR10(root=data_path,
                                            train = False,
                                            transform=transform,
                                            download=True)

        trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10)
        testset = pre_process(testset0, n_test, full=full_test, num_classes=10)

    return trainset, testset 

def get_ImageNet_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting ImageNet dataset")

    data_path = os.environ['DATA_PATH']
    data_folder = data_path + '/ImageNet'

    def load_databatch(idx, img_size=32, test=False):
        
        if test:
            data_file = os.path.join(data_folder, 'val_data')
            d = unpickle(data_file)
        else:
            data_file = os.path.join(data_folder, 'train_data_batch_')
            d = unpickle(data_file + str(idx))

        x = d['data']
        y = d['labels']

        x = x/np.float32(255)
        #mean_image = mean_image/np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]

        #x -= mean_image

        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        return torch.from_numpy(x), torch.tensor(y)
    
    X = []
    Y = []
    for idx in range(1,11):
        x, y = load_databatch(idx)
        X.append(x)
        Y.append(y)

    X = torch.concat(X,dim=0)
    Y = torch.concat(Y,dim=0)
    trainset0 = list(zip(X,Y))

    Xtest, Ytest = load_databatch(None, test=True)
    testset0 = list(zip(Xtest,Ytest))

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=1000)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=1000)
    return trainset, testset 

def get_FGVCAircraft_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting FGVCAircraft dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'FGVCAircraft/'
    trainset0 = torchvision.datasets.FGVCAircraft(root=data_path,
                                        split  = "train",
                                        annotation_level='manufacturer',
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.FGVCAircraft(root=data_path,
                                        split = "test",
                                        annotation_level='manufacturer',
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=30)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=30)
    return trainset, testset 

def get_PCAM_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting PCAM dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'PCAM/'
    trainset0 = torchvision.datasets.PCAM(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.PCAM(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=2)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=2)
    return trainset, testset 

def get_stl10_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting STL10 dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'STL10/'
    trainset0 = torchvision.datasets.STL10(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.STL10(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10)
    return trainset, testset 

def get_gtsrb_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting GTSRB dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'GTSRB/'
    trainset0 = torchvision.datasets.GTSRB(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.GTSRB(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=43)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=43)
    return trainset, testset 

def get_svhn_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting SVHN dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'SVHN/'
    trainset0 = torchvision.datasets.SVHN(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.SVHN(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    C = 256
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10)
    return trainset, testset 

def get_EuroSAT_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting EuroSAT dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'EuroSAT/'
    trainset0 = torchvision.datasets.EuroSAT(root=data_path,
                                        transform=transform,
                                        download=True)
    
    n_tot = len(trainset0)
    n_train = int(0.8*n_tot)
    trainset0 = torch.utils.data.Subset(trainset0, list(range(n_train)))
    testset0 = torch.utils.data.Subset(trainset0, list(range(n_train,n_tot)))

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10)
    return trainset, testset 

def get_USPS_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting USPS dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'USPS/'
    trainset0 = torchvision.datasets.USPS(root=data_path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.USPS(root=data_path,
                                        train = False,
                                        transform=transform,
                                        download=True)
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10, channels=1)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10, channels=1)
    return trainset, testset 


def get_StanfordCars_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting StanfordCars dataset")
    transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
    ])

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'StanfordCars/'
    trainset0 = torchvision.datasets.StanfordCars(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.StanfordCars(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=196)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=196)
    return trainset, testset 

def get_MNIST_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting MNIST dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'MNIST/'
    trainset0 = torchvision.datasets.MNIST(root=data_path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.MNIST(root=data_path,
                                        train = False,
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10)

    return trainset, testset 

def get_QMNIST_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting QMNIST dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'QMNIST/'
    trainset0 = torchvision.datasets.QMNIST(root=data_path,
                                        what='train',
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.QMNIST(root=data_path,
                                        what='test',
                                        transform=transform,
                                        download=True)
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10, channels=1)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10, channels=1)

    return trainset, testset 

def get_EMNIST_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting EMNIST dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'EMNIST/'
    trainset0 = torchvision.datasets.EMNIST(root=data_path,
                                        split="letters",
                                        transform=transform,
                                        download=True)
    n_tot = len(trainset0)
    n_train = int(0.8*n_tot)
    n_test = n_tot-n_train
    trainset0, testset0 = torch.utils.data.random_split(trainset0, [n_train, n_test])
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=27, channels=1)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=27, channels=1)

    return trainset, testset 

def get_Caltech101_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting Caltech101 dataset")
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]       
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose(
        [ 
            transforms.Resize(size=(32,32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'Caltech101/'
    trainset0 = torchvision.datasets.Caltech101(root=data_path,
                                        transform=transform,
                                        download=True)

    n_tot = len(trainset0)
    n_train = int(0.8*n_tot)
    n_test = n_tot-n_train
    trainset0, testset0 = torch.utils.data.random_split(trainset0, [n_train, n_test])
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=101, channels=3)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=101, channels=3)

    return trainset, testset 


def get_RenderedSST2_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting RenderedSST2 dataset")
    transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
    ])

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'RenderedSST2/'

    trainset0 = torchvision.datasets.RenderedSST2(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.RenderedSST2(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=2)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=2)
    return trainset, testset 


def get_Food101_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting Food101 dataset")
    transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
    ])

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'Food101/'

    trainset0 = torchvision.datasets.Food101(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset0 = torchvision.datasets.Food101(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=101)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=101)
    return trainset, testset 

def get_SEMEION_data(n_train=1000, n_test=1000, full_train=False, full_test=False):
    print("Getting SEMEION dataset")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'SEMEION/'

    trainset0 = torchvision.datasets.SEMEION(root=data_path,
                                        transform=transform,
                                        download=True)
    
    n_tot = len(trainset0)
    n_train = int(0.8*n_tot)
    n_test = n_tot-n_train
    trainset0, testset0 = torch.utils.data.random_split(trainset0, [n_train, n_test])
    trainset = pre_process(trainset0, n_train, full=full_train, num_classes=10, channels=1)
    testset = pre_process(testset0, n_test, full=full_test, num_classes=10, channels=1)

    return trainset, testset 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default="cifar10")
    parser.add_argument('-patch_size', type=int, default=3)
    args = parser.parse_args()
    patch_size = args.patch_size
    verbose = 1
    dataset = args.dataset
    data_dir = os.environ["DATA_PATH"] + f'{dataset}/'
    print(f'Whitening {dataset}')

    if args.dataset == "cifar10":
        train_X, test_X, train_y, test_y = get_cifar_data()
    elif args.dataset == "bars":
        train_X, test_X, train_y, test_y = get_toy_data()
    elif args.dataset == "stl10":
        train_X, test_X, train_y, test_y = get_stl10_data()
    elif args.dataset == "gtsrb":
        train_X, test_X, train_y, test_y = get_gtsrb_data()
    elif args.dataset == "svhn":
        train_X, test_X, train_y, test_y = get_svhn_data()

    print('whitening Xtrain...')
    train_X = whiten_images(train_X, verbose=verbose, patch_size=patch_size)
    print('saving train...')
    torch.save(train_X, os.path.join(data_dir, f'{dataset}_white{patch_size}_xtrain.pt'))
    torch.save(train_y, os.path.join(data_dir, f'train_y.pt'))
    print('whitening Xtest...')
    test_X = whiten_images(test_X, verbose=verbose, patch_size=patch_size)
    print('saving test...')
    torch.save(test_X, os.path.join(data_dir, f'{dataset}_white{patch_size}_xtest.pt'))
    torch.save(test_y, os.path.join(data_dir, f'test_y.pt'))


def get_zca_matrix(X_):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X : [n x C x P x Q]
    OUTPUT: ZCAMatrix: [d x d] matrix
    """
    
    n, C, P, Q = X_.shape
    eps = 0.1*50000/n

    d = C*P*Q
    
    X = X_.reshape(n,d).double().cuda()
    
    D = torch.zeros(d).cuda()

    _, D[:min(n,d)], UT = torch.linalg.svd(X/sqrt(n))

    D[D < 0] = 0
    D = D**2
    D = D + eps*torch.sum(D)/d

    D = 1/D.sqrt()
    D[min(n,d):] = 0

    W = (UT.T*D) @ UT

    return W.float().cpu()

def zca_whiten(X, ZCA_mat):
    n, C, P, Q = X.shape
    #print(f'ZCA shape {ZCA_mat.shape}')
    X = X.reshape(n,-1)@ZCA_mat
    return X.reshape(n, C, P, Q).float()


