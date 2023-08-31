import torch
import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt
import neural_tangents.stax as stax

from tqdm import tqdm
import time
import argparse
import csv
import os
import pickle
from copy import deepcopy

import dataset
import models
import utils
import grads

import torch.nn.functional as F
from einops import rearrange

def expand_image(X, ps, padding=False):
    """
    X : (n, c, p, q)
    out : (n, c, p*ps, q*ps)
    """
    
    if padding:
        pad_sz = ps//2
        pad = (pad_sz,pad_sz,pad_sz,pad_sz)
        X_patched = F.pad(X, pad)
    else:
        X_patched = X

    X_patched = X_patched.unfold(2,ps,1).unfold(3,ps,1) # (n, c, p, q, ps, ps)

    n, c, p, q, _, _ = X_patched.shape
    X_patched = X_patched.transpose(-2,-3) # (n, c, p, ps, q, ps)
    X_expanded = X_patched.reshape(n,c,p*ps,q*ps)
    return X_expanded

def batch_kernel(kernel, X, Z, sqrtM, args):

    nx = len(X)
    nz = len(Z)
    chunk = 1000
    xbatches = torch.split(torch.arange(nx), chunk)
    zbatches = torch.split(torch.arange(nz), chunk)

    num_devices = jax.device_count()
    K = [None]*len(xbatches)

    for i, bx in tqdm(enumerate(xbatches)):
        Xb_small = X[bx]
        Xb = utils.expand_image(Xb_small, args.patch_size, args.padding)
        Xb = jnp.array(torch.moveaxis(Xb,1,-1).numpy())
        Xb = utils.apply_M(Xb, sqrtM, args.patch_size)
        if i==0:
            print("Xb",Xb.shape)

        Kx = [None]*len(zbatches)
        for j, bz in enumerate(zbatches):


            BS = utils.get_batch_size(len(bx), len(bz), num_devices)
            ntk_fn = nt.batch(kernel,
                               device_count=-1,
                               batch_size=BS,
                               store_on_device=False)

            Zb_small = Z[bz]
            Zb = utils.expand_image(Zb_small, args.patch_size, args.padding)
            Zb = jnp.array(torch.moveaxis(Zb,1,-1).numpy())
            Zb = utils.apply_M(Zb, sqrtM, args.patch_size)

            Kx[j] = ntk_fn(Xb, Zb).ntk

        K[i] = np.concatenate(Kx,axis=1)

    return np.array(np.concatenate(K,axis=0))

def main(args):
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

    if args.dataset=="cifar":
        train_X, test_X, train_y, test_y = dataset.get_cifar(args.n_train, args.n_test)
        num_classes=10
    if args.dataset=="cifar100":
        train_X, test_X, train_y, test_y = dataset.get_cifar100(args.n_train, args.n_test)
        num_classes=100
    elif args.dataset=="toy":
        train_X, test_X, train_y, test_y = dataset.get_toy(args.n_train, args.n_test, args.sigma)
        num_classes=2
    elif args.dataset=="toy_mnist":
        train_X, test_X, train_y, test_y = dataset.get_toy_mnist(args.n_train, args.n_test, args.sigma)
        num_classes=10
    elif args.dataset=="svhn":
        train_X, test_X, train_y, test_y = dataset.get_svhn(args.n_train, args.n_test)
        num_classes=10
    elif args.dataset=="gtsrb":
        train_X, test_X, train_y, test_y = dataset.get_gtsrb(args.n_train, args.n_test)
        num_classes=43
    elif args.dataset=="stl10":
        train_X, test_X, train_y, test_y = dataset.get_stl10(args.n_train, args.n_test)
        num_classes=10
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
    elif args.dataset=="gtsrb_binary":
        train_X, test_X, train_y, test_y = dataset.get_gtsrb_binary(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="flowers":
        train_X, test_X, train_y, test_y = dataset.get_flowers(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="celeba":
        train_X, test_X, train_y, test_y = dataset.get_celeba(args.class1, args.class2)
        num_classes=2
    elif args.dataset == "EMNIST_binary":
        train_X, test_X, train_y, test_y = dataset.get_EMNIST_binary(args.class1, args.class2)
        num_classes = 2
    elif args.dataset=="stl10_binary":
        train_X, test_X, train_y, test_y = dataset.get_stl10_binary(args.class1, args.class2)
        num_classes=2
    elif args.dataset=="DTD":
        train_X, test_X, train_y, test_y = dataset.get_dtd(args.class1, args.class2)
        num_classes=2

    n = len(train_X)
    ntest = len(test_X)
    print("Num train:",n)
    print("Num test:",ntest)
    
    dtype = jnp.float32
    y_train = train_y.numpy()
    y_test = test_y.numpy()
    
    
    deep_accs = []
    deep_mses = []
    for ell in range(args.depth):
        
        print()
        print("Feature depth:",ell)
        _, _, kernel_fn = models.Vanilla(ps=args.patch_size, depth=args.depth-ell, expanded=True)

        if ell>0:
            print("Featurizing")
            num_colors = train_X.shape[1]

            layer = utils.get_layer_from_M(num_colors, n_filters=args.width, M=best_M, padding=args.padding, sampleM=args.sampleM)
            featurize = lambda x: layer(x).detach()

            train_X = featurize(train_X)
            test_X = featurize(test_X)

        data_path = os.path.join(os.environ["DATA_PATH"], f'embeddings/{args.dataset}/X_{ell}_sigma_{args.sigma}_thresh_{args.thresh}_padding_{args.padding}_sqrtM_{args.sqrtM}_sampleM_{args.sampleM}.pt')
        print("data",os.environ["DATA_PATH"])
        print("path",data_path)
        torch.save(train_X[:10].cpu(), data_path)


        accs = []
        mses = []
        best_M = None
        next_M = None
        next_sqrtM = None
        best_iter_acc = 0
        for t in range(args.num_iters):
            

            start = time.time()
            Ktrain = batch_kernel(kernel_fn, train_X, train_X, next_sqrtM, args)
            end = time.time()
            print(f'Ktrain time {end-start}')

            start = time.time()
            Ktest = batch_kernel(kernel_fn, test_X, train_X, next_sqrtM, args)
            end = time.time()
            print(f'Ktest time {end-start}')

            scale = Ktrain.max()
            Ktrain /= scale
            Ktest /= scale

            best_reg = None
            best_acc = 0
            best_mse = None
            
            regs = [1e-8, 1e-5, 1e-3]
            print("regs",regs)
            for reg in regs:
                start = time.time()
                alphas=np.linalg.solve(reg*np.eye(n, dtype=np.float32)+Ktrain, y_train.astype(np.float32))
                end = time.time()
                print(f'Solve time {end-start}')

                ypred = Ktest@alphas
                acc = np.mean(1.0*(np.argmax(ypred,axis=1) == np.argmax(y_test,axis=1))) * 100
                mse = np.mean(np.sum((ypred - y_test)**2,axis=1))

                if acc>best_acc:
                    best_acc = acc
                    best_mse = mse
                    best_reg = reg

            accs.append(best_acc)
            mses.append(best_mse)
            print(f'Round {t} accuracy:', best_acc)
            print(f'Round {t} MSE:', best_mse)
            print(f'Round {t} Reg:', best_reg)
            print()

            if best_acc > best_iter_acc:
                best_M = deepcopy(next_M)
                best_iter_acc = best_acc

            # no gradient on last iter
            if t == args.num_iters-1:
                continue

            s = 20
            Xs = train_X[:s] 
            next_M = grads.get_grads(kernel_fn, jnp.array(alphas), train_X, Xs, next_sqrtM, num_classes, args)

            if args.sqrtM:
                next_sqrtM = utils.matrix_sqrt(next_M, args.thresh)
                next_sqrtM = jnp.array(next_sqrtM.numpy())
            else:
                next_sqrtM = jnp.array(next_M.numpy())
                next_M = next_M@next_M

            next_M = jnp.array(next_M)
            next_M = next_M/np.linalg.norm(next_M)
            next_sqrtM = next_sqrtM/np.linalg.norm(next_sqrtM)

        if best_M is not None:
            best_M = torch.from_numpy(np.array(best_M))
            M_path = os.environ["DATA_PATH"] + f'deepMs/{args.dataset}_classes_{args.class1}_{args.class2}_sigma_{args.sigma}_n_{n}_kernel_depth_{args.depth}_feature_depth_{ell}_width_{args.width}_bestM_sqrt_{args.sqrtM}_thresh_{args.thresh}_padding_{args.padding}_sampleM_{args.sampleM}.pt'
            torch.save(best_M, M_path)

        deep_accs.append(accs)
        deep_mses.append(mses)
    
    
    results_file = "results/"
    if "toy" in args.dataset:
        results_file += f'{args.dataset}_n_{n}_depth_{args.depth}_width_{args.width}_sqrt_{args.sqrtM}_sigma_{args.sigma}_thresh_{args.thresh}_padding_{args.padding}_sampleM_{args.sampleM}.csv'
    else:
        results_file += f'{args.dataset}_n_{n}_depth_{args.depth}_width_{args.width}_sqrt_{args.sqrtM}_thresh_{args.thresh}_padding_{args.padding}_sampleM_{args.sampleM}.csv'

    with open(results_file, "w") as f:
        write = csv.writer(f)
        for ell in range(len(deep_accs)):
            write.writerow(["feature depth",ell])
            write.writerow(deep_accs[ell])
            write.writerow(deep_mses[ell])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_train',type=int, default=100)
    parser.add_argument('-n_test',type=int, default=100)
    parser.add_argument('-patch_size',type=int, default=3)
    parser.add_argument('-depth',type=int, default=5)
    parser.add_argument('-num_iters',type=int, default=5)
    parser.add_argument('-width',type=int, default=64)
    parser.add_argument('-reg',type=float, default=1e-4)
    parser.add_argument('-sigma',type=float, default=0.)
    parser.add_argument('-dataset',default="cifar")
    parser.add_argument('-sqrtM',action='store_true')
    parser.add_argument('-thresh',action='store_true')
    parser.add_argument('-padding',action='store_true')
    parser.add_argument('-sampleM',action='store_true')
    parser.add_argument('-class1',type=int, default=-1)
    parser.add_argument('-class2',type=int, default=-1)

    args = parser.parse_args()
    
    main(args)
