import os
import sys
import time

import neural_tangents as nt

import torch
import numpy as np
import jax 
from jax import random
import jax.numpy as jnp
from jax import grad, jit, vmap, jacrev
from einops import rearrange

from torch.nn.functional import pad as torch_pad
import torchvision
import torchvision.transforms as transforms
from math import sqrt, pi
from tqdm import tqdm
import time

import utils

def get_grads(kernel_fn, alphas, train_X, Xs, sqrtM, num_classes, args):
    
    print("Grad input shapes",train_X.shape, Xs.shape)

    ps = args.patch_size
    Xs = utils.expand_image(Xs, ps, args.padding)
    Xs = jnp.array(torch.moveaxis(Xs,1,-1).numpy())

    def get_solo_grads(sol, X, x):
        X_M = utils.expand_image(X, ps, args.padding)
        X_M = jnp.array(torch.moveaxis(X_M,1,-1).numpy())
        X_M = utils.apply_M(X_M, sqrtM, ps)


        def egop_fn(z):
            if sqrtM is not None:
                z_ = utils.multiply_patches(z, sqrtM, ps)
            else:
                z_ = z
            K = kernel_fn(X_M, z_, fn='ntk').ntk
            return (sol @ K).squeeze()
        grads = jax.vmap(jax.grad(egop_fn))(jnp.expand_dims(x,1)).squeeze()
        grads = jnp.nan_to_num(grads)
        return grads 

    n, c, p, q = train_X.shape
    w, h = (ps,ps)
    s = len(Xs)
    
    if not args.padding:
        p = p - ps + 1
        q = q - ps + 1

    chunk = 1000
    train_batches = torch.split(torch.arange(n), chunk)

    egop = 0
    sol = jnp.array(alphas.T)
    for o in tqdm(range(num_classes)):
        grads = 0
        for btrain in train_batches:
            grads += get_solo_grads(sol[o][jnp.array(btrain.numpy())], train_X[btrain], Xs)
        grads = grads.reshape(-1, p, w, q, h, c) # n, p, w, q, h, c
        G = torch.from_numpy(np.array(grads))
        G = torch.moveaxis(G.transpose(2,3),-1,-3)
        G = G.reshape(-1, c*w*h)
        egop += G.T @ G/s

    return egop

