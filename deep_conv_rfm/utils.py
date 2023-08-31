import jax.numpy as jnp
import torch
from einops import rearrange

import torch
import numpy as np
import torch.nn as nn

import torch.nn.functional as F


def multiply_patches(X, M, ps):
    """
    X : (n, p*ps, q*ps, c)
    M : (c*w*h, c*w*h)
    out : (n, p*ps, q*ps, c)
    """
    n = X.shape[0]
    chunk = 5000
    leftover_bool = int(n%chunk>0)
    batches = jnp.array_split(jnp.arange(n), n//chunk + leftover_bool)

    Xs = []
    for i, b in enumerate(batches):
        Xb = X[b]
        m, P, Q, c = Xb.shape
        p = P//ps
        q = Q//ps
        Xb = rearrange(Xb, 'm (p w) (q h) c -> (m p q) (c w h)', p=p, q=q, w=ps, h=ps)
        Xb = Xb @ M
        Xb = rearrange(Xb, '(m p q) (c w h) -> m (p w) (q h) c', m=m, p=p, q=q, c=c, w=ps, h=ps)
        Xs.append(jnp.array(Xb))
    return jnp.concatenate(Xs, axis=0)

def matrix_sqrt(M, thresh=False):
    S, V = torch.linalg.eigh(M.cuda())
    S[S<0] = 0

    if thresh:
        k = int(3*len(S)//4)
        print("Thresh k:",k)
        S[:k] = 0

    S = torch.diag(S**0.5)
    return (V @ S @ V.T).cpu()

def get_batch_size(n1, n2, num_devices):
    n1_ = n1//num_devices
    max_batch_size = 100
    best_batch_size = 1
    for i in range(1,max_batch_size+1):
        if (n1_%i == 0) and (n2%i==0):
            best_batch_size = i
    return best_batch_size


# get_layer_from_M(num_colors, n_filters=args.width, M=current_M)
def get_layer_from_M(num_colors, n_filters, M, padding=False, ps=3, sampleM=False):
    
    if padding:
        pad = 1
    else:
        pad = 0
    
    layer = nn.Conv2d(in_channels=num_colors, out_channels=n_filters, kernel_size=(3,3), padding=pad, bias=False)

    if M is None:
        return layer

    weights = layer.weight.detach()

    C_out, _, _, _ = weights.shape
    Mn = weights.reshape(C_out,-1)
    Mn = Mn.T@Mn
    var = np.trace(Mn)
    
    if sampleM:
        M_ = M / np.trace(M)
        new_weights = torch.from_numpy(np.random.multivariate_normal(np.zeros(len(M)), M_, size=n_filters)).float()
    else:
        sqrtM = matrix_sqrt(M)
        new_weights = np.random.normal(size=(n_filters, num_colors*ps*ps))
        new_weights = torch.from_numpy(new_weights).float()
        new_weights = new_weights @ sqrtM

    new_weights = new_weights.reshape(-1,num_colors,ps,ps)
    layer.weight = nn.Parameter(new_weights, requires_grad=False)

    return layer

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

def apply_M(X, sqrtM, ps):
    if sqrtM is not None:
        X_M = multiply_patches(X, sqrtM, ps)
    else:
        X_M = X
    return X_M



