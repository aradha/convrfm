import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.functional import fold

import os
import matplotlib.pyplot as plt

def matrix_sqrt(M):
    S, V = torch.linalg.eigh(M)
    S[S<0] = 0
    S = torch.diag(S**0.5)
    return (V @ S @ V.T).cpu()

def corr(A_, B_):
    A = A_.reshape(-1)
    B = B_.reshape(-1)

    return (torch.dot(A,B) / A.norm() / B.norm()).item()

def scale(x_):
    #x = torch.abs(x_)
    x = x_
    return (x-x.min())/(x.max() - x.min())

def expand_image(X, ps=3, pad_mode="circular"):
    """
    X : (n, c, p, q)
    out : (n, c, p*ps, q*ps)
    """

    n, c, p, q = X.shape

    pad_sz = ps//2
    if pad_mode=="zero":
        pad = (pad_sz,pad_sz,pad_sz,pad_sz)
        X_patched = F.pad(X, pad)
    elif pad_mode=="circular":
        X_patched = torch.from_numpy(np.pad(X, ((0,0),(0,0),(pad_sz,pad_sz),(pad_sz,pad_sz)), mode='wrap'))

    X_patched = X_patched.unfold(2,ps,1).unfold(3,ps,1) # (n, c, p, q, ps, ps)
    X_patched = X_patched.transpose(-2,-3) # (n, c, p, ps, q, ps)
    X_expanded = X_patched.reshape(n,c,p*ps,q*ps)
    return X_expanded

def reduce_image(X, depth, ps=3):
    """
    X : (n, c, p*ps, q*ps)
    out : (n, c, p, q)
    """
    n, c, P, Q = X.shape
    p = P//ps
    q = Q//ps

    X = X.reshape(n, c, p, ps, q, ps)
    if depth == 0:
        return X.norm(dim=(3,5))
    else:
        X = X.norm(dim=(3,5))
        #X = torch.max(X, dim=1)[0]
        return X
        #X = X**2 
        #X = X.sum(dim=(1,3,5))
        #return X.sqrt()

    #X = torch.permute(X, (0, 1, 3, 5, 2, 4))
    #X = X.reshape(n, c*ps*ps, p*q)
    #pad_sz = ps//2
    #folded = fold(X, output_size=(p, q), kernel_size=(ps, ps), padding=(pad_sz, pad_sz))

    #ones = torch.ones(X.shape)
    #ones = fold(ones, output_size=(p, q), kernel_size=(ps, ps), padding=(pad_sz, pad_sz))
    #return folded/ones

def multiply_patches(X, M_, ps=3):
    """
    X : (n, c, p*ps, q*ps)
    M_ : (c*w*h, c*w*h)
    out : (n, c, p*ps, q*ps)
    """
    n = X.shape[0]
    chunk = 5000
    leftover_bool = int(n%chunk>0)
    batches = np.array_split(np.arange(n), n//chunk + leftover_bool)

    M = matrix_sqrt(M_)

    Xs = []
    for i, b in enumerate(batches):
        Xb = X[b]
        m, c, P, Q = Xb.shape
        p = P//ps
        q = Q//ps
        Xb = rearrange(Xb, 'm c (p w) (q h) -> (m p q) (c w h)', p=p, q=q, w=ps, h=ps)
        Xb = Xb @ M
        Xb = rearrange(Xb, '(m p q) (c w h) -> m c (p w) (q h)', m=m, p=p, q=q, c=c, w=ps, h=ps)
        Xs.append(Xb)
    return torch.cat(Xs, dim=0)


if __name__ == "__main__":
    embeddings = []
    for j in range(7,8):
        print("Image",j)
        for i in range(8):
            emb = torch.load(f'embeddings/X_{i}.pt').detach() # (n, c, P, Q)
            if i==0:
                plt.imshow(torch.moveaxis(emb[j],0,-1))
                plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
                plt.savefig(f'figures/{j}_original_X.pdf',format="pdf")
                #plt.show()
                plt.close()
            
            X = expand_image(emb[j:j+1])
            X = reduce_image(X, depth=i)
            if i>0:
                plt.imshow(scale(X.sum(dim=1).squeeze(0)))
                #plt.imshow(scale(X.squeeze(0)))
            else:   
                plt.imshow(scale(torch.moveaxis(X,1,-1).squeeze(0)))

            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.savefig(f'figures/{j}_X_layer_{i}.pdf',format="pdf")
            plt.title("Before filter")
            plt.close()
            
            filter = torch.load(f'filters/w_{i}.pt').detach() # (c_out, c_in, p, q)
            c_out, c_in, p, q = filter.shape
            M = filter.reshape(c_out, -1)
            M = M.T@M
            plt.imshow(scale(M),cmap="gray")
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.savefig(f'figures/filter_M_{i}.pdf',format="pdf")
            plt.close()

            #_, U = torch.linalg.eigh(M)
            #print(U.shape)
            #print(filter.shape)
            #for k in range(8):
            #    uk = U[:,-k-1].unsqueeze(1)
            #    u = uk.reshape(c_in,p,q)
            #    u = torch.moveaxis(u,0,-1)
            #    if c_in==3:
            #        plt.imshow(scale(u))
            #    else:
            #        plt.imshow(scale(u.sum(-1)))
            #    plt.xticks([], [])
            #    plt.yticks([], [])
            #    plt.savefig(f'vgg_eigvs/u_{i}_{k}.pdf',format="pdf")
            #    plt.close()

            X = expand_image(emb[j:j+1])
            X = multiply_patches(X, M)
            X = reduce_image(X, depth=i)

            if i>0:
                plt.imshow(scale(X.sum(dim=1).squeeze(0)))
                #plt.imshow(scale(X.squeeze(0)))
            else:   
                plt.imshow(scale(torch.moveaxis(X,1,-1).squeeze(0)))
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.savefig(f'figures/{j}_X_layer_{i}_M_filtered.pdf',format="pdf")
            plt.close()

            X = expand_image(emb[j:j+1])
            egop = torch.load(f'egops/egop_{i}.pt')
            plt.imshow(scale(egop),cmap="gray")
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            #plt.title("AGOP")
            plt.savefig(f'figures/agop_{i}.pdf',format="pdf")
            plt.close()

            X = multiply_patches(X, egop)
            X = reduce_image(X, depth=i)
            if i>0:
                plt.imshow(scale(X.sum(dim=1).squeeze(0)))
                #plt.imshow(scale(X.squeeze(0)))
            else:   
                plt.imshow(scale(torch.moveaxis(X,1,-1).squeeze(0)))
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

            print("Ansatz correlation:",corr(egop,M))

            plt.savefig(f'figures/{j}_X_layer_{i}_agop_filtered.pdf',format="pdf")
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.close()
