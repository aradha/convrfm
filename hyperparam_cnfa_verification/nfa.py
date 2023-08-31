import torch
import torchvision
import torch.nn as nn
from torchsummary import summary

from torch.nn.functional import pad
import torch.autograd as autograd
from torch.autograd import Variable
from functorch import vmap, jacrev, jacfwd

from tqdm import tqdm 
import pickle
import os

from einops import rearrange
import trainer
from models import *
from copy import deepcopy
from math import sqrt
from utils import check_nan

def corr(A, B):
    normA = torch.nan_to_num(torch.linalg.norm(A.reshape(-1)))
    normB = torch.nan_to_num(torch.linalg.norm(B.reshape(-1)))
    print("egop norm",normA)
    print("nfm norm",normB)
    out = (torch.sum(A*B)/(normA*normB))
    return torch.nan_to_num(out).item()

def egop_on_patches(net, loader, layer_idx, patch_size, pad_type):
    print("Egop on patches")

    i=-1
    for ell, layer in enumerate(net.layers):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            i += 1

        if i==layer_idx:
            modules = [ConvPatches2d(nn.Parameter(layer.weight, requires_grad=True))]
            for m in net.layers[ell+1:]:
                modules.append(m)
            new_model = nn.Sequential(*modules)
            
            if layer_idx == 0:
                fmap_modules = None
                feature_map = nn.Identity()
            else:
                fmap_modules = net.layers[:ell]
                feature_map = nn.Sequential(*fmap_modules)
            break
    
    if fmap_modules is not None:
        assert(not isinstance(fmap_modules[-1], nn.Conv2d))
        assert(not isinstance(fmap_modules[-1], nn.Linear))

    new_model.eval()
    feature_map.eval()

    def get_jacobian(net, X):
        def fnet_single(x):
            return net(x.unsqueeze(0)).squeeze(0)
        return vmap(jacrev(fnet_single))(X) # (n, O, C, P, Q, q, q)

    pad_sz = patch_size//2
    pad_dims = (pad_sz,pad_sz,pad_sz,pad_sz)

    gop = 0
    n = 0

    feature_map.cuda()
    new_model.cuda()

    for batch_idx, batch in tqdm(enumerate(loader)):

        X_, _ = batch
        n += len(X_)
        i=-1

        X = X_.cuda()
        with torch.no_grad():
            X = feature_map(X)

        if pad_type=="zeros":
            X = pad(X,pad_dims)
        elif pad_type=="circular":
            X = pad(X,pad_dims,"circular")

        X = X.unfold(2,patch_size,1).unfold(3,patch_size,1)
        
        with torch.no_grad():
            grad = get_jacobian(new_model, X)
        del X
        
        num_classes = grad.shape[1]
        for o in range(num_classes):
            G = rearrange(grad[:,o,:,:,:,:,:], 'b c P Q w h -> (c w h) (b P Q)')
            Gt = G.transpose(0,1)
            gop += G@Gt
            del Gt

        torch.cuda.empty_cache()
    
    return (gop/n).cpu()

def get_filter(net, layer):
    layer_name = list(net.state_dict().keys())[layer]
    weights = net.state_dict()[layer_name].clone().cpu() # (C_out, C_in, q, q)
    return weights

def filter_covariances(weights):
    print("Filter covariances")
    n_filters = len(weights)
    M = weights.reshape(n_filters,-1,1)/sqrt(n_filters) # (C_out, C_in*q**2, 1)
    check_nan(M)
    M = M@M.transpose(1,2) # (C_out, C_in*q**2, C_in*q**2)
    check_nan(M)
    return torch.sum(M,dim=0) #(C_in*q**2, C_in*q**2)

def get_correlations(egop_net, filters, trainloader, patch_size, depth, pad_type):
    correlations = []
    for layer in range(depth):
        egop = egop_on_patches(egop_net, trainloader, layer, patch_size, pad_type)
        nfm = filter_covariances(filters[layer])
        correlations.append(corr(egop, nfm))
    return correlations
    
def verify_ansatz(trainset, testset, dataset, arch, patch_size, pad_type, k, epochs, lr, init, opt, num_classes):
    print("Verifying ansatz")

    num_colors, img_shape, _ =trainset[0][0].shape
    print("Num colors:",num_colors)
    print("Image width:",img_shape)
    
    if num_classes<=10:
        mb_size_egop = 64
    elif num_classes<=50:
        mb_size_egop = 32
    elif patch_size<7 or num_classes<=100:
        mb_size_egop = 32
    else:
        mb_size_egop = 4

    def get_depth(net):
        depth = 0
        for layer in net.layers:
            if isinstance(layer, nn.Conv2d):
                depth += 1
        return depth

    if arch=="vanilla":
        net = VanillaCNN(img_shape=img_shape, init=init, width=k, ps=patch_size, num_classes=num_classes, num_colors=num_colors, pad_type=pad_type)
    elif arch=="MyrtleMax":
        net = MyrtleMax(init=init, width=k, ps=patch_size, num_classes=num_classes, num_colors=num_colors, pad_type=pad_type)
    elif arch=="Myrtle5":
        net = Myrtle5(init=init, width=k, ps=patch_size, num_classes=num_classes, num_colors=num_colors, pad_type=pad_type)
    elif arch=="Myrtle7":
        net = Myrtle7(init=init, width=k, ps=patch_size, num_classes=num_classes, num_colors=num_colors, pad_type=pad_type)
    elif arch=="Myrtle10":
        net = Myrtle10(init=init, width=k, ps=patch_size, num_classes=num_classes, num_colors=num_colors, pad_type=pad_type)
    elif arch=="Resnet18":
        net = flatten_model(torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False))
    elif arch=="VGG11":
        net = flatten_model(torchvision.models.vgg11())
    
    #net.cuda()
    #print(summary(net, input_size=(3,224,224)))
    #net.cpu()

    depth = get_depth(net)
    print("depth", depth)
    
    init_filters = []
    for layer in range(depth):
        init_filters.append(deepcopy(get_filter(net, layer)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                             shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=0)
    
    #losses, accuracies = trainer.train_network(net, trainloader, testloader, num_epochs=epochs, lr=lr, opt=opt)
    
    net.eval()

    #data_path = os.environ["DATA_PATH"]
    #trained_filters = []
    #for layer in range(depth):
    #    w = get_filter(net, layer)
    #    
    #    with open(f'{data_path}weights/{dataset}_{arch}_ps_{patch_size}_layer_{layer}_epochs_{epochs}.pkl', "wb") as f:
    #        pickle.dump(w, f)

    #    #w_minus_w0 = w - init_filters[layer]
    #    #trained_filters.append(w_minus_w0)
    #    trained_filters.append(w)

    #del trainloader
    #del testloader

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mb_size_egop,
                                             shuffle=False, num_workers=0)

    #print("trained model correlations")
    #trained_correlations = get_correlations(net, trained_filters, trainloader, patch_size, depth, pad_type)
    #print()
    trained_correlations = []

    print("init model correlations")
    init_correlations = get_correlations(net, init_filters, trainloader, patch_size, depth, pad_type)
    losses = []
    accuracies = []
    
    return trained_correlations, init_correlations, losses, accuracies

