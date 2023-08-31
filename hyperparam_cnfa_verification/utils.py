import os
import sys
import torch

import numpy as np
import jax.numpy as jnp

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

def check_nan(A):
    assert(torch.sum(torch.isnan(A))==0)
    assert(torch.sum(torch.isinf(A))==0)
    return


def get_pretrained_params(params, dataset="cifar10", model_name="vgg19", covariance=False):
        # filters are (64, 3, 3, 3) 
        n_filters = params[0][0].shape[-1]
        new_params = list(params)

        if "laplace":
            M = torch.load(f'./filters/{dataset}_3_Ms.pt')
            M = torch.moveaxis(M,2,1)
            M = M.reshape(27,3,9)
            M = M.reshape(27,27).numpy()

            filters = np.random.multivariate_normal(mean=np.zeros(27,), cov=M, size=n_filters)
            filters = torch.from_numpy(filters.reshape(n_filters,3,3,3))

            filters = torch.moveaxis(filters,1,-1)
            filters = torch.moveaxis(filters,0,-1)


            bias = params[0][1]
            new_params[0] = (jnp.array(filters), bias)
            
            print("new filters", filters.shape)
            return tuple(new_params)

        elif model_name=="vgg19":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).cpu()
        elif model_name=="vgg11":
            model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).cpu()
        elif model_name=="wide_resnet":
            model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)

        for i, model_params in enumerate(model.parameters()):
            if i==0:
                filters = model_params.clone().detach().cpu()
            break

        del model

        print("Filters",filters.shape)
        n_model_filters = filters.shape[0]

        if covariance:
            filters = filters.reshape(n_model_filters,-1,1)
            M = filters@filters.transpose(1,2)
            M = torch.mean(M, dim=0).numpy()
            print("M norm", np.linalg.norm(M.reshape(-1)))
            
            # normalize
            old_filters = np.array(params[0][0]) # (3,3,c,n)
            old_filters = np.moveaxis(old_filters.reshape(1,27,n_filters),-1,0)
            print("old_filters",old_filters.shape)

            ggT = np.mean(np.swapaxes(old_filters,1,-1)@old_filters,axis=0)
            old_norm = np.linalg.norm(ggT.reshape(-1))
            print("Old norm", old_norm)

            #M /= np.linalg.norm(M.reshape(-1))
            #M *= old_norm 
            
            filters = np.random.multivariate_normal(mean=np.zeros(27,), cov=M, size=n_filters)
            filters = torch.from_numpy(filters.reshape(n_filters,3,3,3))

        filters = torch.moveaxis(filters,1,-1)
        filters = torch.moveaxis(filters,0,-1)


        bias = params[0][1]
        new_params[0] = (jnp.array(filters), bias)
        
        print("new filters", filters.shape)
        return tuple(new_params)

def sample_from_covariance(n_filters):
    M = M = torch.load(f'./filters/cifar10_3_Ms.pt')
    M = torch.moveaxis(M,2,1)
    M = M.reshape(27,3,9)
    M = M.reshape(27,27).numpy().astype('float64')

    filters = np.random.multivariate_normal(mean=np.zeros(len(M),), cov=M, size=n_filters)
    filters = torch.from_numpy(filters.reshape(n_filters,3,3,3))
    return filters.double()
