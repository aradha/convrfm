import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FlatModel(nn.Module):
    def __init__(self, layers):
        super(FlatModel, self).__init__()

        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def flatten_model(model):
    model_type = type(model)

    modules = []
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, model_type):
            modules.append(module)

    flat_model = FlatModel(nn.ModuleList(modules))
    return flat_model

class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, X):
        # X: (n, C, P, Q)
        return X.unfold(2,self.patch_size,1).unfold(3,self.patch_size,1)

class ConvPatches2d(nn.Module):
    def __init__(self, filters):
        # filters: (C_out, C_in, q, q)
        super().__init__()
        self.weight = filters 

    def forward(self, X):
        # X: (n, C, P, Q, q, q)
        n, _, P, Q, _, _ = X.shape
        X_ = rearrange(X, 'n C P Q w h -> (n P Q) (C w h)')
        W_ = rearrange(self.weight, 'c C w h -> (C w h) c')
        prod = X_@W_ # (n P Q) c
        return rearrange(prod, '(n P Q) c -> n c P Q', n=n, P=P, Q=Q) 

class MyrtleMax(nn.Module):
    def __init__(self, init, width, ps, num_classes, num_colors, pad_type):
        super(MyrtleMax, self).__init__()
        k1=width
        k2=width

        pad_sz = ps//2
        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            #nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            #nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(k2, num_classes, bias=False)
        ])

        i=-1
        for layer in self.layers:
            print(f'Layer: {layer}')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                i = i + 1
                if i==0:
                    W_std = init
                else:
                    W_std = 1e-2
                nn.init.normal_(layer.weight, mean=0.0, std=W_std)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Myrtle5(nn.Module):
    def __init__(self, init, width, ps, num_classes, num_colors, pad_type):
        super(Myrtle5, self).__init__()
        k1=width
        k2=width
        k3=width
        
        pad_sz = ps//2

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(k2, num_classes, bias=False)
        ])

        i=-1
        for layer in self.layers:
            print(f'Layer: {layer}')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                i = i + 1
                if i==0:
                    W_std = init
                else:
                    W_std = 1e-2
                nn.init.normal_(layer.weight, mean=0.0, std=W_std)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Myrtle10(nn.Module):
    def __init__(self, width, ps, num_classes, num_colors, pad_type):
        super(Myrtle10, self).__init__()
        k1=width
        k2=width
        k3=width
        
        pad_sz = ps//2

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k1),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k1),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k2),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k2),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k2),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k3, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k3),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k3),
            activation_fn,
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k3),
            activation_fn,
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            #nn.BatchNorm2d(k3),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(k3, num_classes, bias=False)
        ])

        i=-1
        for layer in self.layers:
            print(f'Layer: {layer}')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                i = i + 1
                if i==1:
                    W_std = 5e-3
                else:
                    W_std = 5e-3
                nn.init.normal_(layer.weight, mean=0.0, std=W_std)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Myrtle7(nn.Module):
    def __init__(self, width, ps, num_classes, num_colors, pad_type):
        super(Myrtle7, self).__init__()
        k1=width
        k2=width
        k3=width
        
        pad_sz = ps//2

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False, padding_mode=pad_type),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(k2, num_classes, bias=False)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VanillaCNN(nn.Module):
    def __init__(self, img_shape, init, width, ps, num_classes, num_colors, pad_type):
        super(VanillaCNN, self).__init__()
        k1=width
        k2=width
        k3=width

        pad_sz = ps//2

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(ps, ps), stride=1, padding=pad_sz, bias=False),
            activation_fn,
            nn.Flatten(),
            nn.Linear(img_shape**2*k2, num_classes, bias=False)
        ])

        i=-1
        for layer in self.layers:
            print(f'Layer: {layer}')
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                i = i + 1
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    W_std = init
                else:
                    W_std = 1e-2
                nn.init.normal_(layer.weight, mean=0.0, std=W_std)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
