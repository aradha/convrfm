import torch
import torch.nn as nn
import torch.nn.functional as F

def get_param_count(model, ntk_layer=0):
    tot = 0
    for i, p in enumerate(model.parameters()):
        if i >= ntk_layer:
            tot += p.numel()
    return tot

class MLP(nn.Module):
    def __init__(self, width):
        super(MLP, self).__init__()
        k1=width
        k2=width
        k3=width
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(in_features=3*32*32, out_features=k1, bias=False),
            activation_fn,
            #nn.Linear(in_features=k1, out_features=k1, bias=False),
            #activation_fn,
            #nn.Linear(in_features=k1, out_features=k1, bias=False),
            #activation_fn,
            nn.Linear(in_features=k1, out_features=1, bias=False)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Vanilla(nn.Module):
    def __init__(self, num_classes, num_colors, width, in_shape=(32, 32), depth=1, padding=False):
        super(Vanilla, self).__init__()
        k=width
        P, Q = in_shape
        use_bias=False
        activation_fn = nn.ReLU()
        if padding:
            pad = 1
        else:
            pad = 0
        k1 = width

        self.layers = [
            nn.Conv2d(in_channels=num_colors, out_channels=k1, kernel_size=(3, 3), stride=1, padding=pad, bias=use_bias), 
            activation_fn
            ]

        for _ in range(1,depth):
            self.layers += [
                nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=pad, bias=False),
                activation_fn
            ]
        
        if pad==0:
            final_shape = k1 * (P-2*depth) * (Q-2*depth)
        elif pad==1:
            final_shape = k1 * P * Q
        self.layers += [
            nn.Flatten(),
            nn.Linear(in_features=final_shape, out_features=num_classes, bias=False)
        ]
        
        #for layer in self.layers:
        #    if isinstance(layer, nn.Conv2d):
        #        init = 0.2
        #        print("Using small init:",init)
        #        torch.nn.init.xavier_uniform_(layer.weight, gain=init)

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, width):
        super(SimpleNet, self).__init__()
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
                             nn.Conv2d(3, width, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(width, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
                             nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
                             nn.ReLU(inplace=True)
                        ])


        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(256, num_classes, bias=False))

        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = F.max_pool2d(x, kernel_size=x.size()[2:])
        out = self.classifier(out)
        return out


class VGG(nn.Module):
    def __init__(self, width):
        super(VGG, self).__init__()
        k1=width
        k2=2*width
        k3=4*width
        k4=8*width
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k3, out_channels=k4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k4, out_channels=k4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k4, out_channels=k4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k4, out_channels=k4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k4, out_channels=k4, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(k4, k4, bias=False),
            activation_fn,
            nn.Linear(k4, k4, bias=False),
            activation_fn,
            nn.Linear(k4, 1, bias=False)
        ])
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MyrtleMax(nn.Module):
    def __init__(self, width):
        super(MyrtleMax, self).__init__()
        k1=width
        k2=width*2
        k3=width*4
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=k2, out_features=k2, bias=False),
            activation_fn,
            nn.Linear(in_features=k2, out_features=1, bias=False)
        ])
        #for layer in self.layers:
        #    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #        k = len(layer.weight)
        #        W_std = (2.0/k)**0.5
        #        nn.init.normal_(layer.weight, mean=0.0, std=W_std)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Myrtle5(nn.Module):
    def __init__(self, width):
        super(Myrtle5, self).__init__()
        k1=width
        k2=2*width
        k3=4*width
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=k3, out_features=1, bias=False)
        ])

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                k = len(layer.weight)
                W_std = (0.1/k)**0.5
                nn.init.normal_(layer.weight, mean=0.0, std=W_std)

    def forward(self, x):
        x = x
        for layer in self.layers:
            x = layer(x)
        return x

class Myrtle10(nn.Module):
    def __init__(self, width):
        super(Myrtle10, self).__init__()
        k1=width
        k2=width
        k3=width
        num_classes = 1
        
        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k1, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k2, out_channels=k2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=k2, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k3, out_channels=k3, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=k3, out_features=1, bias=False)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CNNGAP(nn.Module):
    def __init__(self, width):
        super(CNNGAP, self).__init__()
        k1=width
        k2=width
        k3=width
        num_classes = 1

        activation_fn = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.Conv2d(in_channels=k1, out_channels=k1, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            activation_fn,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=k1, out_features=1, bias=False)
        ])

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                k = len(layer.weight)
                W_std = (1/k)**0.5
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
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
