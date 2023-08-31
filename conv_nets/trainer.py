import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import time

from copy import deepcopy

criterion = torch.nn.MSELoss()

def train_network(net, train_loader, test_loader, num_epochs=400, lr=1e-3, opt="adam",
                    kernel_size=3, in_channels=3, binary=False):

    params = 0
    for i, param in enumerate(list(net.parameters())):
        size = 1
        for j in range(len(param.size())):
            size *= param.size()[j]
            params += size

    print("NUMBER OF PARAMS: ", params)
    
    if opt=="adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt=="sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif opt=="sgd_wd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5)


    net.cuda()
    best_acc = 0
    best_model = None
    best_loss = None

    for i in range(num_epochs):
        #print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)

        #if i%1==0:
        test_loss = val_step(net, test_loader)
        train_acc = get_acc(net, train_loader, binary)
        test_acc = get_acc(net, test_loader, binary)

        if train_loss < 1e-15:
            break
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = deepcopy(net.cpu())
            best_loss = test_loss
            net.cuda()


        print("Epoch: ", i,
              "Train Loss: ", train_loss, "Test Loss: ", test_loss,
              "Train Acc: ", train_acc, "Test Acc: ", test_acc,
              "Best Test Acc: ", best_acc)

    return net.cpu(), best_acc, best_loss


def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)
    
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs).cuda()).float()
        target = Variable(targets).cuda().float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        loss = criterion(output, target)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader, binary):
    net.eval()
    count = 0
    n = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        if binary:
            labels = torch.round(output)
            target = torch.round(target)
        else:
            labels = torch.argmax(output, axis=1)
            target = torch.argmax(target, axis=1)
        count += torch.sum(labels == target).cpu().data.numpy()
        n += len(labels)
    return count / n * 100
