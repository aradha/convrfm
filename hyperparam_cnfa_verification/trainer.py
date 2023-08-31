import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import time

from copy import deepcopy

criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.MSELoss()

def train_network(net, train_loader, test_loader, num_epochs=100, lr=5e-3, opt="adam"):

    params = 0
    depth = len(list(net.parameters()))
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    if opt=="sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    elif opt=="momentum":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif opt=="adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt=="adamW":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    net.cuda()
    #best_loss = float("inf")
    best_acc = -1*float("inf")
    best_state_dict = None

    for i in range(num_epochs):
        #print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)

        if train_loss< 1e-5:
            break

        if i%5==0:
            test_loss = val_step(net, test_loader)
            train_acc = get_acc(net, train_loader)
            test_acc = get_acc(net, test_loader)

            if test_acc > best_acc:
                best_acc = test_acc 
                net.cpu()
                best_state_dict = deepcopy(net.state_dict())
                net.cuda()

            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Train Acc: ", train_acc, "Test Acc: ", test_acc,
                  "Best Test Acc: ", best_acc)

    net.cpu()
    net.load_state_dict(best_state_dict)
    return [train_loss, test_loss], [train_acc, test_acc]


def train_step(net, optimizer, train_loader):
    
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(inputs.cuda())
        target = targets.cuda()

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


def get_acc(net, loader):
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        labels = torch.argmax(output, dim=-1)
        target= torch.argmax(target, dim=-1)
        count += torch.sum(labels == target).cpu().data.numpy()
    return count / len(loader.dataset) * 100
