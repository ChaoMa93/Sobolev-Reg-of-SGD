import math
import numpy as np
import torch
from .models.vgg import vgg11, vgg11_big
from .models.fnn import fnn
from .models.resnet import resnet
from .data import load_fmnist,load_cifar10, load_1dfcn
from .trainer import accuracy
from .linalg import eigen_variance, eigen_hessian



def load_net(network, dataset, num_classes):
    if network == 'fnn':
        return fnn(dataset, num_classes)
    elif network == 'vgg':
        if dataset == 'fashionmnist':
            in_channel = 1
        else:
            in_channel = 3
        return vgg11(num_classes, in_channel)
    elif network == 'resnet':
        return resnet(num_classes=num_classes)  # only support cifar, b/c in_channels
    else:
        raise ValueError('Network %s is not supported'%(network))


def load_data(dataset, num_classes, train_per_class, batch_size):
    if dataset == 'fashionmnist':
        return load_fmnist(num_classes, train_per_class, batch_size)
    elif dataset == 'cifar10':
        return load_cifar10(num_classes, train_per_class, batch_size)
    elif dataset == '1dfunction':
        return load_1dfcn(train_per_class, batch_size)
    else:
        raise ValueError('Dataset %s is not supported'%(dataset))


def get_sharpness(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_hessian(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    return v


def get_nonuniformity(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    v = eigen_variance(net, criterion, dataloader, \
                      n_iters=n_iters, tol=tol, verbose=verbose)
    return math.sqrt(v)


def eval_accuracy(model, criterion, dataloader):
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0

    loss_t, acc_t = 0.0, 0.0
    for i in range(n_batchs):
        inputs,targets = next(dataloader)
        # inputs, targets = inputs.cuda(), targets.cuda()

        logits = model(inputs)
        loss_t += criterion(logits,targets).item()
        acc_t += accuracy(logits.data,targets)

    return loss_t/n_batchs, acc_t/n_batchs
    
    
def eval_output(model, dataloader):
    '''
    batch_size is assumed to be 1
    '''
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0
    X = np.zeros((n_batchs))
    Y = np.zeros((n_batchs))
    for i in range(n_batchs):
        x, _ = next(dataloader)
        y = model(x)
        X[i] = x.item()
        Y[i] = y.item()
    return X, Y


def get_gradW(model, dataloader, ndata, k):
    '''
    batch size of dataloader is assumed to be 1
    '''
    gradW = np.zeros((ndata, model.num_classes))
    for i in range(ndata):
        X, y = next(dataloader)
        logits = model(X)
        for j in range(model.num_classes):
            logit = logits[0][j]
            model.zero_grad()
            logit.backward(retain_graph=True)
            
            grad = [p.grad.detach().numpy() for p in model.parameters()]
            grad = [np.reshape(g, (-1)) for g in grad]
            grad = np.concatenate(grad)
            gradW[i,j] = np.sum(grad**(2*k))
    return (np.sum(gradW) / ndata) ** (1./2/k)


def get_gradx(model, dataloader, ndata, k):
    '''
    get gradient with respect to x
    '''
    gradx = np.zeros((ndata, model.num_classes))
    for i in range(ndata):
        X, y = next(dataloader)
        X.requires_grad = True
        logits = model(X)
        for j in range(model.num_classes):
            logit = logits[0][j]
            model.zero_grad()
            grad = torch.autograd.grad(logit, X, retain_graph=True)[0]
            grad = grad.detach().numpy()
            grad = np.reshape(grad, (-1))
            gradx[i,j] = np.sum(grad**(2*k))
    return (np.sum(gradx) / ndata) ** (1./2/k)
    


