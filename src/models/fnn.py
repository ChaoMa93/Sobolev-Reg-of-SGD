import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1) # 28-5+1=24
        self.conv2 = nn.Conv2d(6,16,5,stride=1) # 12-5+1=8
        self.fc1 = nn.Linear(4*4*16,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        if x.ndimension()==3:
            x = x.unsqueeze(0)
        o = F.relu(self.conv1(x))
        o = F.avg_pool2d(o,2,2)

        o = F.relu(self.conv2(o))
        o = F.avg_pool2d(o,2,2)

        o = o.view(o.shape[0],-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

class FNN(nn.Module):
    def __init__(self, dataset, num_classes):
        super(FNN,self).__init__()
        self.num_classes = num_classes
        if dataset == 'fashionmnist':
            self.net = nn.Sequential(nn.Linear(784,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,num_classes))
        elif dataset == 'cifar10':
            self.net = nn.Sequential(nn.Linear(3072,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,num_classes))
        elif dataset == '1dfunction':
            self.net = nn.Sequential(nn.Linear(1,20),
                            nn.ReLU(),
                            nn.Linear(20,20),
                            nn.ReLU(),
                            nn.Linear(20,1))

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        o = self.net(x)
        return o


def lenet():
    return LeNet()

def fnn(dataset, num_classes):
    return FNN(dataset, num_classes)

