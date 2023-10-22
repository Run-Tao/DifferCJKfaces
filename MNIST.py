from torchvision import transforms
import numpy as np
import torch 
import torchvision
import torch.nn.functional as F
from torch import nn

train_batch_size = 64
test_batch_size = 1000
img_size = 28

def get_dataloader(train=True):
    assert isinstance(train,bool)

    dataset = torchvision.datasets.MNIST('/data',train=train,download=True,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,),(0.3081,)),]))
    
    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28*1,28)
        self.fc2 = torch.nn.Linear(28,10)

    def forward(self,x):
        x = x.view(-1,28*28*1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.log_softmax(x,dim=-1)
    
mnist_net = MnistNet()
optimizer = torch.optim.Adam(mnist_net.parameters(),lr=0.001)
train_loss_list = []
train_count_list = []

def train(epoch):
    mnist_net.train(True)
    train_dataloader = get_dataloader(True)
    for idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = mnist_net(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if idx%100 == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss:{:.6f}'.format(epoch, idx*len(data)/len(train_dataloader),loss.item()))
            train_loss_list.append(loss.item())
            train_count_list.append(idx*train_batch_size+(epoch-1)*len(train_dataloader))

epoch = 25
for i in range(epoch):
    train(i)