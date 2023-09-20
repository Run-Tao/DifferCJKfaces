from torchvision.datasets import MNIST
from torchvision.transforms import Normalize,Compose,ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as f
def get_dataloader(train=True):
    transform_fn = Compose([ToTensor(),Normalize(mean=(0.1307,),std=(0.3081,))])
    dataset = MNIST(root="./data",train=True,transform=transform_fn)
    data_loader=DataLoader(dataset,batch_size=2,shuffle=True)
    return data_loader

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1=nn.Linear(1*28*28,28)
    
    def forward(self, input):
        x=input.view([input.size(0),1*28*28])
        x=self.fc1(x)
        x=f.relu(x)
        out = self.fc2(x)
        return out