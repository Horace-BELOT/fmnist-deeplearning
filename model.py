"""
Deep learning models to learn on the task
"""
import torch.nn as nn
import torch
from typing import Optional, Any

class LinearNet(nn.Module):
    """
    Linear fully connected layer for classification
    """

    def __init__(
            self, 
            n_input: int,
            n_classes: int) -> None:
        """
        Args:
            n_input: int
                dimension of input data
            n_classes: int
                number of classes for classification (dim of output data with 1-hot encoding)

        """
        super(LinearNet, self).__init__()
        self.n_input: int = n_input
        self.n_classes: int = n_classes

        self.regularized: bool = False
        
        # Creating linear layer
        self.classifier = nn.Linear(self.n_input, self.n_classes)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes sample through layer"""
        # We flatten the data to make it 2D:
        # Example: 
        #       a.size() = torch.Size([128, 1, 28, 28]);     
        #       a.view(a.size()[0], -1).size() = torch.Size([128, 784])
        x = x.view(x.size()[0], -1)
        y: Any = self.classifier(x)
        return y
    

def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]


class FullyConnected(nn.Module):

    def __init__(
            self, 
            n_input: int, 
            n_classes: int) -> None:
        super(FullyConnected, self).__init__()
        self.classifier =  nn.Sequential(
            *linear_relu(n_input, 256),
            *linear_relu(256, 256),
            nn.Linear(256, n_classes)
        )
        self.regularized: bool = False

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnectedRegularized(nn.Module):

    def __init__(self, n_input: int, n_classes: int, l2_reg: float):
        super(FullyConnectedRegularized, self).__init__()
        self.l2_reg: float = l2_reg
        self.lin1 = nn.Linear(n_input, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, n_classes)

        self.regularized: bool = True


    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) + self.lin2.weight.norm(2) + self.lin3.weight.norm(2))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y
    
def conv_relu_maxp(in_channels, out_channels, ks):
    """
    Definition of Conv-Relu-MaxPool layer in Pytorch
    """
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]

def bn_dropout_linear(dim_in, dim_out, p_drop):
    return [nn.BatchNorm1d(dim_in),
            nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out)]

def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

def bn_dropout_linear_relu(dim_in, dim_out, p_drop):
    return bn_dropout_linear(dim_in, dim_out, p_drop) + [nn.ReLU(inplace=True)]

class VanillaCNN(nn.Module):
    
    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(1, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5)
        )

        # Or you create a dummy tensor for probing the size of the feature maps
        probe_tensor = torch.zeros((1,1,28,28))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256, num_classes)
        )
        self.regularized: bool = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  #  OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y