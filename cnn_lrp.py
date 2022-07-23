# RESOURCES:
# https://git.tu-berlin.de/gmontavon/lrp-tutorial
# https://github.com/sebastian-lapuschkin/lrp_toolbox
# https://github.com/albermax/innvestigate/blob/master/README.md
# https://github.com/chr5tphr/zennit
# https://giorgiomorales.github.io/Layer-wise-Relevance-Propagation-in-Pytorch/
# http://www.heatmapping.org/index.html#software
# https://lrpserver.hhi.fraunhofer.de/handwriting-classification

from random import sample
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np
import copy
import cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + str(device))
model = cnn.CNN()
model.load_state_dict(torch.load('/Users/sean/Documents/Neubauer_Research/MNIST/cnn_model', map_location=device))
print('Model loaded...')

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.ToTensor()
)
print('Retrieved dataset...')

# getting a predictions and labels of sample data from test_data
def predict_activation(sample_idx):
    return torch.max(model(test_data[sample_idx][0].unsqueeze(0)).detach())

def predict_label(sample_idx):
    max_ = predict_activation(sample_idx)
    x = model(test_data[sample_idx][0].unsqueeze(0)).detach()[0]
    return (x==max_).nonzero().item()

def plot_sample_image(sample_idx):
    sample_image = test_data[sample_idx][0]
    sample_label = test_data[sample_idx][1] 

    plt.imshow(sample_image.reshape(28, 28))
    plt.show()
    print('Label: ' + str(sample_label))
    print('Prediction: ' + str(predict_label(sample_idx)))
    print('Activation: ' + str(predict_activation(sample_idx)))
    for i in range(len(model(sample_image.unsqueeze(0)).detach().flatten())):
        print(str(i) + ' : ' + str(model(sample_image.unsqueeze(0)).detach().flatten()[i]))

def rho(w,l):  
    return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)

def incr(z,l): 
    return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9

def standardize_image(sample_image):
    mean = torch.Tensor([0.1307]).reshape(1,-1,1,1)
    std  = torch.Tensor([0.3081]).reshape(1,-1,1,1)
    return (torch.FloatTensor(sample_image*1) - mean) / std

def newlayer(layer,g):
    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass
    return layer


layers = [layer for layer in model.modules() if not isinstance(layer, nn.Sequential) and not isinstance(layer, cnn.CNN)]
print('Layers: ' + str(layers))
R = model(test_data[0][0].unsqueeze(0).detach())
# # from R only choose desired class to explain
# # here we choose the predicted class
# R = torch.max(R)

params = [layers[i].state_dict() for i in range(len(layers))]
keys = [param.keys() for param in params]
weights = [params[i]['weight'] if 'weight' in keys[i] else None for i in range(len(params))]
biases = [params[i]['bias'] if 'bias' in keys[i] else None for i in range(len(params))]
L = len(weights)

# sample image and label
sample_idx = 0
sample_image = test_data[sample_idx][0]
sample_label = test_data[sample_idx][1] 

X = standardize_image(sample_image=sample_image)

A = [X]+[None]*L
for l in range(L):
    A[l+1] = layers[l].forward(A[l])

scores = np.array(A[-1].data.view(-1))
T = torch.FloatTensor((1.0*(np.arange(1000)==483).reshape([1,1000,1,1])))
R = [None]*L + [(A[-1]*T).data]

for l in range(1,L)[::-1]:
    A[l] = (A[l].data).requires_grad_(True)

    if isinstance(layers[l],torch.nn.MaxPool2d): 
        layers[l] = torch.nn.AvgPool2d(2)
    if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
        if l <= 2:       
            rho = lambda p: p + 0.25*p.clamp(min=0)
            incr = lambda z: z+1e-9
        if 3 <= l <= 5: 
            rho = lambda p: p
            incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if l >= 6:       
            rho = lambda p: p
            incr = lambda z: z+1e-9

        z = incr(newlayer(layers[l],rho).forward(A[l]))        # step 1
        print(R[l+1].shape)
        print(z.shape)
        s = (R[l+1]/z).data                                    # step 2
        (z*s).sum().backward() 
        c = A[l].grad                  # step 3
        R[l] = (A[l]*c).data                                   # step 4
    else:
        R[l] = R[l+1]