import numpy as np
from emnist import *
import torch
import torch.nn as nn
import torch.nn.functional as F
images_tr, labels_tr = extract_training_samples('letters')
images_ts, labels_ts = extract_test_samples('letters')
#print(images_tr.shape)
def CCE(pred,gt):
    return F.cross_entropy(pred,gt)

(n,h,w) = images_tr.shape
images_tr = images_tr.reshape(n,h*w)
images_val = images_tr[99840:,:]
labels_val = labels_tr[99840:]
num_class = 26
# define neural network
class olp(nn.Module):
    def __init__(self, input_neurons, output_neurons):
        nn.Module.__init__(self)
        self.fc = nn.Linear(input_neurons, output_neurons)
    def forward(self,x):
        x = self.fc(x)
        x = F.softmax(x)
net = olp(h*w,num_class)
num_epochs = 100
images_tr = torch.tensor(images_tr.astype(float),requires_grad=True)
labels_tr = torch.tensor(labels_tr.astype(float),requires_grad=True)
images_val = torch.tensor(images_val.astype(float),requires_grad=True)
labels_val = torch.tensor(labels_val.astype(float),requires_grad=True)
for ep in range(num_epochs):