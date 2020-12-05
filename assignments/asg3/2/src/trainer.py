import numpy as np
from emnist import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from data_val import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
images_tr, labels_tr = extract_training_samples('letters')
images_ts, labels_ts = extract_test_samples('letters')

loss_fn = torch.nn.MSELoss()


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
        x = F.sigmoid(x)
        return x


model = olp(h*w,num_class)
optimizer = optim.Adam(model.parameters(), lr = 0.01, betas = (0.9,0.999), weight_decay = 0.00005)
model.cpu()
num_epochs = 100
dataset_train = EMNIST(images_tr, labels_tr)
train_loader = DataLoader(dataset_train,batch_size= 1,shuffle=True, num_workers=4)
dataset_valid = EMNIST_val(images_val, labels_val)
val_loader = DataLoader(dataset_valid, batch_size = 1, shuffle = True, num_workers = 4)
acc_tmp = 0
for ep in range(num_epochs):
    # Setting the model to train mode
    loss_agg = 0
    acc = 0
    model.train
    for batch_idx, (subject) in enumerate(train_loader):
        # Load the subject and its ground truth
        image = subject['image']
        mask = subject['gt']
        image.cpu()
        mask.cpu()
        optimizer.zero_grad()
        # Forward Propagation to get the output from the models
        output = model(image.float())
        # Computing the loss    
        loss = loss_fn(output.double(), mask.double())
        # Back Propagation for model to learn
        loss.backward()
        #Updating the weight values
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        loss_agg = loss_agg + loss
        output = output.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        tmp = np.zeros(output.shape)
        tmp[:,np.argmax(output)] = 1
        acc = acc + np.sum(tmp*mask)
    print("Train Loss for epoch :",ep,"  ",loss_agg/(batch_idx+1))
    print("Accuracy of Training is: ", acc/(batch_idx+1))
    model.eval   
    loss_agg_val = 0
    acc = 0
    for batch_idx, (subject) in enumerate(val_loader):
        with torch.no_grad():
            image = subject['image']
            mask = subject['gt']
            output = model(image.float())
            loss = loss_fn(output.double(), mask.double())
            loss = loss.detach().cpu().numpy()
            loss_agg_val = loss_agg_val + loss
            output = output.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
            tmp = np.zeros(output.shape)
            tmp[:,np.argmax(output)] = 1
            acc = acc + np.sum(tmp*mask)

    print("Validation loss for epoch :",ep," ",loss_agg_val/(batch_idx+1))
    print("Accuracy of Validation is: ", acc/(batch_idx+1))
    if acc>acc_tmp:
        acc_tmp = acc
        torch.save(model,"mod.pt")



        



