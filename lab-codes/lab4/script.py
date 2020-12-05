import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt
text = np.loadtxt("/Users/megh/Work/academics/Machine_Learning/lab4/data/data_perceptron.txt")
data = text[:,0:2]
labels = text[:,2].T
# Scatter plotting the points on the plane of different classes
plt.figure(1)
plt.scatter(data[0:3,0],data[0:3,1],c='r')
plt.scatter(data[3:5,0],data[3:5,1],c='b')

# defining the dimensions
dim1_min, dim1_max, dim2_min, dim2_max = 0,1,0,1
num_output = np.expand_dims(labels,axis=0).shape[0]
dim1 = [dim1_min,dim1_max]
dim2 = [dim2_min,dim2_max]
# define the perceptron
perceptron = nl.net.newp([dim1,dim2],num_output)
# Training of model 
error_progress = perceptron.train(data,labels,epochs=100,show=10,lr=0.03)
