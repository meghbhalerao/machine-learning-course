import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import random
from all_augmentations import *
from utils import *
import random
import scipy

class EMNIST(Dataset):
    def __init__(self, images_tr, labels_tr):
        self.images_tr = images_tr
        self.labels_tr = labels_tr
    def __len__(self):
        return len(self.labels_tr)
    def one_hot(gt):
        oh = np.zeros(26)
        oh[gt-1] = 1
        return oh
    def __getitem__(self, index):
        image = self.images_tr[index,:]
        gt = self.images_tr(index,:)
        gt = one_hot(gt)
        sample = {'image': image, 'gt' : gt}
        return sample

