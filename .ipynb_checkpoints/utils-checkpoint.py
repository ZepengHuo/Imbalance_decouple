
import pdb
import torch 
import pickle
import numpy as np  
from torch.utils.data import Dataset

class Labeled_dataset(Dataset):

    def __init__(self, _dir, transform, target_list, offset=0, num=None):
        super(Labeled_dataset, self).__init__()

        self.imgdir=_dir
        self.transforms=transform
        self.all_image = pickle.load(open(self.imgdir,'rb'))
        self.img = []
        self.target = []

        print('target list = ', target_list)
        for i,idx in enumerate(target_list):
            self.img.append(self.all_image[idx])
            self.target.append((i+offset)*np.ones(self.all_image[idx].shape[0]))

        self.image = np.concatenate(self.img, 0)
        self.label = np.concatenate(self.target, 0)
        self.number = self.image.shape[0]

        if num:
            index = np.random.permutation(self.number)
            select_index = index[:int(num)]
            self.image = self.image[select_index]
            self.label = self.label[select_index]
            self.number = num
        
    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.image[index]
        target = self.label[index]
        img = self.transforms(img)

        return img, target



