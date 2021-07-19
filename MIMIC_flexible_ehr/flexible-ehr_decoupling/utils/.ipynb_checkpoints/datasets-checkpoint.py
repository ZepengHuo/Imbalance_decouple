"""Datasets module."""

import logging
import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .sampler import MultilabelBalancedRandomSampler

def get_dataloaders(root, t_hours, n_bins, validation, task, dt=1.0, dynamic=True,
                    shuffle=True, pin_memory=True, batch_size=128,
                    logger=logging.getLogger(__name__)):
    """A generic data loader.

    Parameters
    ----------
    root: str
        Root directory.

    t_hours: int


    n_bins: int


    validation: bool
        Whether or not to return a validation DataLoader also (when training).
        
    task: pheno/mortality
        what task to predict

    dt: float, optional
        Time step between intervals.

    dynamic: bool, optional
        Whether the model should predict in a dynamic fashion.

    shuffle: bool, optional
        Whether to shuffle data during training.

    pin_memory: bool, optional
        Whether to pin memory in the GPU when using CUDA.

    batch_size: int, optional

    logger: logging.Logger, optional
    """
    pin_memory = pin_memory and torch.cuda.is_available

    arrs = np.load(
        os.path.join(root, '_dicts', f'{t_hours}_{n_bins}_arrs_new.npy'),
        #os.path.join(root, '_dicts', f'{t_hours}_{n_bins}_arrs.npy'),
        allow_pickle=True).item()
    if validation:
        if task == 'pheno':
            X_train, X_valid, y_train, y_valid = train_test_split(
                #arrs['X_train'], arrs['Y_train'],
                #arrs['X'], arrs['Y'],
                arrs['X'], arrs['pheno'],
                #test_size=1000, stratify=arrs['Y_train'])
                test_size=1000, stratify=arrs['Y'])

            train_dataset = EHR_pheno(X_train, y_train, t_hours, dt, dynamic)

            valid_dataset = EHR_pheno(X_valid, y_valid, t_hours, dt, dynamic)
            
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(
                #arrs['X_train'], arrs['Y_train'],
                arrs['X'], arrs['Y'],
                #arrs['X'], arrs['pheno'],
                #test_size=1000, stratify=arrs['Y_train'])
                test_size=1000, stratify=arrs['Y'])

            train_dataset = EHR(X_train, y_train, t_hours, dt, dynamic)

            valid_dataset = EHR(X_valid, y_valid, t_hours, dt, dynamic)
        
        
        if task != 'pheno':
            train_dataset_pos = EHR_pos(X_train, y_train, t_hours, dt, dynamic)
            train_dataset_neg = EHR_neg(X_train, y_train, t_hours, dt, dynamic)
            
            valid_dataset_pos = EHR_pos(X_valid, y_valid, t_hours, dt, dynamic)
            valid_dataset_neg = EHR_neg(X_valid, y_valid, t_hours, dt, dynamic)
            
            train_dataloader_pos = DataLoader(train_dataset_pos,
                                          batch_size=int(batch_size/2),
                                          shuffle=shuffle,
                                          pin_memory=pin_memory)
            train_dataloader_neg = DataLoader(train_dataset_neg,
                                          batch_size=int(batch_size/2),
                                          shuffle=shuffle,
                                          pin_memory=pin_memory)        
            train_dataloader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          pin_memory=pin_memory)


            valid_dataloader_pos = DataLoader(valid_dataset_pos,
                                          batch_size=int(batch_size/2),
                                          shuffle=False,
                                          pin_memory=pin_memory)
            valid_dataloader_neg = DataLoader(valid_dataset_neg,
                                          batch_size=int(batch_size/2),
                                          shuffle=False,
                                          pin_memory=pin_memory)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=pin_memory)

            return train_dataloader_pos, train_dataloader_neg, train_dataloader, \
        valid_dataloader_pos, valid_dataloader_neg, valid_dataloader
        
        else:
            train_dataloader_rand = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          pin_memory=pin_memory)

            train_sampler = MultilabelBalancedRandomSampler(train_dataset.Y, class_choice="least_sampled")
            train_loader_bal = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            
            
            valid_dataloader_rand = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          pin_memory=pin_memory)
            
            valid_sampler = MultilabelBalancedRandomSampler(valid_dataset.Y, class_choice="least_sampled")
            valid_loader_bal = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
            
            return train_loader_bal, train_dataloader_rand, valid_loader_bal, valid_dataloader_rand

    else:
        test_dataset = EHR(
            arrs['X_test'], arrs['Y_test'], t_hours, dt, dynamic)
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     pin_memory=pin_memory)

        return test_dataloader, None


class EHR(Dataset):
    """
    EHR Dataset.

    Parameters
    ----------
    X: numpy.ndarray
        Array containing patient sequences, shape (n_patients, 10000, 2)

    Y: numpy.ndarray
        Array containing patient outcomes, shape (n_patients,)

    t_hours: int, optional


    dt: float, optional
        Time step between intervals.

    dynamic: bool, optional
        Whether the model should predict in a dynamic fashion.

    logger: logging.Logger, optional
    """

    def __init__(self, X, Y, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):

        self.logger = logger

        self.X = torch.tensor(X)
        if dynamic:  # shape (n_patients,) -> (n_patients, n_intervals)
            Y = np.tile(Y[:, None], (1, int(t_hours / dt)))
        temp = []
        for i in Y:
            temp.append(i.tolist())
        Y = temp
        self.Y = torch.tensor(Y).float()

        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    
class EHR_pheno(Dataset):
    def __init__(self, X, Y, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):
        #self.n_examples = n_examples
        #self.n_features = n_features
        #self.n_classes = n_classes
        #self.X = np.random.random([self.n_examples, self.n_features])

        #class_probabilities = np.random.random([self.n_classes])
        #class_probabilities = class_probabilities / sum(class_probabilities)
        #class_probabilities *= mean_labels_per_example
        #self.Y = (
        #    np.random.random([self.n_examples, self.n_classes]) < class_probabilities
        #).astype(int)
        self.X = torch.tensor(X)
        temp = []
        for i in Y:
            temp.append(i.tolist())
        Y = temp
        self.Y = torch.tensor(Y).float()

        self.len = len(self.X)
        
    def __len__(self):
        #return self.n_examples
        return self.len

    def __getitem__(self, index):
        #example = Variable(torch.tensor(self.X[index]), requires_grad=False)
        #labels = Variable(torch.tensor(self.Y[index]), requires_grad=False)
        #return {"example": example, "labels": labels}
        return self.X[index], self.Y[index]

    
class EHR_pos(Dataset):
    """
    EHR Dataset.

    Parameters
    ----------
    X: numpy.ndarray
        Array containing patient sequences, shape (n_patients, 10000, 2)

    Y: numpy.ndarray
        Array containing patient outcomes, shape (n_patients,)

    t_hours: int, optional


    dt: float, optional
        Time step between intervals.

    dynamic: bool, optional
        Whether the model should predict in a dynamic fashion.

    logger: logging.Logger, optional
    """

    def __init__(self, X, Y, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):

        self.logger = logger
        
        # balanced batch
        pos_idx = np.where(Y == 1)[0]
        
        X_pos = torch.Tensor(X[pos_idx])

        Y_pos = torch.Tensor(Y[pos_idx])

        self.X_pos = X_pos
        
        if dynamic:  # shape (n_patients,) -> (n_patients, n_intervals)
            Y_pos = np.tile(Y_pos[:, None], (1, int(t_hours / dt)))
            Y_pos = torch.Tensor(Y_pos)
            
        self.Y_pos = Y_pos.float()

        self.len = len(self.X_pos)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X_pos[idx], self.Y_pos[idx]
    
    
class EHR_neg(Dataset):
    """
    EHR Dataset.

    Parameters
    ----------
    X: numpy.ndarray
        Array containing patient sequences, shape (n_patients, 10000, 2)

    Y: numpy.ndarray
        Array containing patient outcomes, shape (n_patients,)

    t_hours: int, optional


    dt: float, optional
        Time step between intervals.

    dynamic: bool, optional
        Whether the model should predict in a dynamic fashion.

    logger: logging.Logger, optional
    """

    def __init__(self, X, Y, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):

        self.logger = logger
        
        # balanced batch
        neg_idx = np.where(Y == 0)[0]
        
        X_neg = torch.Tensor(X[neg_idx])

        Y_neg = torch.Tensor(Y[neg_idx])

        self.X_neg = X_neg
        
        if dynamic:  # shape (n_patients,) -> (n_patients, n_intervals)
            Y_neg = np.tile(Y_neg[:, None], (1, int(t_hours / dt)))
            Y_neg = torch.Tensor(Y_neg)
            
        self.Y_neg = Y_neg

        self.len = len(self.X_neg)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X_neg[idx], self.Y_neg[idx]