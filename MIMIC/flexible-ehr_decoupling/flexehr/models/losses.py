"""Module containing all binary classification losses."""

import abc
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch

class BaseLoss(abc.ABC):
    """
    Base class for losses.

    Parameters
    ----------
    record_loss_every: int, optional
        How many steps between each loss record.
    """

    def __init__(self, record_loss_every=1):
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every

    @abc.abstractmethod
    def __call__(self, y_pred, y_true, is_train, storer):
        """Calculates loss for a batch of data."""

    def _pre_call(self, is_train, storer):
        if is_train:
            self.n_train_steps += 1

        if not is_train or self.n_train_steps % self.record_loss_every == 0:
            storer = storer
        else:
            storer = None

        return storer


class BCE(BaseLoss):
    def __init__(self):
        """Compute the binary cross entropy loss."""
        super().__init__()

    def __call__(self, y_pred, y_true, is_train, if_main, weight, storer):
        """Binary cross entropy loss function.

        Parameters
        ----------
        y_pred : torch.Tensor

        y_true : torch.Tensor

        is_trin : bool
            Whether model is training.

        storer: collections.defaultdict
        """
        storer = self._pre_call(is_train, storer)

        self.weight = weight
        #loss = F.binary_cross_entropy(y_pred, y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)

        if storer is not None:
            if is_train:
                storer['train_loss'].append(loss.item())
            else:
                storer['valid_loss'].append(loss.item())

        return loss

class Focal_loss(BaseLoss):
    def __init__(self):
        """Compute the binary cross entropy loss."""
        super().__init__()

    def __call__(self, y_pred, y_true, is_train, if_main, weight, storer):
        """Binary cross entropy loss function.

        Parameters
        ----------
        y_pred : torch.Tensor

        y_true : torch.Tensor

        is_trin : bool
            Whether model is training.

        storer: collections.defaultdict
        """
        self.gamma = 2.
        
        storer = self._pre_call(is_train, storer)

        self.weight = weight
        #loss = F.binary_cross_entropy(y_pred, y_true)
        BCE_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        #focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        focal_loss = (1-pt)**self.gamma * BCE_loss
        loss = focal_loss.mean()

        if storer is not None:
            if is_train:
                storer['train_loss'].append(loss.item())
            else:
                storer['valid_loss'].append(loss.item())

        return loss
    
class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        """Compute the binary cross entropy loss."""
        super().__init__()

    def __call__(self, y_pred, y_true, is_train, if_main, weight, storer):
        """Binary cross entropy loss function.

        Parameters
        ----------
        y_pred : torch.Tensor

        y_true : torch.Tensor

        is_trin : bool
            Whether model is training.

        storer: collections.defaultdict
        """
        storer = self._pre_call(is_train, storer)
        self.weight = weight
        #criterion = nn.BCEWithLogitsLoss()
        #loss = criterion(y_pred, y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        if storer is not None:
            if is_train:
                storer['train_loss'].append(loss.item())
            else:
                storer['valid_loss'].append(loss.item())

        return loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target)

class LDAMLoss_(BaseLoss):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss_, self).__init__()
        self.cls_num_list = cls_num_list
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def __call__(self, x, target, is_train, if_main, weight, storer):
        x, target = x.reshape(-1,), target.reshape(-1,)
        storer = self._pre_call(is_train, storer)

        x = torch.unsqueeze(x, dim=1)
        target = torch.unsqueeze(target, dim=1)
        x = torch.cat((1 - x, x), dim=1)
        
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1).long(), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)

        loss = F.cross_entropy(self.s*output, target.squeeze().long(), weight=weight)

        if storer is not None:
            if is_train:
                storer['train_loss'].append(loss.item())
            else:
                storer['valid_loss'].append(loss.item())

        return loss