"""Training module."""

import logging
import numpy as np
import os
import torch

from collections import defaultdict
from sklearn.metrics import roc_auc_score
from timeit import default_timer
from tqdm import trange

from flexehr.utils.modelIO import save_model
from utils.helpers import array

import itertools 

class Trainer():
    """
    Class to handle model training and evaluation

    Parameters
    ----------
    model: flexehr.models.model
        Model to be evaluated.

    loss_f: flexehr.models.losses
        Loss function.

    optimizer: torch.optim.optimizer
        PyTorch optimizer used to minimize `loss_f`.

    device: torch.device, optional
        Device used for running the model.

    early_stopping: bool, optional
        Whether to make use of early stopping.

    save_dir: str, optional
        Name of save directory.

    p_bar: bool, optional
        Whether to have a progress bar.

    logger: logger.Logger, optional
        Logger to record info.
    """

    def __init__(self, model, loss_f, args,
                 optimizer=None,
                 device=torch.device('cpu'),
                 early_stopping=True,
                 save_dir='results',
                 p_bar=True,
                 coef_pos=0.5,
                 bal_loss_r=0.5,
                 logger=logging.getLogger(__name__)):

        self.model = model
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = 0 if early_stopping else None
        self.save_dir = save_dir
        self.p_bar = p_bar
        self.logger = logger

        self.max_v_auroc = -np.inf
        if self.optimizer is not None:
            self.losses_logger = LossesLogger(
                os.path.join(self.save_dir, 'train_losses.log'))
        
        self.logger.info(f'Device: {self.device}')

    def train(self, train_loader_pos, train_loader_neg, train_loader, 
              valid_loader_pos, valid_loader_neg, valid_loader,
              epochs=10,
              early_stopping=5):
        """Trains the model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader

        valid_loader : torch.utils.data.DataLoader

        epochs : int, optional
            Number of training epochs.

        early_stopping : int, optional
            Number of epochs to allow before early stopping.
        """
        start = default_timer()
        
        for epoch in range(epochs):
            storer = defaultdict(list)
            
            self.model.train()
            t_loss = self._train_epoch(train_loader_pos, 
                                       train_loader_neg, 
                                       train_loader, 
                                       storer)
            
            self.model.eval()
            v_loss = self._valid_epoch(valid_loader_pos, 
                                       valid_loader_neg, 
                                       valid_loader, 
                                       storer)

            self.logger.info(f'Train loss {t_loss:.4f}')
            self.logger.info(f'Valid loss {v_loss:.4f}')
            self.logger.info(f'Valid bal auroc {storer["auroc_bal"][0]:.4f}')
            self.logger.info(f'Valid rand auroc {storer["auroc_rand"][0]:.4f}')
            self.losses_logger.log(epoch, storer)

            if storer['auroc_bal'][0] > self.max_v_auroc:
                self.max_v_auroc = storer['auroc_bal'][0]
                save_model(self.model, self.save_dir, filename='model.pt')
                self.early_stopping = 0

            if self.early_stopping == early_stopping:
                break
            self.early_stopping += 1

        delta_time = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {delta_time:.1f} minutes.')

    def _train_epoch(self, data_loader_pos, data_loader_neg, data_loader
                     , storer):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.
        
        
        loader_pos = iter(data_loader_pos)
        loader_neg = iter(data_loader_neg)

        with trange(len(data_loader)) as t:  
            i = 0
            for data, y_true in data_loader:
                i += 1
                #print(i)
                try:
                    X_pos, y_pos = next(loader_pos) 
                except StopIteration:
                    loader_pos = iter(data_loader_pos)
                    X_pos, y_pos = next(loader_pos) 
                try:
                    X_neg, y_neg = next(loader_neg) 
                except StopIteration:
                    loader_neg = iter(data_loader_neg)
                    X_neg, y_neg = next(loader_neg)
                
                data = data.to(self.device)
                y_true = y_true.to(self.device)
                X_pos = X_pos.to(self.device)
                y_pos = y_pos.to(self.device)
                X_neg = X_neg.to(self.device)
                y_neg = y_neg.to(self.device)
                #print(data.shape, y_true.shape, X_pos.shape, y_pos.shape, X_neg.shape, y_neg.shape)

                # random sampler
                if_main = False
                y_pred = self.model(data, if_main=if_main)

                iter_loss_rand = self.loss_f(y_pred, y_true, 
                                             self.model.training, if_main, storer)
                #print(y_pred)
                
                # balanced sampler
                if_main = True
                y_pred_pos = self.model(X_pos, if_main=if_main)
                
                iter_loss_pos = self.loss_f(y_pred_pos, y_pos, 
                                             self.model.training, if_main, storer)
                y_pred_neg = self.model(X_neg, if_main)
                
                iter_loss_neg = self.loss_f(y_pred_neg, y_neg, 
                                             self.model.training, if_main, storer)

                #iter_loss_balance = iter_loss_pos * self.model.coef_pos.expand_as(iter_loss_pos) + \
                #                    iter_loss_neg * self.model.coef_neg.expand_as(iter_loss_neg)
                iter_loss_balance = iter_loss_pos * self.model.coef_pos.expand_as(iter_loss_pos) + \
                                    iter_loss_neg * torch.tensor(1-self.model.coef_pos).expand_as(iter_loss_neg)
                
                #iter_loss = iter_loss_rand * self.model.rand_loss_r.expand_as(iter_loss_rand) + \
                #            iter_loss_balance * self.model.bal_loss_r.expand_as(iter_loss_balance)
                iter_loss = iter_loss_rand * torch.tensor(1-self.model.bal_loss_r).expand_as(iter_loss_rand) + \
                            iter_loss_balance * self.model.bal_loss_r.expand_as(iter_loss_balance)



                #print(self.model.coef_pos, self.model.coef_neg)
                epoch_loss += iter_loss.item()

                self.optimizer.zero_grad()
                iter_loss.backward()
                self.optimizer.step()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()
                
        return epoch_loss / len(data_loader)

    def _valid_epoch(self, data_loader_pos, data_loader_neg, 
                     data_loader, storer=defaultdict(list)):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.
        
        coef_pos = 0.5
        coef_neg = 0.5
        
        y_preds_bal = []
        y_trues_bal = []
        y_preds_rand = []
        y_trues_rand = []
        
        loader_pos = iter(data_loader_pos)
        loader_neg = iter(data_loader_neg)

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                
                try:
                    X_pos, y_pos = next(loader_pos) 
                except StopIteration:
                    loader_pos = iter(data_loader_pos)
                    X_pos, y_pos = next(loader_pos) 
                try:
                    X_neg, y_neg = next(loader_neg) 
                except StopIteration:
                    loader_neg = iter(data_loader_neg)
                    X_neg, y_neg = next(loader_neg)
                
                
                data = data.to(self.device)
                y_true = y_true.to(self.device)
                X_pos = X_pos.to(self.device)
                y_pos = y_pos.to(self.device)
                X_neg = X_neg.to(self.device)
                y_neg = y_neg.to(self.device)

                if_main = False
                y_pred = self.model(data, if_main)
                
                iter_loss_rand = self.loss_f(
                    y_pred, y_true, self.model.training, if_main, storer)
                
                
                if_main = True
                y_pred_pos = self.model(X_pos, if_main)
                
                iter_loss_pos = self.loss_f(y_pred_pos, y_pos, 
                                             self.model.training, if_main, storer)
                
                y_pred_neg = self.model(X_neg, if_main)
                
                iter_loss_neg = self.loss_f(y_pred_neg, y_neg, 
                                             self.model.training, if_main, storer)
                
                iter_loss_balance = iter_loss_pos * coef_pos + iter_loss_neg * coef_neg
                
                iter_loss = (iter_loss_rand + iter_loss_balance) * 0.5
                
                epoch_loss += iter_loss.item()

                #y_preds += [array(y_pred)]
                # only validate on balanced batch
                y_trues_bal += [array(y_pos)]
                y_trues_bal += [array(y_neg)]
                
                y_preds_bal += [array(y_pred_pos)]
                y_preds_bal += [array(y_pred_neg)]
                
                y_trues_rand += [array(y_true)]
                y_preds_rand += [array(y_pred)]
                
                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        y_preds_bal = np.concatenate(y_preds_bal)
        y_trues_bal = np.concatenate(y_trues_bal)
        
        y_trues_rand = np.concatenate(y_trues_rand)
        y_preds_rand = np.concatenate(y_preds_rand)
        
        #y_trues = data_loader.dataset.Y

        metrics = self.compute_metrics(y_preds_bal, y_trues_bal, if_bal=True)
        storer.update(metrics)
        
        metrics = self.compute_metrics(y_preds_rand, y_trues_rand, if_bal=False)
        storer.update(metrics)

        return epoch_loss / len(data_loader)

    def compute_metrics(self, y_pred, y_true, if_bal=False):
        """Compute metrics for predicted vs true labels."""
        if not isinstance(y_pred, np.ndarray):
            y_pred = array(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = array(y_true)

        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
            y_true = y_true[:, -1]

        metrics = {}
        if if_bal:
            metrics['auroc_bal'] = [roc_auc_score(y_true, y_pred)]
        else:
            metrics['auroc_rand'] = [roc_auc_score(y_true, y_pred)]
        
        return metrics


class LossesLogger(object):
    """
    Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """Create a logger to store information for plotting."""
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger('losses_logger')
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.addHandler(file_handler)

        header = ','.join(['Epoch', 'Train Loss', 'Valid Loss', 'bal AUROC', 'rand AUROC'])
        self.logger.debug(header)

    def log(self, epoch, storer):
        """Write to the log file."""
        log_string = [epoch+1]
        for k in storer.keys():
            log_string += [sum(storer[k]) / len(storer[k])]
        log_string = ','.join(str(item) for item in log_string)
        self.logger.debug(log_string)
