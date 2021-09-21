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
        #self.coef_pos = torch.nn.Parameter(torch.tensor(coef_pos), requires_grad=True)
        #self.coef_neg = torch.nn.Parameter(torch.tensor(1-coef_pos), requires_grad=True)
        #self.bal_loss_r = torch.nn.Parameter(torch.tensor(bal_loss_r), requires_grad=True)
        #self.rand_loss_r = torch.nn.Parameter(torch.tensor(1-bal_loss_r), requires_grad=True)
        #self.coef_pos = self.coef_pos.to(self.device)
        #self.coef_neg= self.coef_neg.to(self.device)
        #self.bal_loss_r = self.bal_loss_r.to(self.device)
        #self.rand_loss_r = self.rand_loss_r.to(self.device)
        #self.optimizer.add_param_group({'params': [self.coef_pos, self.coef_neg, 
        #                                           self.bal_loss_r, self.rand_loss_r]})
        #self.optimizer = torch.optim.Adam(itertools.chain(model.parameters(), 
        #                                                  (self.coef_pos, 
        #                                                  self.coef_neg, 
        #                                                  self.bal_loss_r, 
        #                                                  self.rand_loss_r)), lr=args.lr, weight_decay=0.005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        
    def train(self, train_loader_bal, train_loader_rand,  
              valid_loader_bal, valid_loader_rand,
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
            t_loss = self._train_epoch(train_loader_bal, 
                                       train_loader_rand, 
                                       storer)

            self.model.eval()
            v_loss = self._valid_epoch(valid_loader_bal, 
                                       valid_loader_rand, 
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

            self.scheduler.step()
            
        delta_time = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {delta_time:.1f} minutes.')

    def _train_epoch(self, data_loader_bal, data_loader_rand, storer):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.

        
        loader_bal = iter(data_loader_bal)
        data_loader = data_loader_rand

        with trange(len(data_loader_rand)) as t:            
            for data, y_true in data_loader_rand:
                
                try:
                    X_bal, y_bal = next(loader_bal) 
                except StopIteration:
                    loader_bal = iter(data_loader_bal)
                    X_bal, y_bal = next(loader_bal)                 
                
                data = data.to(self.device)
                y_true = y_true.to(self.device)
                X_bal = X_bal.to(self.device)
                y_bal = y_bal.to(self.device)
                
                

                # random sampler
                if_main = False
                y_pred = self.model(data, if_main=if_main)
                #print(y_true.shape, y_pred.shape)
                iter_loss_rand = self.loss_f(y_pred, y_true, 
                                             self.model.training, if_main, storer)
                
                
                # balanced sampler
                if_main = True
                y_pred_bal = self.model(X_bal, if_main)
                iter_loss_balance = self.loss_f(y_pred_bal, y_bal, 
                                             self.model.training, if_main, storer)
                
                iter_loss = iter_loss_rand * self.model.rand_loss_r.expand_as(iter_loss_rand) + \
                            iter_loss_balance * self.model.bal_loss_r.expand_as(iter_loss_balance)
                epoch_loss += iter_loss.item()

                self.optimizer.zero_grad()
                iter_loss.backward()
                self.optimizer.step()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        return epoch_loss / len(data_loader)

    def _valid_epoch(self, data_loader_bal, data_loader_rand, 
                     storer=defaultdict(list)):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.
        
        coef_pos = 0.5
        coef_neg = 0.5
        
        y_preds_bal = []
        y_trues_bal = []
        y_preds_rand = []
        y_trues_rand = []
        
        loader_bal = iter(data_loader_bal)
        data_loader = data_loader_rand

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                
                try:
                    X_bal, y_bal = next(loader_bal) 
                except StopIteration:
                    loader_bal = iter(data_loader_bal)
                    X_bal, y_bal = next(loader_bal)
                                
                data = data.to(self.device)
                y_true = y_true.to(self.device)
                X_bal = X_bal.to(self.device)
                y_bal = y_bal.to(self.device)

                #random samper
                if_main = False
                y_pred = self.model(data, if_main)
                
                iter_loss_rand = self.loss_f(
                    y_pred, y_true, self.model.training, if_main, storer)
                
                
                if_main = True
                y_pred_bal = self.model(X_bal, if_main)
                
                iter_loss_balance = self.loss_f(y_pred_bal, y_bal,
                                             self.model.training, if_main, storer)
                
                iter_loss = (iter_loss_rand + iter_loss_balance) * 0.5
                
                epoch_loss += iter_loss.item()

                #y_preds += [array(y_pred)]
                # only validate on balanced batch
                y_trues_bal += [array(y_bal)]
                y_preds_bal += [array(y_pred_bal)]

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
            
            if y_pred.shape[-1] == 2:
                y_pred = y_pred[:, -1]
                y_true = y_true[:, -1]

                metrics = {}
                if if_bal:
                    metrics['auroc_bal'] = [roc_auc_score(y_true, y_pred)]
                else:
                    metrics['auroc_rand'] = [roc_auc_score(y_true, y_pred)]

                return metrics
            
            elif y_pred.shape[-1] == 25:
                auroc_allpheno = []
                for pheno_idx in range(y_pred.shape[-1]):
                    y_pred_ = y_pred[:, pheno_idx]
                    y_true_ = y_true[:, pheno_idx]
                    auroc = roc_auc_score(y_true_, y_pred_)
                    auroc_allpheno.append(auroc)

                macro_mean = np.mean(auroc_allpheno)
                metrics = {}
                if if_bal:
                    metrics['auroc_bal'] = [macro_mean]
                else:
                    metrics['auroc_rand'] = [macro_mean]

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
