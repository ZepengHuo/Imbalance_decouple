"""Training module."""

import logging
import numpy as np
import os
import torch

from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from timeit import default_timer
from tqdm import trange
from matplotlib import pyplot

from flexehr.utils.modelIO import save_model
from utils.helpers import array


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
                 logger=logging.getLogger(__name__)):

        self.model = model
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = 0 if early_stopping else None
        self.save_dir = save_dir
        self.p_bar = p_bar
        self.logger = logger
        self.args = args
        self.weight = None # weight for loss function

        self.max_v_auroc = -np.inf
        if self.optimizer is not None:
            self.losses_logger = LossesLogger(
                os.path.join(self.save_dir, 'train_losses.log'))
        self.logger.info(f'Device: {self.device}')

    def train(self, train_loader, valid_loader,
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
            
            # freeze backbone, lower lr for classifier after first epoch
            if self.args.annealing_lr and epoch == 0:
                #for p in self.model.gru.parameters():
                #        p.requires_grad = False
            
                #for g in self.optimizer.param_groups:
                #    g['lr'] = g['lr'] * self.args.annealing_lr
                    
                if self.args.train_rule == 'DRW':
                    
                    #idx = epoch // 1
                    idx = 1
                    betas = [0, 0.9999]
                    effective_num = 1.0 - np.power(betas[idx], self.loss_f.cls_num_list)
                    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.loss_f.cls_num_list)
                    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.args.gpu_num)
                    self.weight = per_cls_weights    

            self.model.train()
            t_loss = self._train_epoch(train_loader, storer)

            self.model.eval()
            v_loss = self._valid_epoch(valid_loader, epoch, storer)

            self.logger.info(f'Train loss {t_loss:.4f}')
            self.logger.info(f'Valid loss {v_loss:.4f}')
            self.logger.info(f'Valid auroc {storer["auroc"][0]:.4f}')
            self.logger.info(f'Valid auprc {storer["auprc"][0]:.4f}')
            self.losses_logger.log(epoch, storer)

            if storer['auroc'][0] > self.max_v_auroc:
                self.max_v_auroc = storer['auroc'][0]
                save_model(self.model, self.save_dir, filename='model.pt')
                self.early_stopping = 0

            if self.early_stopping == early_stopping:
                break
            self.early_stopping += 1

        delta_time = (default_timer() - start) / 60
        self.logger.info(f'Finished training after {delta_time:.1f} minutes.')

    def _train_epoch(self, data_loader, storer):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                #print(data[torch.where(data != data)])
                data = data.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(data)
                iter_loss = self.loss_f(
                    y_pred, y_true, self.model.training, self.weight, storer)
                epoch_loss += iter_loss.item()

                self.optimizer.zero_grad()
                iter_loss.backward()
                self.optimizer.step()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        return epoch_loss / len(data_loader)

    def _valid_epoch(self, data_loader, epoch, storer=defaultdict(list)):
        """Trains the model on the validation set for one epoch."""
        epoch_loss = 0.
        y_preds = []

        with trange(len(data_loader)) as t:
            for data, y_true in data_loader:
                data = data.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(data)
                y_preds += [array(y_pred)]
                iter_loss = self.loss_f(
                    y_pred, y_true, self.model.training, self.weight, storer)
                epoch_loss += iter_loss.item()

                if self.p_bar:
                    t.set_postfix(loss=iter_loss.item())
                    t.update()

        y_preds = np.concatenate(y_preds)
        y_trues = data_loader.dataset.Y
        
        if not os.path.isdir('saved'):
            os.makedirs('saved')

        with open(f'saved/y_preds_{epoch}.npy', 'wb') as f:
            np.save(f, y_preds)
        with open(f'saved/y_trues_{epoch}.npy', 'wb') as f:
            np.save(f, y_trues)
                    

        metrics = self.compute_metrics(y_preds, y_trues)
        storer.update(metrics)
        
        metrics = self.compute_metrics_auprc(y_preds, y_trues)
        storer.update(metrics)
        
        self.SavePlot_auprc(y_preds, y_trues, epoch)
        
        self.SavePlot_auroc(y_preds, y_trues, epoch)

        #self.SavePlot_brier(y_preds, y_trues, epoch)
        self.SavePlot_calibration(y_preds, y_trues, epoch)
        
        return epoch_loss / len(data_loader)

    def compute_metrics(self, y_pred, y_true):
        """Compute metrics for predicted vs true labels."""
        if not isinstance(y_pred, np.ndarray):
            y_pred = array(y_pred)
        if not isinstance(y_true, np.ndarray):
            y_true = array(y_true)

        if y_pred.ndim == 2:
            y_pred = y_pred[:, -1]
            y_true = y_true[:, -1]

        metrics = {}
        if np.isnan(np.sum(y_pred)):
            print('y_predc', y_pred)
        metrics['auroc'] = [roc_auc_score(y_true, y_pred)]
        
        return metrics

    def compute_metrics_auprc(self, y_pred, y_true):
            """Compute metrics for predicted vs true labels."""
            if not isinstance(y_pred, np.ndarray):
                y_pred = array(y_pred)
            if not isinstance(y_true, np.ndarray):
                y_true = array(y_true)

            if y_pred.ndim == 2:
                y_pred = y_pred[:, -1]
                y_true = y_true[:, -1]

            metrics = {}
            if np.isnan(np.sum(y_pred)):
                print('y_predc', y_pred)
            metrics['auprc'] = [average_precision_score(y_true, y_pred)]
            
            return metrics
        
    def SavePlot_auroc(self, y_pred, y_true, epoch):
        y_true, y_pred = y_true.reshape(1,-1), y_pred.reshape(1,-1)
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        fpr, tpr, _ = roc_curve(np.squeeze(y_true), np.squeeze(y_pred_sig))
        fig = pyplot.figure()
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.savefig('results/auroc' + str(epoch) +'.png')    
        
    def SavePlot_auprc(self, y_pred, y_true, epoch):
        y_true, y_pred = y_true.reshape(1,-1), y_pred.reshape(1,-1)
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        lr_precision, lr_recall, _ = precision_recall_curve(np.squeeze(y_true), np.squeeze(y_pred_sig))
        fig = pyplot.figure()
        pyplot.plot(lr_recall, lr_precision, marker='.')
        
        pyplot.savefig('results/auprc'+ str(epoch) +'.png')
        
        
    def SavePlot_brier(self, y_pred, y_true, epoch):
        cls_num_list = self.loss_f.cls_num_list
        ratio = float(cls_num_list[1]) / np.sum(cls_num_list)
        y_true, y_pred = y_true.reshape(1,-1).squeeze(), y_pred.reshape(1,-1).squeeze()
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        predictions_ref = [ratio] * len(y_pred)
        indices = np.argsort(y_pred_sig)
        y_pred_sig = y_pred_sig[indices]
        y_true = y_true[indices]
        BS = [brier_score_loss([target], [pred]) for (target, pred) in zip(y_true, y_pred_sig)]
        BS_ref = [brier_score_loss(y_true, [y for x in range(len(y_true))]) for y in predictions_ref]
        
        BS_skill = 1 - np.array(BS)/np.array(BS_ref)
        fig = pyplot.figure()
        pyplot.plot(y_pred_sig, BS_skill)
        pyplot.savefig('results/brier'+ str(epoch) +'.png')
        
        
    def SavePlot_calibration(self, y_pred, y_true, epoch):
        cls_num_list = self.loss_f.cls_num_list
        ratio = float(cls_num_list[1]) / np.sum(cls_num_list)
        y_true, y_pred = y_true.reshape(1,-1).squeeze(), y_pred.reshape(1,-1).squeeze()
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        predictions_ref = [ratio] * len(y_pred)
        BS = brier_score_loss(y_true, y_pred_sig)
        BS_ref = brier_score_loss(y_true, predictions_ref)
        BS_skill = 1 - float(BS)/BS_ref
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10, normalize=True)
        fig = pyplot.figure()
        pyplot.plot(mean_predicted_value, fraction_of_positives)
        pyplot.plot([0, 1], [0, 1], "k:")
        pyplot.title(f'Brier Skill Score {BS_skill}')
        pyplot.savefig('results/calibration'+ str(epoch) +'.png')

class LossesLogger(object):
    """
    Class definition for objects to write bata to log files in a
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

        header = ','.join(['Epoch', 'Train Loss', 'Valid Loss', 'AUROC'])
        self.logger.debug(header)

    def log(self, epoch, storer):
        """Write to the log file."""
        log_string = [epoch+1]
        for k in storer.keys():
            log_string += [sum(storer[k]) / len(storer[k])]
        log_string = ','.join(str(item) for item in log_string)
        self.logger.debug(log_string)
