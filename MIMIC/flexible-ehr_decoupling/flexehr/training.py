"""Training module."""

import logging
import numpy as np
import os
from numpy.core.numeric import False_
import torch

from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from timeit import default_timer
from tqdm import trange

from flexehr.utils.modelIO import save_model
from utils.helpers import array
from matplotlib import pyplot

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
        self.args = args
        self.weight = None # weight for loss function

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

            # freeze backbone, lower lr for classifier after first epoch
            if self.args.annealing_lr and epoch == 1:
                for p in self.model.gru.parameters():
                        p.requires_grad = False
            
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * self.args.annealing_lr
                    

                if self.args.train_rule == 'DRW':
                    
                    idx = epoch // 1
                    betas = [0, 0.9999]
                    effective_num = 1.0 - np.power(betas[idx], self.loss_f[-1].cls_num_list)
                    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.loss_f[-1].cls_num_list)
                    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.args.gpu_num)
                    self.weight = per_cls_weights                
            
            self.model.train()
            t_loss = self._train_epoch(train_loader_pos, 
                                       train_loader_neg, 
                                       train_loader, 
                                       storer)
            
            self.model.eval()
            v_loss = self._valid_epoch(valid_loader_pos, 
                                       valid_loader_neg, 
                                       valid_loader, 
                                       epoch, 
                                       storer)

            self.logger.info(f'Train loss {t_loss:.4f}')
            self.logger.info(f'Valid loss {v_loss:.4f}')
            self.logger.info(f'Valid bal auroc {storer["auroc_bal"][0]:.4f}')
            self.logger.info(f'Valid rand auroc {storer["auroc_rand"][0]:.4f}')
            self.logger.info(f'Valid bal auprc {storer["auprc_bal"][0]:.4f}')
            self.logger.info(f'Valid rand auprc {storer["auprc_rand"][0]:.4f}')
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
                
                
                # balanced sampler
                if_main = True
                y_pred_neg = self.model(X_neg, if_main)
                
                iter_loss_neg = self.loss_f[self.args.bal_ldam](y_pred_neg, y_neg, 
                                             self.model.training, if_main, self.weight, storer)

                y_pred_pos = self.model(X_pos, if_main=if_main)
                
                iter_loss_pos = self.loss_f[self.args.bal_ldam](y_pred_pos, y_pos, 
                                             self.model.training, if_main, self.weight, storer)
                

                # random sampler
                if_main = False
                y_pred = self.model(data, if_main=if_main)

                iter_loss_rand = self.loss_f[self.args.rand_ldam](y_pred, y_true, 
                                             self.model.training, if_main, self.weight, storer)
                #print(y_pred)
                
                #iter_loss_balance = iter_loss_pos * self.model.coef_pos.expand_as(iter_loss_pos) + \
                #                    iter_loss_neg * self.model.coef_neg.expand_as(iter_loss_neg)
                
                coef_neg = 1-self.model.coef_pos
                coef_neg = coef_neg.clone().detach().requires_grad_(True)
                
                iter_loss_balance = iter_loss_pos * self.model.coef_pos.expand_as(iter_loss_pos) + \
                                    iter_loss_neg * coef_neg.expand_as(iter_loss_neg)
                
                
                rand_loss_r = 1-self.model.bal_loss_r
                rand_loss_r = rand_loss_r.clone().detach().requires_grad_(True)
                
                #iter_loss = iter_loss_rand * self.model.rand_loss_r.expand_as(iter_loss_rand) + \
                #            iter_loss_balance * self.model.bal_loss_r.expand_as(iter_loss_balance)
                iter_loss = iter_loss_rand * rand_loss_r.expand_as(iter_loss_rand) + \
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
                     data_loader, epoch, storer=defaultdict(list)):
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
                
                iter_loss_rand = self.loss_f[self.args.rand_ldam](
                    y_pred, y_true, self.model.training, if_main, self.weight, storer)
                
                
                if_main = True
                y_pred_pos = self.model(X_pos, if_main)
                
                iter_loss_pos = self.loss_f[self.args.bal_ldam](y_pred_pos, y_pos, 
                                             self.model.training, if_main, self.weight, storer)
                
                y_pred_neg = self.model(X_neg, if_main)
                
                iter_loss_neg = self.loss_f[self.args.bal_ldam](y_pred_neg, y_neg, 
                                             self.model.training, if_main, self.weight, storer)
                
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

        metrics = self.compute_metrics_auroc(y_preds_bal, y_trues_bal, if_bal=True)
        storer.update(metrics)
        
        metrics = self.compute_metrics_auroc(y_preds_rand, y_trues_rand, if_bal=False)
        storer.update(metrics)
        
        metrics = self.compute_metrics_auprc(y_preds_bal, y_trues_bal, if_bal=True)
        storer.update(metrics)
        
        metrics = self.compute_metrics_auprc(y_preds_rand, y_trues_rand, if_bal=False)
        storer.update(metrics)
        
        
        self.SavePlot_auroc(y_preds_bal, y_trues_bal, epoch, if_bal=True)
        self.SavePlot_auroc(y_preds_rand, y_trues_rand, epoch, if_bal=False)
        
        self.SavePlot_auprc(y_preds_bal, y_trues_bal, epoch, if_bal=True)
        self.SavePlot_auprc(y_preds_rand, y_trues_rand, epoch, if_bal=False)
        
        self.SavePlot_brier(y_preds_bal, y_trues_bal, epoch, if_bal=True)
        self.SavePlot_brier(y_preds_rand, y_trues_rand, epoch, if_bal=False)
        
        self.SavePlot_calibration(y_preds_bal, y_trues_bal, epoch, if_bal=True)
        self.SavePlot_calibration(y_preds_rand, y_trues_rand, epoch, if_bal=False)
        
        
        if not os.path.isdir('saved'):
            os.makedirs('saved')
        with open(f'saved/y_preds_bal{epoch}.npy', 'wb') as f:
            np.save(f, y_preds_bal)
        with open(f'saved/y_trues_bal{epoch}.npy', 'wb') as f:
            np.save(f, y_trues_bal)
        with open(f'saved/y_preds_rand{epoch}.npy', 'wb') as f:
            np.save(f, y_preds_rand)
        with open(f'saved/y_trues_rand{epoch}.npy', 'wb') as f:
            np.save(f, y_trues_rand)    

        return epoch_loss / len(data_loader)

    def compute_metrics_auroc(self, y_pred, y_true, if_bal=False):
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
    
    def compute_metrics_auprc(self, y_pred, y_true, if_bal=False):
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
            metrics['auprc_bal'] = [average_precision_score(y_true, y_pred)]
        else:
            metrics['auprc_rand'] = [average_precision_score(y_true, y_pred)]
        
        return metrics
    
    def SavePlot_auroc(self, y_pred, y_true, epoch, if_bal=False):
        y_true, y_pred = y_true.reshape(1,-1), y_pred.reshape(1,-1)
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        fpr, tpr, _ = roc_curve(np.squeeze(y_true), np.squeeze(y_pred_sig))
        fig = pyplot.figure()
        pyplot.plot(fpr, tpr, marker='.')
        if if_bal:
            bal = 'bal'
        else:
            bal = 'rand'
        pyplot.savefig('results/auroc'+bal+ str(epoch) +'.png')
    
    def SavePlot_auprc(self, y_pred, y_true, epoch, if_bal=False):
        y_true, y_pred = y_true.reshape(1,-1), y_pred.reshape(1,-1)
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        lr_precision, lr_recall, _ = precision_recall_curve(np.squeeze(y_true), np.squeeze(y_pred_sig))
        fig = pyplot.figure()
        pyplot.plot(lr_recall, lr_precision, marker='.')
        if if_bal:
            bal = 'bal'
        else:
            bal = 'rand'
        pyplot.savefig('results/auprc'+bal+ str(epoch) +'.png')
        
        
    def SavePlot_brier(self, y_pred, y_true, epoch, if_bal=False):
        cls_num_list = self.loss_f[self.args.bal_ldam].cls_num_list
        ratio = float(cls_num_list[1]) / np.sum(cls_num_list)
        y_true, y_pred = y_true.reshape(1,-1).squeeze(), y_pred.reshape(1,-1).squeeze()
        predictions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        predictions_ref = [ratio] * len(predictions)
        testy = y_true
        BS = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions]
        BS_ref = [brier_score_loss(testy, [y for x in range(len(testy))]) for y in predictions_ref]
        BS_skill = 1 - np.array(BS)/np.array(BS_ref)
        
        fig = pyplot.figure()
        pyplot.plot(predictions, BS_skill)
        if if_bal:
            bal = 'bal'
        else:
            bal = 'rand'
        pyplot.savefig('results/brier'+bal+ str(epoch) +'.png')
        
    def SavePlot_calibration(self, y_pred, y_true, epoch, if_bal):
        cls_num_list = self.loss_f[self.args.bal_ldam].cls_num_list
        ratio = float(cls_num_list[1]) / np.sum(cls_num_list)
        y_true, y_pred = y_true.reshape(1,-1).squeeze(), y_pred.reshape(1,-1).squeeze()
        y_pred_sig = 1/(1 + np.exp(-y_pred))
        predictions_ref = [ratio] * len(y_pred)
        BS = brier_score_loss(y_true, y_pred_sig)
        BS_ref = brier_score_loss(y_true, predictions_ref)
        BS_skill = 1 - float(BS)/BS_ref

        y_true, y_pred = y_true.reshape(1,-1).squeeze(), y_pred.reshape(1,-1).squeeze()
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, n_bins=10, normalize=True)
        fig = pyplot.figure()
        pyplot.plot(mean_predicted_value, fraction_of_positives)
        pyplot.plot([0, 1], [0, 1], "k:")
        if if_bal:
            bal = 'bal'
        else:
            bal = 'rand'
        pyplot.title(f'Brier Skill Score {BS_skill}')
        pyplot.savefig('results/calibration'+bal+ str(epoch) +'.png')

    def ErrorRateAt95Recall1(labels, scores):
        recall_point = 0.95
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        # Sort label-score tuples by the score in descending order.
        indices = np.argsort(scores)[::-1]    #降序排列
        sorted_labels = labels[indices]
        sorted_scores = scores[indices]
        n_match = sum(sorted_labels)
        n_thresh = recall_point * n_match
        thresh_index = np.argmax(np.cumsum(sorted_labels) >= n_thresh)
        FP = np.sum(sorted_labels[:thresh_index] == 0)
        TN = np.sum(sorted_labels[thresh_index:] == 0)
        return float(FP) / float(FP + TN)




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
