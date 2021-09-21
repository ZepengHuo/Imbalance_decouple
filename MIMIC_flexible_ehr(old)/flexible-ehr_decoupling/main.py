"""Main training module."""

import argparse
import logging
import numpy as np
import os
import sys
import torch

from flexehr.training import Trainer
from flexehr.training_pheno import Trainer as Trainer_pheno
from flexehr.utils.modelIO import load_metadata, load_model, save_model
from flexehr.models.losses import BCE, BCEWithLogitsLoss
from flexehr.models.models import MODELS, init_model
from utils.datasets import get_dataloaders
from utils.helpers import (get_n_param, new_model_dir, set_seed,
                           FormatterNoDuplicate)


def parse_arguments(args_to_parse):
    """Parse command line arguments."""
    desc = 'Pytorch implementation and evaluation of flexible EHR embedding.'
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name',
                         type=str,
                         help='Name of the model for storing and loading.')
    general.add_argument('-r', '--results',
                         type=str, default='results',
                         help='Directory to store results.')
    general.add_argument('--p-bar',
                         action='store_true', default=True,
                         help='Show progress bar.')
    general.add_argument('--cuda',
                         action='store_true', default=True,
                         help='Whether to use CUDA training.')
    general.add_argument('-s', '--seed',
                         type=int, default=0,
                         help='Random seed. `None` for stochastic behavior.')
    general.add_argument('-g', '--gpu_num',
                         type=int, default=1,
                         help='which gpu device to use')
    
    # Learning options
    training = parser.add_argument_group('Training options')
    training.add_argument('data',
                          type=str,
                          help='Path to data directory')
    training.add_argument('-e', '--epochs',
                          type=int, default=20,
                          help='Maximum number of epochs.')
    training.add_argument('-bs',
                          type=int, default=128,
                          help='Batch size for training.')
    training.add_argument('--lr',
                          type=float, default=5e-4,
                          help='Learning rate.')
    training.add_argument('--early-stopping',
                          type=int, default=5,
                          help='Epochs before early stopping.')
    training.add_argument('--coef_pos',
                          type=float, default=0.5,
                          help='ratio for pos class in loss')
    training.add_argument('--bal_loss_r',
                          type=float, default=0.5,
                          help='ratio of loss for balanced classifier')

    # Model options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default='Mortality', choices=MODELS,
                       help='Type of decoder to use.')
    model.add_argument('-t', '--t_hours',
                       type=int, default=48,
                       help='ICU data time length.')
    model.add_argument('-n', '--n_bins',
                       type=int, default=20,
                       help='Number of bins per continuous variable.')
    model.add_argument('--dt',
                       type=float, default=1.0,
                       help='Time increment between sequence steps.')
    model.add_argument('-z', '--latent-dim',
                       type=int, default=32,
                       help='Dimension of the token embedding.')
    model.add_argument('-H', '--hidden-dim',
                       type=int, default=256,
                       help='Dimension of the LSTM hidden state.')
    model.add_argument('-p', '--p-dropout',
                       type=float, default=0.0,
                       help='Embedding dropout rate.')
    model.add_argument('-w', '--weighted',
                       type=bool, default=True,
                       help='Whether to weight embeddings.')
    model.add_argument('-D', '--dynamic',
                       default=True, action='store_false',
                       help='Whether to perform dynamic prediction.')

    # Evaluation options
    evaluation = parser.add_argument_group('Evaluation options')
    evaluation.add_argument('--eval',
                            action='store_true', default=False,
                            help='Whether to evaluate using pretrained model.')
    evaluation.add_argument('--test',
                            action='store_true', default=True,
                            help='Whether to compute test losses.')

    args = parser.parse_args(args_to_parse)

    return args


def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """

    # Logging info
    formatter = logging.Formatter('%(asctime)s %(levelname)s - '
                                  '%(funcName)s: %(message)s',
                                  '%H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    stream = logging.StreamHandler()
    stream.setLevel('INFO')
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = torch.device(
        'cuda:{}'.format(str(args.gpu_num)) if torch.cuda.is_available() and args.cuda else 'cpu')
    if torch.cuda.is_available():
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(args.gpu_num)
    model_name = f'{args.name}_lr{args.lr}_z{args.latent_dim}' \
                 + f'_h{args.hidden_dim}_p{args.p_dropout}'
    model_dir = os.path.join(args.results, model_name)
    logger.info(f'Directory for saving and loading models: {model_dir}')
    if not args.eval:
        # Model directory
        new_model_dir(model_dir, logger=logger)
        # Dataloaders
        if args.model_type != 'pheno':
            train_loader_pos, train_loader_neg, train_loader, \
            valid_loader_pos, valid_loader_neg, valid_loader = get_dataloaders(
                args.data, args.t_hours, args.n_bins,
                validation=True, task=args.model_type, dynamic=args.dynamic,
                batch_size=args.bs, logger=logger)
        else:
            train_loader_bal, train_loader_rand, \
            valid_loader_bal, valid_loader_rand = get_dataloaders(
                args.data, args.t_hours, args.n_bins,
                validation=True, task=args.model_type, dynamic=args.dynamic,
                batch_size=args.bs, logger=logger)
            train_loader = train_loader_rand
        logger.info(
            f'Train {args.model_type}-{args.t_hours} ' +
            f'with {len(train_loader.dataset)} samples')

        # Load model
        n_tokens = len(np.load(
            os.path.join(
                args.data, '_dicts', f'{args.t_hours}_{args.n_bins}.npy'),
            allow_pickle=True).item())
        model = init_model(
            args.model_type, n_tokens, args.latent_dim, args.hidden_dim,
            args.coef_pos, args.bal_loss_r,
            p_dropout=args.p_dropout, dt=args.dt,
            weighted=args.weighted, dynamic=args.dynamic)
        logger.info(f'#params in model: {get_n_param(model)}')

        # Optimizer
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
        

        # Training
        if args.model_type != 'pheno':
            loss_f = BCE()
            trainer = Trainer(
                model, loss_f, args, optimizer,
                device=device, logger=logger, save_dir=model_dir, p_bar=args.p_bar
                , coef_pos=args.coef_pos, bal_loss_r=args.bal_loss_r)
            trainer.train(
                train_loader_pos, train_loader_neg, train_loader, valid_loader_pos, valid_loader_neg, valid_loader,
                epochs=args.epochs, early_stopping=args.early_stopping)
        else:
            loss_f = BCEWithLogitsLoss()
            trainer = Trainer_pheno(
                model, loss_f, args, optimizer, 
                device=device, logger=logger, save_dir=model_dir, p_bar=args.p_bar
                , coef_pos=args.coef_pos, bal_loss_r=args.bal_loss_r)
            trainer.train(
                train_loader_bal, train_loader_rand, valid_loader_bal, valid_loader_rand,
                epochs=args.epochs, early_stopping=args.early_stopping)

        # Save model
        metadata = vars(args)
        metadata['n_tokens'] = n_tokens
        save_model(trainer.model, model_dir, metadata=metadata)

    if args.test:
        # Load model
        model = load_model(model_dir, is_gpu=args.cuda)
        metadata = load_metadata(model_dir)

        # Dataloader
        test_loader, _ = get_dataloaders(
            metadata['data'], metadata['t_hours'], metadata['n_bins'],
            validation=False, dynamic=metadata['dynamic'], batch_size=128,
            shuffle=False, logger=logger)

        # Evaluate
        loss_f = BCE()
        evaluator = Trainer(
            model, loss_f,
            device=device, logger=logger, save_dir=model_dir, p_bar=args.p_bar)
        evaluator._valid_epoch(test_loader)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
