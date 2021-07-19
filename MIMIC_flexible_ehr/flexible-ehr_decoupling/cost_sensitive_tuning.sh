#!/bin/bash

python main.py gru ../flexible-ehr/data/root_dicts --lr 0.001 --p-dropout 0.5 --bal_loss_r 0.55 --epochs 5 --g 2
#python main.py gru ../flexible-ehr/data/root_dicts --lr 0.001 --p-dropout 0.5 --bal_loss_r 0.6 --epochs 5
#python main.py gru ../flexible-ehr/data/root_dicts --lr 0.001 --p-dropout 0.5 --bal_loss_r 0.65 --epochs 5
#python main.py gru ../flexible-ehr/data/root_dicts --lr 0.001 --p-dropout 0.5 --bal_loss_r 0.7 --epochs 5
#python main.py gru ../flexible-ehr/data/root_dicts --lr 0.001 --p-dropout 0.5 --bal_loss_r 0.75 --epochs 5