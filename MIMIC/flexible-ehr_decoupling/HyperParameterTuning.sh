#!/bin/bash


# python main.py gru ../flexible-ehr/data/root_dicts --epochs 5 --g 2 --lr 0.001 --p-dropout 0.5 -rand_ldam 1 -bal_ldam 1 -train_rule DRW

#:'
for r in 0;
do
    for b in 2;
    do
        echo $r
        echo $b
        python main.py gru ../flexible-ehr/data/root_dicts --epochs 5 --g 2 --lr 0.001 --p-dropout 0.5 -rand_ldam $r -bal_ldam $b -annealing_lr 0.1 -train_rule DRW -bs 40
    done
done

#'


