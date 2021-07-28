#!/bin/bash

for z in 3 10 30 100 300;
do
    for H in 0 1;
    do
        for b in 0 1;
        do
            echo $z 
            echo $H
            echo $b
            python main.py gru ../flexible-ehr/data/root_dicts --epochs 5 --g 2 --lr 0.001 --p-dropout 0.5 -s_constant $z -rand_ldam $H -bal_ldam $b
        done
    done
done
