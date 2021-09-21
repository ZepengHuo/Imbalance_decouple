#!/bin/bash

for z in 16 32 64 128;
do
    for H in 64 128 256 512;
    do
        echo $z 
        echo $H
        python main.py gru ../flexible-ehr/data/root_dicts -D --epochs 5 --g 2 -m pheno -bs 64 --p-dropout 0.1 -z $z -H $H
    done
done