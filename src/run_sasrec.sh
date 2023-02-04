#!/bin/sh -x

# Run GRU4REC with full data
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1

# Run SASRec with 0.1 data
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1

# Run SASRec with 0.25 data
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1

# Run SASRec with 0.5 data
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1

