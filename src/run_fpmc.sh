#!/bin/sh -x

# Run FPMC with full data
python main.py --model_name FPMC --epoch 500 --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Video_Games' --augment None --train_ratio 1

# Run FPMC with 0.1 data
python main.py --model_name FPMC --epoch 500 --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Video_Games' --augment None --train_ratio 0.1

# Run FPMC with 0.25 data
python main.py --model_name FPMC --epoch 500 --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Video_Games' --augment None --train_ratio 0.25

# Run FPMC with 0.5 data
python main.py --model_name FPMC --epoch 500 --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Video_Games' --augment None --train_ratio 0.5
