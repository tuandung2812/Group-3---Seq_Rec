#!/bin/sh -x

# Run GRU4REC with full data
python main.py --model_name GRU4Rec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1 --gpu '0'

# Run GRU4REC with 0.1 data
python main.py --model_name GRU4Rec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 0.1 --gpu '0'

# Run GRU4REC with 0.25 data
python main.py --model_name GRU4Rec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 0.25 --gpu '0'

# Run GRU4REC with 0.5 data
python main.py --model_name GRU4Rec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 0.5 --gpu '0'

# python main.py --model_name FPMC --epoch 2000 --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Video_Games'
# --augment None --train_ratio 1