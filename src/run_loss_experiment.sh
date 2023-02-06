#!/bin/sh -x

# Run SASREC with Cross Entropy loss
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --dropout 0.1 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1 --gpu '0' --loss_type 'Cross_Entropy'

# Run SASREC with TOP1 loss
python main.py --model_name SASRec --epoch 200 --early_stopping 20 --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --dropout 0.1 --history_max 20 --dataset 'Video_Games' --augment 'None' --train_ratio 1 --gpu '0' --loss_type 'TOP1'

