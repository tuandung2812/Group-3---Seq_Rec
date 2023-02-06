#!/bin/sh -x

# Run POP
python main.py --model_name POP --train 0 --dataset 'Video_Games' --augment None --train_ratio 1
