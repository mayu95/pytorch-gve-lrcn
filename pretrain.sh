#!/bin/sh

# pretrain cub
# python main.py --model sc --dataset cub > train_cub&

# pretrain iu
python main.py --model sc --dataset iu > ./log_pretrain/train_iu_orgin&

