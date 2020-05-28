#!/bin/sh

# pretrain cub
# python main.py --model sc --dataset cub > train_cub&

# pretrain iu
# python main.py --model sc --dataset iu > ./log_pretrain/train_iu_orgin&

# 5.28 00:15  input_size = cub input size = cub image feature
# python main.py --model sc --dataset iu > ./log_pretrain/input_8192&

 # 5.28 11:50  input_8192, bilstm, epoch 20
python main.py --model sc --dataset iu --num-epochs 20 > ./log_pretrain/input8192_bilstm_ep20&
