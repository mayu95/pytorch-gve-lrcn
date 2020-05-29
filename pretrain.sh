#!/bin/sh

# pretrain cub
# python main.py --model sc --dataset cub > train_cub&

# pretrain iu
# python main.py --model sc --dataset iu > ./log_pretrain/train_iu_orgin&

# 5.28 00:15  input_size = cub input size = cub image feature
# python main.py --model sc --dataset iu > ./log_pretrain/input_8192&

 # 5.28 11:50  input_8192, bilstm, epoch 20
# python main.py --model sc --dataset iu --num-epochs 20 > ./log_pretrain/input8192_bilstm_ep20&

 # 5.28 12:53   bilstm, epoch 20/10/30/50
# python main.py --model sc --dataset iu --num-epochs 20 > ./log_pretrain/bilstm_ep20&
# python main.py --model sc --dataset iu --num-epochs 10 > ./log_pretrain/bilstm_ep10&
# python main.py --model sc --dataset iu --num-epochs 30 > ./log_pretrain/bilstm_ep30&
# python main.py --model sc --dataset iu --num-epochs 50 > ./log_pretrain/bilstm_ep50&

 # 5.29 13:10  resnet50, bilstm, epoch 20/30/50
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 20 > ./log_pretrain/res50_bi_ep20&
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 30 > ./log_pretrain/res50_bi_ep30&
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 50 > ./log_pretrain/res50_bi_ep50&

    #  14:05  resnet50, bilstm, epoch 20/30/50, pretrained model change several lines
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 20 > ./log_pretrain/res50_pre_bi_ep20&
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 30 > ./log_pretrain/res50_pre_bi_ep30&
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 50 > ./log_pretrain/res50_pre_bi_ep50&


    #  16:25  resnet50 , bilstm, epoch 20, pretrained model change part 2
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 30 > ./log_pretrain/res50_pre2_bi_ep30&

    #  17:15  resnet50 , bilstm, epoch 20, pretrained model change part1 
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 30 > ./log_pretrain/res50_pre1_bi_ep30&

    #  17:55  resnet50, bilstm, epoch 50
# python main.py --model sc --dataset iu --pretrained-model resnet50 --num-epochs 50 > ./log_pretrain/res50_ep50_again&

    #  19:30  resnet152, bilstm, epoch 30/50
# python main.py --model sc --dataset iu --pretrained-model resnet152 --num-epochs 30 > ./log_pretrain/res152_ep30&
python main.py --model sc --dataset iu --pretrained-model resnet152 --num-epochs 50 > ./log_pretrain/res152_ep50&
