#!/bin/sh


# evaluate cub -vgg16 -epoch 20
# python main.py --model gve --dataset cub --num-epochs 20 --sc-ckpt ./data/cub/sentence_classifier_ckpt.pth > cub_eval&


# evaluate iu origin
# python main.py --model gve --dataset iu --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/origin&

# 5.28 00:30 input size = 8192 = cub input size = cub image feature
# python main.py --model gve --dataset iu --pretrained-model vgg16  --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/input_size8192&

# 5.28 12:15  input8192, bilstm hid*4, sc ep20, gve ep30
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/input8192_bi_sc20&

# 5.28 13:15  bilstm hid*4, sc ep20, gve ep30
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc20&
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc10&
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc30&
python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc50&
