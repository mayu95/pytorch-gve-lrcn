#!/bin/sh


# evaluate cub -vgg16 -epoch 20
# python main.py --model gve --dataset cub --num-epochs 20 --sc-ckpt ./data/cub/sentence_classifier_ckpt.pth > cub_eval&


# evaluate iu origin
# python main.py --model gve --dataset iu --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/origin&

# 5.28 00:30 input size = 8192 = cub input size = cub image feature
# python main.py --model gve --dataset iu --pretrained-model vgg16  --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/input_size8192&

# 5.28 12:15  input8192, bilstm hid*4, sc ep20, gve ep30
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/input8192_bi_sc20&

# 5.28 13:15  bilstm hid*4, sc ep20/10/30/50, gve ep30
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc20&
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc10&
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc30&
# python main.py --model gve --dataset iu --pretrained-model vgg16  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc50&

# 5.29 13:20  resnet50, bilstm hid*4, sc ep20/30/50, gve ep30
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/bi_sc20&
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_bi_sc30&
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_bi_sc50&

    # 14:25  pretained model change, resnet50, bilstm hid*4, sc ep20/30/50, gve ep30/50
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_pre_bi_sc20&
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_pre_bi_sc30&
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 50 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_pre_bi_sc50&

    # 16:40  pretained model change part2, resnet50, bilstm hid*4, sc ep30/50, gve ep50
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 50 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_pre2_bi_sc30&

    # 17:25  pretained model change part1, resnet50, bilstm hid*4, sc ep30/50, gve ep50
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 50 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_pre1_bi_sc30&

    # 18:05  pretained model change, resnet50, bilstm hid*4, sc ep50, gve ep50
# python main.py --model gve --dataset iu --pretrained-model resnet50  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res50_bi_sc50_ep30&

    # 18:45  resnet152, bilstm hid*4, sc ep30/50, gve ep30
# python main.py --model gve --dataset iu --pretrained-model resnet152  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res152_bi_sc30&
python main.py --model gve --dataset iu --pretrained-model resnet152  --num-epochs 30 --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/res152_bi_sc50&










