#!/bin/sh


# evaluate iu origin
python main.py --model gve --dataset iu --sc-ckpt ./data/iu/sentence_classifier_ckpt.pth > ./log_eval/origin&


