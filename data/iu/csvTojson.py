#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
#  import os
import csv
#  import pandas as pd

file_name = "description_train.tsv"
with open(file_name) as f, open("description_train.json", "a") as fw:
    #  data = pd.read_csv(f, delimiter='\t')

    json_descriptions = {}
    json_descriptions['images'] = []
    json_descriptions['annotations'] = []
    json_descriptions['type'] = 'captions'

    im_to_annotations = {}
    im_to_images = {}
    words = []

    # creat dictionaries
    for line in f:
        line = line.strip().split('\t')
        ID = line[0].strip()
        count = line[1].strip()
        caption = line[2].strip()         # donnot forget strip()
        #  count += 1
        #  images:[]
        im = {}
        im['file_name'] = ID
        im['id'] = ID
        json_descriptions['images'].append(im)
        #  annotations:[]
        descrip = {}
        descrip['caption'] = caption
        descrip['id'] = count
        descrip['image_id'] = ID
        json_descriptions['annotations'].append(descrip)

        # words for vocab
        caption = re.sub(r'\.|,|;|:', '', caption)
        words.extend(caption.split())    # words for vocab

    #  build vocab
    vocab = sorted(list(set(words)))
    vocab_file = open('vocab.txt', 'w')
    for v in vocab:
        vocab_file.writelines('%s\n' %v)
    vocab_file.close()

    json_str = json.dumps(json_descriptions)
    fw.write(json_str)

    fw.close()
    f.close()
