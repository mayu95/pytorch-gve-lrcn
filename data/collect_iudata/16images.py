#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import csv
import collections
import pandas as pd

file_name = "all_images.tsv"
dic_file_name = 'uni_major_tag.json'
with open(file_name) as f, open(dic_file_name) as fdic, open("all_image.csv", "a") as fw:
    writer = csv.writer(fw, delimiter='\t')
    writer.writerow(['pic','label', 'ids', "Caption"])

    data = json.load(fdic)
    all_dic = {}
    image_dic = {}
    n = 0
    ids = 0
    label = ''

    for line in f:
        line = line.strip().split('\t')
        pic = line[0]
        caption = line[1].strip()         # donnot forget strip()
        
        if pic in data.keys():
            ids += 1
            label = data[pic]

            writer.writerow([pic,label,ids, caption])

    #  json_str = json.dumps(image_dic)
    #  fjson.write(json_str)

    fdic.close()
    fw.close()
    f.close()
