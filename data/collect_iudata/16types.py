#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import csv
import collections

file_name = "iu_xray_major_tags.json"
with open(file_name) as f, open("uni_major_tag.csv", "a") as fw, open("uni_major_tag_dic.txt", "a") as fdic, open("uni_major_tag.json", "a") as fjson:
#  with open(file_name) as f, open("unique_major_tags.csv", "a") as fw:
    writer = csv.writer(fw, delimiter='\t')
    writer.writerow(['pic_id', "label"])
    fdic.write("Label Name" + '\t' + "Number" + '\n')

    data = json.load(f)
    name_list = ['Calcified Granuloma', 'Thoracic Vertebrae/degenerative', 'Lung/hypoinflation',
            'Calcinosis', 'Opacity', 'Spine/degenerative', 'Cardiomegaly', 'Scoliosis', 'normal',
            'Lung/hyperdistention', 'Surgical Instruments', 'Spondylosis', 'Osteophyte',
            'Catheters, Indwelling', 'Fractures, Bone', 'Granulomatous Disease', 'Nodule']
    label_dic = {}
    image_dic = {}

    for key, labels in data.items():
        if len(labels) == 1:    #  only one label
            label = ''
            for i in range(16, -1,-1):
                if str(name_list[i]).lower() in str(labels).lower():
                    label = str(name_list[i])
                    writer.writerow([key, label])
                    image_dic[key] = label 
                    label_dic[label] = label_dic.setdefault(label,0) + 1
                    break

    label_dic = sorted(label_dic.items(), key=lambda x:x[1], reverse=True)
    label_dic = collections.OrderedDict(label_dic)
    for i,j in label_dic.items():
        allwrite = i + '\t' + str(j) + '\n'
        fdic.write(allwrite)

    json_str = json.dumps(image_dic)
    fjson.write(json_str)

    fdic.close()
    fw.close()
    fjson.close()
