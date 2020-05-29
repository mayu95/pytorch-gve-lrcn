#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import csv
import collections
import pandas as pd
from sklearn.model_selection import train_test_split

file_name = "all_image.csv"
with open(file_name) as f, open('description_train.tsv', 'a') as ftrain, open("description_val.tsv", "a") as fval, open('description_test.tsv', 'a') as ftest:

    data = pd.read_csv(f, delimiter='\t')
    pic = data['pic']
    ids = data['ids']
    label = data['label']
    caption = data['caption']

    train, val_test, train_label, val_test_label = train_test_split(data,label,test_size=0.2,stratify=data['label'],random_state=1)
    val, test, val_label, test_label = train_test_split(val_test,val_test_label,test_size=0.5,random_state=1)

    train.pop('label')
    val.pop('label')
    test.pop('label')

    train.to_csv(ftrain, sep='\t', index=False,header=False)
    val.to_csv(fval, sep='\t', index=False,header=False)
    test.to_csv(ftest, sep='\t', index=False,header=False)

    ftrain.close()
    ftest.close()
    fval.close()
    f.close()
