#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import csv
import collections
import pandas as pd
from sklearn.model_selection import train_test_split

with open('description_test.tsv') as f, open('test_tag.txt', 'a') as ftest:

    data = pd.read_csv(f, delimiter='\t')
    pic = data['pic']
    ids = data['ids']
    caption = data['caption']



    train.to_csv(ftrain, sep='\t', index=False,header=False)
    val.to_csv(fval, sep='\t', index=False,header=False)
    test.to_csv(ftest, sep='\t', index=False,header=False)

    ftest.close()
    f.close()
