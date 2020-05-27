import os
from collections import Counter
from enum import Enum
import pickle
import json
from PIL import Image

import torch
import torch.utils.data as data
import numpy as np
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
#import nltk

from utils.vocabulary import Vocabulary
from utils.tokenizer.ptbtokenizer import PTBTokenizer

from .coco_dataset import CocoDataset

# Adapted from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class IUDataset(CocoDataset):

    """IU Custom Dataset compatible with torch.utils.data.DataLoader."""

    dataset_prefix = 'iu'
    image_path = 'iu_xray_images'
    #  image_features_path = 'CUB_feature_dict.p'
    caption_path = 'description_{}.json'
    #  caption_path = '{}_16images.json'
    vocab_file_name = 'iu_vocab.pkl'
    tokens_file_name = 'iu_tokens_{}.pkl'
    class_labels_path = 'uni_major_tag.json'

    # Available data splits (must contain 'train')
    DATA_SPLITS = set(['train', 'val', 'test'])

    def __init__(self, root, split='train', vocab=None, tokenized_captions=None,
            transform=None, use_image_features=True):
        super().__init__(root, split, vocab, tokenized_captions, transform)

        cls = self.__class__
        #  self.img_features_path= os.path.join(self.root, cls.image_features_path)

        #  if use_image_features:
            #  self.load_img_features(self.img_features_path)
            #  self.input_size = next(iter(self.img_features.values())).shape[0]
        self.transform = transform
        #  self.input_size = 86016 
        self.input_size = 1000 


    #  def load_img_features(self, img_features_path):
        #  with open(img_features_path, 'rb') as f:
            #  feature_dict = pickle.load(f, encoding='latin1')
        #  self.img_features = feature_dict


    def load_class_labels(self, class_labels_path):
        with open(class_labels_path, 'rb') as f:
            label_dict = json.load(f, encoding='latin1')

        #  set label str to int
        word_list = set(label_dict.values())
        word2idx = dict([(w,i) for i, w in enumerate(word_list)])
        label_to_num = label_dict.copy()
        for k,v in label_to_num.items():
            label_to_num[k]= word2idx[v]

        #  self.num_classes = len(set(label_dict.values()))
        #  self.class_labels = label_dict
        self.num_classes = len(set(label_to_num.values()))
        self.class_labels = label_to_num

    def get_image(self, img_id):
        #  if self.img_features is not None:
            #  image = self.img_features[img_id]
            #  image = torch.Tensor(image)
        #  else:
            #  image = super().get_image(img_id)

        path = img_id 
        image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
        if self.transform is not None:
            #  print(self.transform)
            image = self.transform(image)
            #  print('is transformed!!!!!!!!!!!!!!!!!!!!!!!1')
        #  print(type(image))
        #  exit(0)
        return image

    def get_class_label(self, img_id):
        #  class_label = torch.LongTensor([int(self.class_labels[img_id])-1])
        class_label = torch.LongTensor([int(self.class_labels[img_id])])
        return class_label
