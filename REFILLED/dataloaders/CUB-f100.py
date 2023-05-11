# 修改CUB-200为能使用的代码，单独测试成功了
# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 19:22:12
"""

import pickle
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path, flag_mode, flag_tuning):
        super(MyDataset, self).__init__()
        self.root = data_path
        if(flag_mode == 'train'):
            self.is_train = True
        else:
            self.is_train = False

        self.transform_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),

            # transforms.Resize(32),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_simple = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),

            # transforms.Resize(32),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.images_path = {}
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        self.class_ids = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id
        
        self.data_id = []
        if self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if int(is_train):
                        self.data_id.append(image_id)
        if not self.is_train:
            with open(os.path.join(self.root, 'train_test_split.txt')) as f:
                for line in f:
                    image_id, is_train = line.split()
                    if not int(is_train):
                        self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)
    
    def __getitem__(self, index):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[index]
        class_id = int(self._get_class_by_id(image_id)) - 1
        path = self._get_path_by_id(image_id)
        image = cv2.imread(os.path.join(self.root, 'images', path))
        
        if self.is_train:
            image = self.transform_augment(image)
        else:
            image = self.transform_simple(image)
        return image, class_id

    def _get_path_by_id(self, image_id):

        return self.images_path[image_id]
    
    def _get_class_by_id(self, image_id):

        return self.class_ids[image_id]
    
    def get_n_classes(self):
        return 100


def generate_data_loader(data_path, flag_mode, flag_tuning, batch_size, n_workers):
    my_dataset = MyDataset(data_path, flag_mode, flag_tuning)
    # print(len(my_dataset))
    my_data_loader = DataLoader(my_dataset, batch_size, shuffle=True, num_workers=n_workers)
    return my_data_loader



# debug test
if __name__ == '__main__':
    data_path = '../../CUB-200'
    flag_mode = 'train'
    batch_size = 2
    n_workers = 0

    my_data_loader = generate_data_loader(data_path, flag_mode, True,batch_size, n_workers)
    for batch_index, batch in enumerate(my_data_loader):
        image, label = batch
        print(image.size())
        print(label.size())
        break

    print(my_data_loader.dataset.get_n_classes())