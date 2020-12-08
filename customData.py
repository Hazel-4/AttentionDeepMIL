from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms

import time
import os
from torch.utils.data import Dataset

from PIL import Image


"""
    加载图片文件
"""


# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))


# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, txt_path, data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [line.strip().split(',')[0] for line in lines]
            self.img_path = [line.strip().split(',')[1] for line in lines]
            self.img_label = [line.strip().split(',')[-1] for line in lines]
        self.data_transforms = data_transforms
        self.loader = loader

        # img_name, img_label 映射成数字
        for i in range(len(self.img_name)):
            # img_name 映射成数字
            self.img_name[i], _ = self.img_name[i].split('_')
            var = str(ord(self.img_name[i][0]) - ord('A'))
            self.img_name[i] = int(var + self.img_name[i][1:])

            # img_label 映射成数字
            if self.img_label[i] == "funi":
                self.img_label[i] = 1
            else:
                self.img_label[i] = 0

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img_path = self.img_path[item]
        label = self.img_label[item]
        img = self.loader(img_path)

        if self.data_transforms is not None:
            img = self.data_transforms(img)
            # try:
            #     img = self.data_transforms[self.dataset](img)
            # except:
            #     print("Cannot transform image: {}".format(img_name))
        return img_name, img, label

if __name__ == "__main__":
    loader = customData(txt_path='./data/image.txt',
               data_transforms=transforms.Compose([
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor()
                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
               ]))
    dataloaders = torch.utils.data.DataLoader(loader, batch_size=32, shuffle=True)
    # print(dataloaders)
    for (batch_name, batch_data, batch_labels) in dataloaders:
        print(len(batch_data))
