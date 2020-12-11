"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from customData import customData

import util_data

class MnistBags(data_utils.Dataset):
    def __init__(self, bag_name, ins_num, train=True):
        self.bag_name = bag_name
        self.ins_num = ins_num
        self.train = train

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(customData(txt_path='./data/train.txt',
                                           data_transforms=transforms.Compose([
                                               transforms.RandomResizedCrop(28),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])),
                                           batch_size=self.ins_num,
                                           shuffle=True)
        else:
            loader = data_utils.DataLoader(customData(txt_path='./data/test.txt',
                                           data_transforms=transforms.Compose([
                                               transforms.RandomResizedCrop(28),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor()
                                               # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                           ])),
                                           batch_size=self.ins_num,
                                           shuffle=False)

        for batch_name, batch_data, batch_labels in loader:
            all_name = batch_name
            all_imgs = batch_data
            all_labels = batch_labels
        # all_name.reshape(len(all_name), 1, 224, 224)


        bags_list = []
        labels_list = []


        for i in range(len(self.bag_name)):
            indices = []
            for j in range(len(all_name)):
                name, _ = all_name[j].split("_")
                if self.bag_name[i] == name:
                    indices.append(j)

            labels_in_bag = all_labels[indices]
            bags_list.append(all_imgs[indices])

            labels_list.append(labels_in_bag)
        return bags_list, labels_list


    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label =  [max(self.train_labels_list[index]), self.train_labels_list[index]]  # 最后的label为[bag_label, labels_list]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":
    img_name, img_path, img_label = util_data.get_img_infos("./data/image.txt")
    train_bag_name, test_bag_name = util_data.split_train_tset(img_name, img_label)
    num_in_train, num_in_test = util_data.generate_train_test_txt("./data/train.txt", "./data/test.txt",
                                                                            train_bag_name, test_bag_name,
                                                                            img_name, img_path, img_label)
    train_loader = data_utils.DataLoader(MnistBags(bag_name=train_bag_name,
                                                   ins_num=num_in_train,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(bag_name=test_bag_name,
                                                   ins_num=num_in_test,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    # len_bag_list_train = []
    # mnist_bags_train = 0
    # for batch_idx, (bag, label) in enumerate(train_loader):
    #     len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
    #     mnist_bags_train += label[0].numpy()[0]
    # print('Number positive train bags: {}/{}\n'
    #       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
    #     mnist_bags_train, len(train_loader),
    #     np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))
    #
    # len_bag_list_test = []
    # mnist_bags_test = 0
    # for batch_idx, (bag, label) in enumerate(test_loader):
    #     len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
    #     mnist_bags_test += label[0].numpy()[0]
    # print('Number positive test bags: {}/{}\n'
    #       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
    #     mnist_bags_test, len(test_loader),
    #     np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
