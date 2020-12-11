# coding=utf-8
import os
import random
import numpy as np


def get_img_infos(img_info_txt):
    '''读取txt中存储的图片信息
    :param mode: train or test
    :param img_info_txt: 文件信息存储的txt路径
    :param label_name_to_num: 将字符串标签转为数字
    :return: 图片id信息和图片的标签(mode为train时不为空,mode为test时为空)
    '''
    # if mode not in ["train","test"]:
    # tang修改的代码(下替换上)
    with open(img_info_txt) as input_file:
        lines = input_file.readlines()
        img_name = [line.strip().split(',')[0] for line in lines]
        img_path = [line.strip().split(',')[1] for line in lines]
        img_label = [line.strip().split(',')[-1] for line in lines]

    return img_name, img_path, img_label


def split_train_tset(img_name, img_label):
    """
    函数功能: 先获得所有包名，根据包名按照一定比例划分数据集
    """
    # 保存腐腻名
    funi_names = []
    # 保存非腐腻名
    no_funi_names = []

    size = len(img_name)
    for i in range(size):
        label = img_label[i]
        bag_name, _ = img_name[i].split('_')
        if label == "funi":
            funi_names.append(bag_name)
        elif label == "no_funi":
            no_funi_names.append(bag_name)

    funi_names = np.unique(funi_names)
    no_funi_names = np.unique(no_funi_names)
    # 打乱顺序
    np.random.shuffle(funi_names)
    np.random.shuffle(no_funi_names)

    funi_names = list(funi_names)
    no_funi_names = list(no_funi_names)

    print('腐腻包的数量={},非腐腻包的数量={}'.format(len(funi_names),len(no_funi_names)))

    funi_train_len = int(len(funi_names)*0.5)
    nofuni_train_len = int(len(no_funi_names)*0.5)

    train = funi_names[0: funi_train_len]
    train.extend(no_funi_names[0: nofuni_train_len])

    test = funi_names[funi_train_len: ]
    test.extend(no_funi_names[nofuni_train_len: ])

    print('训练集的数量={},测试集的数量={}'.format(len(train), len(test)))

    return train, test


def generate_train_test_txt(save_train_txt_path, save_test_txt_path, train, test, img_name, img_path, img_label):
    train_ins_num = 0
    test_ins_num = 0

    # 训练集instance 数量
    with open(save_train_txt_path, mode="w", encoding="utf-8") as f_wirter:
        for i in range(len(img_name)):
            bag_name, _ = img_name[i].split('_')
            if bag_name in train:
                train_ins_num += 1
                f_wirter.write("%s,%s,%s\n" % (img_name[i], img_path[i], img_label[i]))
        f_wirter.close()

    with open(save_test_txt_path, mode="w", encoding="utf-8") as f_wirter:
        for i in range(len(img_name)):
            bag_name, _ = img_name[i].split('_')
            if bag_name in test:
                test_ins_num += 1
                f_wirter.write("%s,%s,%s\n" % (img_name[i], img_path[i], img_label[i]))
        f_wirter.close()

    print('训练集path数量={},测试集path数量={}'.format(train_ins_num, test_ins_num))

    return train_ins_num, test_ins_num


if __name__ == "__main__":
    img_name, img_path, img_label = get_img_infos("./data/image.txt")
    train, test = split_train_tset(img_name, img_label)
    generate_train_test_txt("./data/train.txt", "./data/test.txt", train, test, img_name, img_path, img_label)

    # # 获取原始大图图片名
    # funi_dataset, no_funi_dataset = get_img_dataset("E:/project/rename/funi_nofuni/data_original/funi",
    #                                                 "E:/project/rename/funi_nofuni/data_original/no_funi")
    # # 将腐腻数据和非腐腻数据分别平均分成5份
    # split_funi_dataset,split_no_funi_dataset = split(funi_dataset, no_funi_dataset)
    # # 将切分后的腐腻数据和非腐腻数据整合，对应位置相加
    # datasets = linkDataset(split_funi_dataset, split_no_funi_dataset)
    # # 四份整合成训练集，剩余一份作为测试集
    # dataset = generateDataset(datasets)
    # # 将特征数据分成训练集和测试集
    # split_feature("split_data/resnet_feature_sort.txt", "split_data")



