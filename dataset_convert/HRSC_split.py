# 将标签格式为xml的数据集按照8：2的比例划分为训练集和验证集

import os
import shutil
import random
from tqdm import tqdm


def split_img(img_path, label_path, split_list):
    # 定义数据集路径
    Data = 'E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset'
    train_img_dir =  r'E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\images\train'
    val_img_dir = r'E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\images\val'
    train_label_dir = r'E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\labels\train'
    val_label_dir = r'E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\labels\val'

    try:  # 创建数据集文件夹
        os.mkdir(Data)
        # 创建文件夹
        os.makedirs(train_img_dir)
        os.makedirs(train_label_dir)
        os.makedirs(val_img_dir)
        os.makedirs(val_label_dir)
    except:
        print('文件目录已存在')

    train, val = split_list
    all_img = os.listdir(img_path)
    all_img_path = [os.path.join(img_path, img) for img in all_img]
    # all_label = os.listdir(label_path)
    # all_label_path = [os.path.join(label_path, label) for label in all_label]
    train_img = random.sample(all_img_path, int(train * len(all_img_path)))
    train_img_copy = [os.path.join(train_img_dir, img.split('\\')[-1]) for img in train_img]
    train_label = [toLabelPath(img, label_path) for img in train_img]
    train_label_copy = [os.path.join(train_label_dir, label.split('\\')[-1]) for label in train_label]
    for i in tqdm(range(len(train_img)), desc='train ', ncols=80, unit='img'):
        _copy(train_img[i], train_img_dir)
        _copy(train_label[i], train_label_dir)
        all_img_path.remove(train_img[i])
    val_img = all_img_path
    val_label = [toLabelPath(img, label_path) for img in val_img]
    for i in tqdm(range(len(val_img)), desc='val ', ncols=80, unit='img'):
        _copy(val_img[i], val_img_dir)
        _copy(val_label[i], val_label_dir)


def _copy(from_path, to_path):
    shutil.copy(from_path, to_path)


def toLabelPath(img_path, label_path):
    img = img_path.split('\\')[-1]
    label = img.split('.bmp')[0] + '.txt'
    return os.path.join(label_path, label)


def main():
    img_path = r"E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\Train\AllImages"
    label_path = "E:\PycharmProject\Datasets\CV\HRSC2016\HRSC2016Dataset\Train\YOLO_labels"
    split_list = [0.8, 0.2]  # 数据集划分比例[train:val]
    split_img(img_path, label_path, split_list)


if __name__ == '__main__':
    main()