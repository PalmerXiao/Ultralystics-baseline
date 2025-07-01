# coding:utf-8

import os
import random
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
from os import getcwd
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', 
                    default=r'E:\PycharmProject\Datasets\CV\DIOR\Annotations\Horizontal Bounding Boxes', 
                    type=str, help='input xml label path')
parser.add_argument('--img_path', 
                    default=r'E:\PycharmProject\Datasets\CV\DIOR\JPEGImages',
                    type=str, help='input image path')
parser.add_argument('--output_path', 
                    default=r'E:\PycharmProject\Datasets\CV\DIOR',
                    type=str, help='output path')

opt = parser.parse_args()

# 验证路径是否存在
def check_path(path, create=False):
    if not os.path.exists(path):
        if create:
            try:
                os.makedirs(path)
                print(f"创建目录: {path}")
            except Exception as e:
                print(f"创建目录失败 {path}: {str(e)}")
                return False
        else:
            print(f"路径不存在: {path}")
            return False
    return True

# 检查输入路径
if not check_path(opt.xml_path) or not check_path(opt.img_path):
    print("请检查输入路径是否正确！")
    exit(1)

sets = ['train', 'val', 'test']
classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam',
          'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
          'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']

# 创建输出目录
for split in sets:
    labels_path = os.path.join(opt.output_path, 'labels', split)
    images_path = os.path.join(opt.output_path, 'images', split)
    check_path(labels_path, create=True)
    check_path(images_path, create=True)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    # YOLO格式中心点坐标和宽高
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id, path):
    try:
        in_file = open(os.path.join(opt.xml_path, f'{image_id}.xml'), encoding='UTF-8')
        out_file = open(os.path.join(opt.output_path, 'labels', path, f'{image_id}.txt'), 'w')
        
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            b1 = max(0, min(b1, w))
            b2 = max(0, min(b2, w))
            b3 = max(0, min(b3, h))
            b4 = max(0, min(b4, h))
            
            bb = convert((w, h), (b1, b2, b3, b4))
            out_file.write(f"{cls_id} {' '.join(map(str, bb))}\n")
        
        in_file.close()
        out_file.close()
        return True
    except Exception as e:
        print(f"Error processing {image_id}: {str(e)}")
        return False

# 数据集划分
train_percent = 0.7
val_percent = 0.1
test_percent = 0.2

total_xml = [f for f in os.listdir(opt.xml_path) if f.endswith('.xml')]
num = len(total_xml)
indices = list(range(num))
random.shuffle(indices)

train_end = int(num * train_percent)
val_end = int(num * (train_percent + val_percent))

train_nums = set(indices[:train_end])
val_nums = set(indices[train_end:val_end])
test_nums = set(indices[val_end:])

print(f"总数据集大小: {num}")
print(f"训练集: {len(train_nums)}, 验证集: {len(val_nums)}, 测试集: {len(test_nums)}")

# 转换数据集
for i in tqdm(range(num), desc="Converting datasets"):
    name = total_xml[i][:-4]
    
    if i in train_nums:
        split = 'train'
    elif i in val_nums:
        split = 'val'
    else:
        split = 'test'
        
    if convert_annotation(name, split):
        # 复制图片
        src_img = os.path.join(opt.img_path, f'{name}.jpg')
        dst_img = os.path.join(opt.output_path, 'images', split, f'{name}.jpg')
        try:
            copyfile(src_img, dst_img)
        except Exception as e:
            print(f"Error copying image {name}: {str(e)}")

print("转换完成！")


