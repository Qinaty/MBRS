from pycocotools.coco import COCO
import numpy as np
import os
import argparse
from tqdm import tqdm
from PIL import Image


# parser = argparse.ArgumentParser()
# parser.add_argument('--dir', help='path for coco dataset', required=True)
# args = parser.parse_args()

# data_dir = args.dir  # '/data/micmic123/coco'
data_dir = '.'
# data_types = ['train2014', 'train2017']  # val2017
# data_types = ['train2014']
# for data_type in data_types:
annFile = './annotations/instances_train2014.json'
mask_dir = './train2014_mask'
source_dir = './train2014_test'
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(source_dir, exist_ok=True)

coco = COCO(annFile)
imgIds = coco.getImgIds()
images = coco.loadImgs(imgIds)

cnt = 0
tot = 0
for info in tqdm(images, desc=mask_dir):
    img_id = info['id']
    img_filename = info['file_name']
    img = Image.open(f'./train2014/{img_filename}')
    w, h = img.size

    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)

    mask = np.zeros((), dtype=np.uint8)
    for i in range(len(anns)):
        # 根据category_id随机生成mask的值
        pixel_value = anns[i]['category_id']
        # annToMask  - Convert segmentation in an annotation to binary mask.
        # np.maximum：(X, Y, out=None); X 与 Y 逐位比较取其大者；最少接收两个参数
        mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

    if len(np.unique(mask)) < 2 or w < 256 or h < 256:
        cnt += 1
        continue
    filename = img_filename.split('.')[0]
    mask = Image.fromarray(mask)
    mask.save(os.path.join(mask_dir, f'{filename}.png'))
    img.save(os.path.join(source_dir, f'{filename}.png'))

    tot += 1
    if tot >= 50:
        break

print('skipped:', cnt, f'{cnt / len(images):.2f}%')
