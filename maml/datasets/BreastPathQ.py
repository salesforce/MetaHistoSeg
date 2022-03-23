import os
import numpy as np
import cv2
from PIL import Image
from bs4 import BeautifulSoup as Soup
import torch
from torch.utils.data import Dataset
import maml.transforms as T
import os.path as osp
from pathlib import Path
import shutil
import random

R = 12
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

MALIGNANT = ("IDC", "ILC", "Muc C", "DCIS 1", "DCIS 2", "DCIS 3", "MC- E", "MC - C", "MC - M")
NORMAL = ("normal", "UDH", "ADH")
LYMPHOCYTE = ("TIL-E", "TIL-S")
CLASS_DICT = {}
CLASS_DICT.update(dict.fromkeys(MALIGNANT, 0))
CLASS_DICT.update(dict.fromkeys(NORMAL, 1))
CLASS_DICT.update(dict.fromkeys(LYMPHOCYTE, 2))


def read_coords(fl, verbose=False):
    soup = Soup(open(fl), "lxml")
    elements = soup.findAll("graphic")
    cls = []
    xy = []
    for element in elements:
        if element.attrs["description"] not in CLASS_DICT:
            if verbose:
                print(fl.name, element.attrs["description"])
            continue
        points = element.findAll("point")
        xy.extend(tuple(int(p) for p in point.contents[0].split(",")) for point in points)
        cls.extend([CLASS_DICT[element.attrs["description"]]] * len(points))
    return xy, cls


def read_img_mask(img_path):
    key_path = img_path.parent / img_path.name.replace("_crop.tif", "_key.xml")
    xy, cls = read_coords(key_path)
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    mask3 = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for coord, cl in zip(xy, cls):
        cv2.circle(mask3, coord, R, COLORS[cl], -1)
    mask3 = mask3.transpose(2, 0, 1)

    mask[mask3[0] == 255] = 1
    mask[mask3[1] == 255] = 2
    mask[mask3[2] == 255] = 3

    mask3 = mask3.transpose(1, 2, 0)
    print(np.unique(mask, return_counts=True))

    return Image.fromarray(mask),  Image.fromarray(mask3, 'RGB')


class BreastPathQ(Dataset):
    name = 'breastPathQ'
    out_channels = 4

    def __init__(self, root, split, num_sample_train=None, transform=None, target_transform=None):
        super(BreastPathQ, self).__init__()

        breastpathq_dir = 'grand-challenge/BreastPathQ/breastpathq/datasets'

        # root = /export/medical_ai/
        self.root = os.path.expanduser(root)

        self.imgdir = osp.join(self.root, breastpathq_dir, 'cells')
        self.maskdir = osp.join(self.root, breastpathq_dir, 'Masks')

        self.imglist = sorted(Path(self.imgdir).glob("*_crop.tif"),
                              key=lambda s: int('0' if s.name.split("_")[0] == '' else s.name.split("_")[0]))

        num_img = len(self.imglist)

        if split != 'all':
            random.seed(42)
            test_set = random.sample(self.imglist, k=int(num_img*0.3))

            if split == 'train':
                self.imglist = sorted(list(set(self.imglist).difference(test_set)))
                if num_sample_train:
                    self.imglist = random.sample(self.imglist, k=num_sample_train)
            else:
                self.imglist = test_set

        self.transform = transform
        self.target_transform = target_transform
        self.split = split

    def __getitem__(self, idx):  # return one class
        image_path = self.imglist[idx]
        mask_path = osp.join(self.maskdir, str(image_path.name.replace(".tif", "_mask.png")))
        # print('processing {}'.format(mask_path))
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.split == 'test':
            random.seed(0) # apply this seed to img transforms

        if self.transform:
            image, mask = self.transform(image, mask)
        if self.target_transform:
            mask = self.target_transform(mask)

        # print(np.unique(mask, return_counts=True))
        return image, mask

    def __len__(self):
        return len(self.imglist)


def preprocess(data_dir, out_dir):
    imgList = sorted(Path(data_dir).glob("*_crop.tif"),
                     key=lambda s: int('0' if s.name.split("_")[0] == '' else s.name.split("_")[0]))

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for img_path in imgList:
        print('processing {}'.format(str(img_path)))
        mask, mask3 = read_img_mask(img_path)
        mask_name = img_path.name.replace(".tif", "_mask.png")
        mask.save(str(Path(out_dir) / mask_name))
        mask3_name = img_path.name.replace(".tif", "_mask3.png")
        mask3.save(str(Path(out_dir) / mask3_name))


def main():
    transform = T.Compose([T.RandomResize(800, 1000),
                           T.ColorJitter(0.2, 0.2, 0.1, 0.1),
                           T.RandomHorizontalFlip(0.5),
                           T.RandomVerticalFlip(0.5),
                           T.RandomCrop(768),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = BreastPathQ(root='/export/medical_ai/', split='train', num_sample_train=None, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              sampler=torch.utils.data.RandomSampler(dataset),
                                              num_workers=0, drop_last=True)

    for image, target in data_loader:
        print(image.shape)
        print(target.shape)


if __name__ == "__main__":
    data_dir = '/export/medical_ai/grand-challenge/BreastPathQ/breastpathq/datasets/cells'
    out_dir = '/export/medical_ai/grand-challenge/BreastPathQ/breastpathq/datasets/Masks'
    preprocess(data_dir, out_dir)
    #main()
