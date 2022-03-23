import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import maml.transforms as T
import os.path as osp
from skimage import draw
import numpy as np
from xml.dom import minidom
from pathlib import Path
import shutil
import random


def poly2mask(x, y, nrow, ncol):
    fill_row_coords, fill_col_coords = draw.polygon(y, x, (nrow, ncol))
    mask = np.zeros((nrow, ncol), dtype=np.int)
    mask[fill_row_coords, fill_col_coords] = 1
    return mask


def xml2mask(imagename, xml_label):
    img = Image.open(imagename)
    imarray = np.array(img)
    h, w, c = imarray.shape

    xmldoc = minidom.parse(xml_label)
    regionlist = xmldoc.getElementsByTagName('Region')

    mask = np.zeros((h, w), dtype=np.int)

    for region_id, region in enumerate(regionlist):
        verticies = region.getElementsByTagName('Vertex')
        x = np.array([float(vert.getAttribute('X')) for vert in verticies])
        y = np.array([float(vert.getAttribute('Y')) for vert in verticies])

        polygon = poly2mask(x, y, h, w)
        mask = mask + (1 - mask.clip(max=1)) * polygon

    return mask


class MoNuSeg(Dataset):
    name = 'MoNuSeg'
    out_channels = 2

    def __init__(self, root, split, num_sample_train=None, transform=None, target_transform=None):
        super(MoNuSeg, self).__init__()

        monuseg_dir = 'grand-challenge/MoNuSeg Training Data'

        # root = /export/medical_ai/
        self.root = os.path.expanduser(root)

        self.imgdir = osp.join(self.root, monuseg_dir, 'Tissue Images')
        self.maskdir = osp.join(self.root, monuseg_dir, 'Masks')

        self.imglist = sorted(os.listdir(self.imgdir))
        self.masklist = [item.replace('.png', '_mask.png') for item in self.imglist]

        num_img = len(self.imglist)

        if split != 'all':
            random.seed(42)
            test_set = random.sample(self.imglist, k=int(num_img * 0.3))

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
        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.maskdir, self.imglist[idx].replace(".png", "_mask.png"))

        image = Image.open(image_name)
        mask = Image.open(mask_name)

        # print('processing {}'.format(image_name))
        if self.split == 'test':
            random.seed(0) # apply this seed to img transforms

        if self.transform:
            image, mask = self.transform(image, mask)

        mask[mask <= 127] = 0
        mask[mask > 127] = 1

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.imglist)


def preprocess(image_dir, xml_dir, out_dir):
    imgList = Path(image_dir).glob("*.png")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for img_path in imgList:
        xml_label = str(Path(xml_dir) / img_path.name.replace(".png", ".xml"))
        print('processing {}'.format(str(img_path)))
        mask = xml2mask(img_path, xml_label)
        mask_name = img_path.name.replace(".png", "_mask.png")
        im = Image.fromarray(mask.astype(np.uint8))
        im.save(str(Path(out_dir) / mask_name))


def main():
    transform = T.Compose([T.RandomResize(800, 2000),
                           T.ColorJitter(0.2, 0.2, 0.1, 0.1),
                           T.RandomHorizontalFlip(0.5),
                           T.RandomVerticalFlip(0.5),
                           T.RandomCrop(768),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = MoNuSeg(root='/export/medical_ai/', split='train', num_sample_train=None, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              sampler=torch.utils.data.RandomSampler(dataset),
                                              num_workers=0, drop_last=True)

    for image, target in data_loader:
        print(image.shape)
        print(target.shape)


if __name__ == "__main__":
    image_dir = '/export/medical_ai/grand-challenge/MoNuSeg Training Data/Tissue Images'
    xml_dir = '/export/medical_ai/grand-challenge/MoNuSeg Training Data/Annotations'
    output_dir = '/export/medical_ai/grand-challenge/MoNuSeg Training Data/Masks'
    # preprocess(image_dir, xml_dir, output_dir)
    main()
