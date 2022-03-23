import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import maml.transforms as T
import os.path as osp
from pathlib import Path
import random


class GlandSegmentation(Dataset):
    name = 'GlandSegmentation'
    out_channels = 2

    def __init__(self, root, split, num_sample_train=None, transform=None, target_transform=None):
        super(GlandSegmentation, self).__init__()

        glandSegmentation_dir = 'grand-challenge/Warwick QU Dataset (Released 2016_07_08)'

        # root = /export/medical_ai/
        self.root = os.path.expanduser(root)

        self.imgdir = osp.join(self.root, glandSegmentation_dir)

        self.masklist = sorted(Path(self.imgdir).glob("*_anno.bmp"))
        self.imglist = [str(item).replace('_anno.bmp', '.bmp') for item in self.masklist]

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
        mask_name = osp.join(self.imgdir, self.imglist[idx].replace(".bmp", "_anno.bmp"))

        image = Image.open(image_name)
        mask = Image.open(mask_name)

        # print('processing {}'.format(image_name))

        if self.split == 'test':
            random.seed(0) # apply this seed to img transforms

        if self.transform:
            image, mask = self.transform(image, mask)

        mask[mask != 0] = 1

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.imglist)


def main():
    transform = T.Compose([T.RandomResize(800, 2000),
                           T.ColorJitter(0.2, 0.2, 0.1, 0.1),
                           T.RandomHorizontalFlip(0.5),
                           T.RandomVerticalFlip(0.5),
                           T.RandomCrop(768),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = GlandSegmentation(root='/export/medical_ai/', split='train', transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              sampler=torch.utils.data.RandomSampler(dataset),
                                              num_workers=0, drop_last=True)

    for image, target in data_loader:
        print(image.shape)
        print(target.shape)


if __name__ == "__main__":
    main()
