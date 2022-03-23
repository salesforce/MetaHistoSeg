import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import maml.transforms as T
import os.path as osp
import numpy as np
import multiprocessing
from multiprocessing import Pool
import shutil
import re
import random

import SimpleITK as sitk


def staple(item, inputdirs, outputdir):
    print("processing {}...".format(item))

    imgs = []
    for p in inputdirs:
        if osp.isfile(osp.join(p, item)):
            imgs.append(sitk.ReadImage(osp.join(p, item)))

    # 255 is for those ambiguous pixels
    result = sitk.MultiLabelSTAPLE(imgs, 255)
    p1_data = sitk.GetArrayFromImage(imgs[0])
    result_data = sitk.GetArrayFromImage(result)

    result_data[result_data == 255] = p1_data[result_data == 255]
    result_data[result_data == 7] = p1_data[result_data == 7]
    result_data[result_data == 6] = 2
    
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(imgs[0])
    sitk.WriteImage(result, osp.join(outputdir, item))


class Gleason2019(Dataset):
    name = 'gleason2019'
    out_channels = 6

    def __init__(self, root, split, num_sample_train=None, transform=None, target_transform=None):
        super(Gleason2019, self).__init__()

        assert split in ['all', 'train', 'test']

        gleason_dir = 'grand-challenge/gleason2019'

        # root = /export/medical_ai/
        self.root = os.path.expanduser(root)

        self.imgdir = osp.join(self.root, gleason_dir, 'Train_imgs')
        self.maskdir = osp.join(self.root, gleason_dir, 'mask/finalmask')

        self.imglist = sorted(os.listdir(self.imgdir))

        num_img = len(self.imglist)

        if split != 'all':
            random.seed(42)
            test_set = random.sample(self.imglist, k=int(num_img*0.7))

            if split == 'train':
                self.imglist = test_set #sorted(list(set(self.imglist).difference(test_set)))
                if num_sample_train:
                    self.imglist = random.sample(self.imglist, k=num_sample_train)
            else:
                self.imglist = test_set

        self.transform = transform
        self.target_transform = target_transform
        self.split = split

    def __getitem__(self, idx):  # return one class
        image = Image.open(osp.join(self.imgdir, self.imglist[idx]))
        mask_name = osp.join(self.maskdir, self.imglist[idx].replace(".jpg", "_classimg_nonconvex.png"))
        mask = Image.open(mask_name)
        before_aug = np.array(mask).max()
        # print('processing {}'.format(self.imglist[idx]))
        # print(np.unique(np.array(mask), return_counts=True))
        if self.split == 'test':
            random.seed(0) # apply this seed to img transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        if self.target_transform:
            mask = self.target_transform(mask)
        # print('checking {} max{} before {}'.format(self.imglist[idx], mask.max(), before_aug))
        assert mask.max() < 6 and before_aug < 6
        return image, mask

    def __len__(self):
        return len(self.imglist)


def preprocessing(input_dir, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    maskdirs = [osp.join(input_dir, maskdir) for maskdir in os.listdir(input_dir) if re.match('Maps', maskdir)]
    maskfiles = []
    for maskdir in maskdirs:
        print(maskdir)
        maskfiles = maskfiles + os.listdir(maskdir)

    maskfiles = set(maskfiles)

    processes = multiprocessing.cpu_count()

    with Pool(processes=processes) as pool:
        results = [pool.apply_async(staple, args=(maskfile,
                                                  maskdirs, out_dir))
                   for maskfile in maskfiles]
        _ = [_.get() for _ in results]
    print("Done")


def main():
    transform = T.Compose([T.RandomResize(800, 2000),
                           T.ColorJitter(0.2, 0.2, 0.1, 0.1),
                           T.RandomHorizontalFlip(0.5),
                           T.RandomVerticalFlip(0.5),
                           T.RandomCrop(768),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = Gleason2019(root='/export/medical_ai/', split='train', num_sample_train=None, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              sampler=torch.utils.data.RandomSampler(dataset),
                                              num_workers=0, drop_last=True)

    for image, target in data_loader:
        print(image.shape)
        print(target.shape)


if __name__ == "__main__":
    input_dir = '/export/medical_ai/grand-challenge/gleason2019/mask'
    output_dir = '/export/medical_ai/grand-challenge/gleason2019/mask/finalmask'
    #preprocessing(input_dir, output_dir)
    main()
