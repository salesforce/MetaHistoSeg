import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import maml.transforms as T
import os.path as osp
from pathlib import Path
import random

Image.MAX_IMAGE_PIXELS = 933120000


class digestPath(Dataset):
    name = 'digestPath'
    out_channels = 2

    def __init__(self, root, split, num_sample_train=None, transform=None, target_transform=None):
        super(digestPath, self).__init__()

        digestPath_pos_dir = 'grand-challenge/digestPath/tissue-train-pos-v1'
        digestPath_neg_dir = 'grand-challenge/digestPath/tissue-train-neg'

        # root = /export/medical_ai/
        self.root = os.path.expanduser(root)

        self.imgdir = osp.join(self.root, digestPath_pos_dir)

        self.masklist = sorted(Path(self.imgdir).glob("*_mask.jpg"))
        self.imglist = [str(item).replace('_mask.jpg', '.jpg') for item in self.masklist]

        num_img = len(self.imglist)

        if split != 'all':
            random.seed(42)
            test_set = random.sample(self.imglist, k=int(num_img*0.3))

            if split == 'train':
                self.imglist = sorted(list(set(self.imglist).difference(test_set)))
                if num_sample_train and num_sample_train < len(self.imglist):
                    self.imglist = random.sample(self.imglist, k=num_sample_train)
            else:
                self.imglist = test_set

        self.transform = transform
        self.target_transform = target_transform
        self.split = split

    def __getitem__(self, idx):  # return one class
        image_name = osp.join(self.imgdir, self.imglist[idx])
        mask_name = osp.join(self.imgdir, image_name.replace(".jpg", "_mask.jpg"))

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

        # print(np.unique(np.array(mask), return_counts=True))
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

    dataset = digestPath(root='/export/medical_ai/', split='train', num_sample_train=None, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              sampler=torch.utils.data.RandomSampler(dataset),
                                              num_workers=0, drop_last=True)

    for image, target in data_loader:
        print(image.shape)
        print(target.shape)


if __name__ == "__main__":
    main()
