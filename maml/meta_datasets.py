from maml.datasets import Gleason2019, BreastPathQ, digestPath, MoNuSeg, GlandSegmentation
from torch.utils.data import Dataset
import numpy as np
import torch

name2class = {'gleason2019': Gleason2019,
              'BreastPathQ': BreastPathQ,
              'digestpath': digestPath,
              'MoNuSeg': MoNuSeg,
              'GlandSegmentation': GlandSegmentation}


def get_single_dataset(task_name,
                       folder,
                       transform,
                       num_sample_train=None,
                       split='all'):
    task_class = name2class[task_name](folder, split=split, transform=transform, num_sample_train=num_sample_train)
    return task_class


class MetaDataset(Dataset):
    def __init__(self,
                 task_name,
                 task_prob,
                 support_sz,
                 query_sz,
                 num_episode,
                 folder,
                 transform,
                 shuffle,
                 random_state_seed=0):
        """
        :param datasource_list: list with datasource name and prob
        :param support_size:
        :param query_size:
        :param shuffle:
        :param num_episode:
        :param random_state_seed:
        """
        self.task_name = task_name
        self.task_prob = task_prob
        self.episode_sz = {'all': support_sz + query_sz,
                           'support': support_sz,
                           'query': query_sz}
        self.task_in_episode = np.random.choice(range(len(self.task_name)),
                                                size=num_episode,
                                                p=self.task_prob)
        self.num_episode = num_episode
        self.task_class = {name: name2class[name](folder, split='all', transform=transform)
                           for name in self.task_name}
        self.shuffle = shuffle
        self.random_state_seed = random_state_seed
        self.np_random = np.random.RandomState(seed=random_state_seed)

    def __getitem__(self, index):
        task_id = self.task_in_episode[index]
        task_name = self.task_name[task_id]
        task = self.task_class[task_name]

        instance_index = self.sample_instance_indices(task)

        image_support, mask_support = self.pack_instances(task, instance_index['support'])
        image_query, mask_query = self.pack_instances(task, instance_index['query'])

        return image_support, mask_support, image_query, mask_query, task_name

    def __len__(self):
        return self.num_episode

    def pack_instances(self, task, index_list):
        instance_list = [task[index] for index in index_list]
        images = [instance[0] for instance in instance_list]
        masks = [instance[1] for instance in instance_list]
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks

    def sample_instance_indices(self, task):
        indices = {'support': [], 'query': []}

        num_samples = len(task)
        if num_samples < self.episode_sz['all']:
            raise ValueError('The number of samples for data source ({0}) '
                             'is smaller than the minimum number of required samples per task '
                             .format(num_samples, self.episode_sz['episode']))

        if self.shuffle:
            seed = (hash(task) + self.random_state_seed) % (2 ** 32)
            dataset_indices = np.random.RandomState(seed).permutation(num_samples)
        else:
            dataset_indices = np.arange(num_samples)

        pr = 0
        for split in ['support', 'query']:
            split_indices = dataset_indices[pr:pr + self.episode_sz[split]]
            if self.shuffle:
                self.np_random.shuffle(split_indices)
            indices[split] = split_indices
            pr += self.episode_sz[split]

        return indices


class BatchDataset(Dataset):
    def __init__(self,
                 task_name,
                 task_prob,
                 total_num_samples,
                 folder,
                 transform,
                 shuffle,
                 random_state_seed=0):
        self.task_name = task_name
        self.task_prob = task_prob
        self.total_num_samples = total_num_samples
        self.task_id = np.random.choice(range(len(self.task_name)),
                                        size=total_num_samples,
                                        p=self.task_prob)
        self.task_class = {name: name2class[name](folder, split='all', transform=transform)
                           for name in self.task_name}
        self.shuffle = shuffle
        self.random_state_seed = random_state_seed
        self.np_random = np.random.RandomState(seed=random_state_seed)

    def __getitem__(self, index):
        task_id = self.task_id[index]
        task_name = self.task_name[task_id]
        task = self.task_class[task_name]

        instance_index = self.sample_instance_indice(task)

        image, mask = task[instance_index]

        return image, mask, task_name

    def __len__(self):
        return self.total_num_samples

    def sample_instance_indice(self, task):
        num_samples = len(task)

        if self.shuffle:
            seed = (hash(task) + self.random_state_seed) % (2 ** 32)
            dataset_indices = np.random.RandomState(seed).permutation(num_samples)
        else:
            dataset_indices = np.arange(num_samples)

        indice = dataset_indices[0]

        return indice
