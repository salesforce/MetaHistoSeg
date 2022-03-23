import torch
import os
import time
import json
import logging

from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn as nn

import maml.transforms as T
from maml.meta_datasets import MetaDataset, get_single_dataset
from maml.metalearners import ModelAgnosticMetaLearning
from maml.unet import get_unet_model

from tensorboardX import SummaryWriter


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    device_ids = list(map(int, args.device_ids.split(',')))

    num_devices = torch.cuda.device_count()

    assert len(device_ids) > 1

    if num_devices < len(device_ids):
        raise Exception('#available gpu : {} < --device_ids : {}'.format(num_devices, len(device_ids)))

    assert torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(device_ids[0]))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logging.debug('Creating folder `{0}`'.format(args.output_folder))

    folder = os.path.join(args.output_folder,
                          time.strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(folder)
    logging.debug('Creating folder `{0}`'.format(folder))

    args.folder = os.path.abspath(args.folder)
    args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
    # Save the configuration in a config.json file
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info('Saving configuration file in `{0}`'.format(
        os.path.abspath(os.path.join(folder, 'config.json'))))

    summary_writer = SummaryWriter(folder)
    summary = {'step': 0}

    if not isinstance(args.dataset, list):
        args.dataset = [args.dataset]

    if not isinstance(args.dataset_prob, list):
        args.dataset_prob = [args.dataset_prob]

    assert len(args.dataset) == len(args.dataset_prob)

    transform_aug = T.Compose([T.RandomResize(800, 2000),
                               T.ColorJitter(0.2, 0.2, 0.1, 0.1),
                               T.RandomHorizontalFlip(0.5),
                               T.RandomVerticalFlip(0.5),
                               T.RandomCrop(768),
                               T.ToTensor(),
                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_no_aug = T.Compose([T.RandomResize(800, 2000),
                                  T.RandomCrop(768),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    meta_train_dataset = MetaDataset(task_name=args.dataset,
                                     task_prob=args.dataset_prob,
                                     support_sz=args.support_size,
                                     query_sz=args.query_size,
                                     num_episode=5000,
                                     transform=transform_aug,
                                     folder=args.folder,
                                     shuffle=True)

    meta_train_dataloaders = torch.utils.data.DataLoader(meta_train_dataset, batch_size=args.batch_size,
                                                         sampler=torch.utils.data.RandomSampler(meta_train_dataset),
                                                         num_workers=args.num_workers, drop_last=True)

    meta_validation_dataset_train = get_single_dataset(args.validation_dataset,
                                                       args.folder,
                                                       transform=transform_aug,
                                                       split='train')
    meta_validation_dataset_test = get_single_dataset(args.validation_dataset,
                                                      args.folder,
                                                      transform=transform_no_aug,
                                                      split='test')

    # validation data loader
    meta_validation_dataloader_train = DataLoader(meta_validation_dataset_train,
                                                  batch_size=args.batch_size_test,
                                                  shuffle=True,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    meta_validation_dataloader_test = DataLoader(meta_validation_dataset_test,
                                                 batch_size=args.batch_size_test,
                                                 shuffle=False,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True)

    model = get_unet_model(meta_train_dataset, meta_validation_dataset_train)

    # meta training
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

    model = DataParallel(model, device_ids=device_ids).to(device=device)

    if 0 and args.pretrain_model_path:
        with open(args.pretrain_model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

    metalearner = ModelAgnosticMetaLearning(model,
                                            optimizer=meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_adaptations,
                                            step_size=args.step_size,
                                            test_every=args.test_every,
                                            loss_function=nn.NLLLoss(),
                                            device=device)

    # Training loop
    metalearner.train_episodic(dataloaders=meta_train_dataloaders,
                               validation_dataloader_train=meta_validation_dataloader_train,
                               validation_dataloader_test=meta_validation_dataloader_test,
                               meta_validation_lr=args.meta_test_lr,
                               summary_writer=summary_writer,
                               summary=summary,
                               max_iterations=args.max_iterations,
                               weight_path=args.model_path,
                               validation_epochs=args.validation_epochs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--meta-test-data-root', type=str,
                        help='Path to the meta-test')
    parser.add_argument('--dataset', nargs="*", type=str,
                        default=['gleason2019', 'digestpath', 'BreastPathQ', 'MoNuSeg', 'GlandSegmentation'],
                        help='Name of dataset sources')
    parser.add_argument('--dataset-prob', nargs="*", type=float,
                        default=[1 / 3, 1 / 3, 1 / 3],
                        help='Name of dataset sources')
    parser.add_argument('--validation-dataset', type=str,
                        default='gleason2019',
                        help='Name of dataset sources')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder to save the model.')
    parser.add_argument('--support-size', type=int, default=2,
                        help='Number of samples in support set of an episode ')
    parser.add_argument('--query-size', type=int, default=2,
                        help='Number of samples in query set of an episode ')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of episodes in a batch')
    parser.add_argument('--batch-size-test', type=int, default=4,
                        help='Number of episodes in a batch')
    parser.add_argument('--max-iterations', type=int, default=5000,
                        help='Number of iterations in meta training')
    parser.add_argument('--validation-epochs', type=int, default=1,
                        help='Number of refine epochs in meta validation')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels in each convolution layer of the VGG network '
                             '(default: 64).')

    # Optimization
    parser.add_argument('--pretrain-model-path', type=str, default=None,
                        help='pretrained-model.')
    parser.add_argument('--num-adaptations', type=int, default=1,
                        help='Number of fast adaptation steps, ie. gradient descent '
                             'updates (default: 1).')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Size of the fast adaptation step, ie. learning rate in the '
                             'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
                        help='Use the first order approximation, do not use higher-order '
                             'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
                        help='Learning rate for the meta-optimizer (optimization of the outer '
                             'loss). The default optimizer is Adam (default: 1e-3).')
    parser.add_argument('--meta-test-lr', type=float, default=0.001,
                        help='Learning rate for the meta test stage lr')
    parser.add_argument('--test-every', type=int, default=100,
                        help='validation every Number of training steps (default: 100).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of workers to use for data-loading (default: 0).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                        help="GPU indices ""comma separated, e.g. '0,1' ")

    args = parser.parse_args()

    main(args)
