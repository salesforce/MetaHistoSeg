import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from maml.metalearners import ModelAgnosticMetaLearning

from maml.unet import get_meta_test_unet_model
import maml.transforms as T
from maml.meta_datasets import get_single_dataset


def main(args):
    device_ids = list(map(int, args.device_ids.split(',')))

    assert len(device_ids) > 1

    num_devices = torch.cuda.device_count()
    if num_devices < len(device_ids):
        raise Exception(
            '#available gpu : {} < --device_ids : {}'
                .format(num_devices, len(device_ids)))

    assert torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(device_ids[0]))

    summary_writer = SummaryWriter(args.output_folder)

    summary = {'step': 0}

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

    meta_test_dataset_train = get_single_dataset(args.dataset,
                                                 args.folder,
                                                 transform=transform_aug,
                                                 num_sample_train=args.num_shot_train,
                                                 split='train')
    meta_test_dataset_test = get_single_dataset(args.dataset,
                                                args.folder,
                                                transform=transform_no_aug,
                                                split='test')

    meta_test_dataloader_train = DataLoader(meta_test_dataset_train,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)

    meta_test_dataloader_test = DataLoader(meta_test_dataset_test,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True)

    model = get_meta_test_unet_model(meta_test_dataset_train)

    model = DataParallel(model, device_ids=device_ids).to(device=device)

    ckpt = torch.load(args.model_path, map_location=device)
    model_dict = model.state_dict()

    pretrain_state = {k: v for k, v in ckpt.items() if k in model_dict and not k.startswith('module.classifier')}
    model_dict.update(pretrain_state)
    model.load_state_dict(model_dict)

    meta_test_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metalearner = ModelAgnosticMetaLearning(model)

    metalearner.meta_validation(epochs=args.epochs, unet=model,
                                loaders={'support': meta_test_dataloader_train,
                                         'query': meta_test_dataloader_test},
                                device=device,
                                optimizer=meta_test_optimizer,
                                criterion=nn.NLLLoss(),
                                summary_writer=summary_writer,
                                summary=summary,
                                prediction_folder=args.prediction_folder,
                                save_model=True)

    summary_writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # in
    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', default='gleason2019', type=str, help="")
    parser.add_argument('--model-path', type=str, default=None,
                        help='pretrained-model.')
    # out
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder to save the model.')
    # train hyperparas
    parser.add_argument('--batch-size', default=32, type=int, help="")
    parser.add_argument('--num-shot-train', default=None, type=int, help="")
    parser.add_argument('--lr', default=0.001, type=float, help="")
    parser.add_argument('--epochs', default=20, type=int, help="")
    # misc
    parser.add_argument('--num-workers', default=8, type=int, help="Number of "
                                                                   "workers for each data loader")
    parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                        help="GPU indices ""comma separated, e.g. '0,1' ")
    parser.add_argument('--prediction-folder', type=str, default=None,
                        help='Path to the output folder to save the sample vis.')

    args = parser.parse_args()
    main(args)
