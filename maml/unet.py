from collections import OrderedDict

import torch
import torch.nn as nn
from modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                     MetaSequential, MetaConvTranspose2d)
from modules.utils import get_subdict
from torch.utils.data import Dataset


class UNet(MetaModule):

    def __init__(self, in_channels=3, meta_train_classifer_head=None,
                 meta_validation_out_channels=None, init_features=32):
        super(UNet, self).__init__()

        if meta_train_classifer_head:
            assert isinstance(meta_train_classifer_head, OrderedDict)

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = MetaConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = MetaConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = MetaConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = MetaConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        if meta_train_classifer_head:
            for name, out_channels in meta_train_classifer_head.items():
                setattr(self, 'classifier_{}'.format(name), MetaConv2d(
                    in_channels=features,
                    out_channels=out_channels,
                    kernel_size=1
                ))

        self.classifier_validation = MetaConv2d(
            in_channels=features,
            out_channels=meta_validation_out_channels,
            kernel_size=1
        )

        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, params=None):
        if params:
            return self.meta_train_forward(x, params)
        else:
            return self.meta_validation_forward(x)

    def meta_train_forward(self, x, params):
        for key in params:
            params[key] = params[key][0]

        enc1 = self.encoder1(x, params=get_subdict(params, 'encoder1'))
        enc2 = self.encoder2(self.pool1(enc1), params=get_subdict(params, 'encoder2'))
        enc3 = self.encoder3(self.pool2(enc2), params=get_subdict(params, 'encoder3'))
        enc4 = self.encoder4(self.pool3(enc3), params=get_subdict(params, 'encoder4'))

        bottleneck = self.bottleneck(self.pool4(enc4), params=get_subdict(params, 'bottleneck'))

        dec4 = self.upconv4(bottleneck, params=get_subdict(params, 'upconv4'))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4, params=get_subdict(params, 'decoder4'))
        dec3 = self.upconv3(dec4, params=get_subdict(params, 'upconv3'))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3, params=get_subdict(params, 'decoder3'))
        dec2 = self.upconv2(dec3, params=get_subdict(params, 'upconv2'))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2, params=get_subdict(params, 'decoder2'))
        dec1 = self.upconv1(dec2, params=get_subdict(params, 'upconv1'))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1, params=get_subdict(params, 'decoder1'))
        classifier_name = [key for key in params.keys() if key.startswith("classifier")]
        assert len(classifier_name) > 1
        classifier_name = classifier_name[0].split('.')[0]
        classifier = getattr(self, classifier_name)

        return self.LogSoftmax(classifier(dec1, params=get_subdict(params, classifier_name)))

    def meta_validation_forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.LogSoftmax(self.classifier_validation(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return MetaSequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        MetaConv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", MetaBatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        MetaConv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", MetaBatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def get_unet_model(episodic_dataset, batch_dataset):
    assert isinstance(episodic_dataset, Dataset)
    assert isinstance(batch_dataset, Dataset)

    meta_train_dataset_output_channels = OrderedDict()

    for task_name in episodic_dataset.task_name:
        meta_train_dataset_output_channels[task_name] = episodic_dataset.task_class[task_name].out_channels

    model = UNet(in_channels=3, meta_train_classifer_head=meta_train_dataset_output_channels,
                 meta_validation_out_channels=batch_dataset.out_channels)
    return model


def get_meta_test_unet_model(batch_dataset):
    assert isinstance(batch_dataset, Dataset)

    model = UNet(in_channels=3, meta_train_classifer_head=None,
                 meta_validation_out_channels=batch_dataset.out_channels)
    return model
