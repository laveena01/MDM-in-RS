import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import random

import sys

sys.path.append('../')

__all__ = [
    'model',
]

model_urls = {
    'inceptionv3': ''https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'',
    'inceptionv3bn': ''https://download.pytorch.org/models/inception_v3_bn_google-1a9a5a14.pth'',
    'inceptionv4': 'https://download.pytorch.org/models/inception_v4_google-1a3a8cdf.pth'',
    'inceptionresnetv1': 'https://download.pytorch.org/models/inception_resnet_v1_google-1a9a5a14.pth'',
    'inceptionresnetv2': 'https://download.pytorch.org/models/inception_resnet_v2_google-1a9a5a14.pth'',
}


class Inception(nn.Module):

    def __init__(self, features, num_classes=20, input_size=448):
        super(Inception, self).__init__()

        self.features = features[0:]
        self.num_classes = num_classes
        self.input_size = input_size
        self.cls = self.classifier(512, num_classes)
        self._initialize_weights()

        self.loss_cross_entropy = nn.CrossEntropyLoss()

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            # nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)  # fc8
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x, label=None):
        # x = self.features
        x = self.features[:24](x)

        self.parent_map = x
        x = self.features[24:](x)

        out = self.cls(x)
        self.salience_maps = out
        peak_list, aggregation = None, F.adaptive_avg_pool2d(out, 1).squeeze(2).squeeze(2)
        return aggregation

        # logits0 = torch.mean(torch.mean(self.cam_map, dim=2), dim=2)
        # return logits0

    def get_loss(self, logits, gt_labels):
        # loss_cls = self.loss_cross_entropy(logits0, gt_labels.long())

        logits0 = logits
        loss_cls = F.multilabel_soft_margin_loss(logits0, gt_labels)
        loss_val = loss_cls
        return loss_val


    def get_salience_maps(self):
        return self.parent_map, self.salience_maps

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}


def model(pretrained=False, **kwargs):
    """Inceptionv3 model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = Inceptionv3(make_layers(cfg['D']), **kwargs)
    # print(model)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['inceptionv3'])
        # print(pretrained_dict)
        print('load pretrained model from {}'.format(model_urls['inceptionv3']))
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from inceptionv3'.format(k))
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


if __name__ == "__main__":
    # pass
    model = model(pretrained=True, num_classes=12)
    x = torch.randn(1, 3, 448, 448)
    output = model(x)
    # cam, _ = model.get_salience_maps()
    # cam = cam.squeeze(0)
    # cam = cam.sum(0).cpu().detach().numpy()
    # print(cam.shape)
    # print(model)
