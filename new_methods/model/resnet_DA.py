import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from model.DA import DA
from model.PAM_CAM import *
from expr.train import device
num=0
class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes, cos_alpha, num_maps):
        super(FC_ResNet, self).__init__()
        # super parameters
        self.cos_alpha = cos_alpha

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # DA
        self.DA = DA(num_classes, num_maps, 448)

        # cls
        # self.cls = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
        # )
        self.cls = self.classifier(512, num_classes)

        # loss
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

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


    def forward(self, x, label=None):
        x = self.features[0:7](x)
        global num
        with open('newrscam_file.pkl', 'wb') as f:  # open a text file
            pickle.dump(x, f)  # serialize the list
        self.parent_map = x

        x = self.features[7](x)
        with open('newrdcam_file.pkl', 'wb') as f:  # open a text file
            pickle.dump(x, f)  # serialize the list
            num += 1
        # x = self.cls(x)
        x = self.DA(x)


        self.salience_maps = x

        peak_list, aggregation_child = None, F.adaptive_avg_pool2d(x, 1).squeeze(2).squeeze(2)

        return aggregation_child

    def get_loss(self, logits, gt_labels):
        # import numpy as np
        # gt_label = np.zeros((len(gt_labels), 2))
        # for i in range(len(gt_labels)):
        #     gt_label[i][gt_labels[i]] = 1
        # loss_cos = self.DA.get_loss(torch.from_numpy(gt_label).to(device))
        # loss_cls = self.CrossEntropyLoss(logits, gt_labels.long())
        # loss_val = loss_cls * 0.7 + self.cos_alpha * loss_cos
        #
        loss_cos = self.DA.get_loss(gt_labels)
        loss_cls = F.multilabel_soft_margin_loss(logits, gt_labels)
        loss_val = loss_cls * 0.7 + self.cos_alpha * loss_cos

        return loss_val  # , loss_cls, loss_cos

    def get_salience_maps(self):
        salience_maps = self.DA.get_salience_maps()
        return self.parent_map, self.salience_maps


def model(pretrained=True, num_classes=10, cos_alpha=0.01, num_maps=4):
    model = models.resnet34(pretrained=pretrained)
    model_ft = FC_ResNet(model, num_classes=num_classes, cos_alpha=cos_alpha, num_maps=num_maps)
    return model_ft


if __name__ == '__main__':
    model = models.resnet34(pretrained=True)
    model_ft = FC_ResNet(model, 12, 0.01, 4).to(device)
    x = torch.randn(1, 3, 448, 448).to(device)
    output = model_ft(x)
    loss = model_ft.get_loss(output, torch.randn(1).to(device))
    print(loss)