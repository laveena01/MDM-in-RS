from time import sleep

import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from new_methods.model.my_DA import DA
from new_methods.model.PAM_CAM import *
from new_methods.expr.train import device
from matplotlib import pyplot as plt
import csv
import pandas as pd
import pickle
global flag
flag = 0
global num
num=16

class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes, cos_alpha, num_maps, enable_PAM=True, enable_CAM=True):
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
        self.DA = DA(256, num_maps, 448)
        self.PAM = PAM_CAM(256)
        self.enable_PAM = enable_PAM
        self.enable_CAM = enable_CAM

        # self.cls = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1)
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




    def forward(self, x, labels=None):
        x = self.features[0:7](x)

        # for i in range(16):
        #     fig = plt.figure(figsize=(13, 13))
        #     for j in range(16):
        #         fig.add_subplot(4, 4, j + 1)
        #         plt.imshow(x[0][i * 16 + j].cpu().detach().numpy())
        #         plt.axis('off')
        #     plt.savefig(f"D:\\CODE_AND_RESULTS\\new_methods\\utils\\scam_maps\\{i}.png")
            # plt.savefig(f"C:\\Users\\meemu\\Downloads\\CODE_AND_RESULTS\\{i}.png")

        # y = x.squeeze()
        # floc = '../utils/trans.csv'
        global num
        with open(f"D:\\CODE_AND_RESULTS\\new_methods\\utils\\scam_pkl\\scam{num}_file.pkl", 'wb') as f:  # open a text file
            pickle.dump(x, f)  # serialize the list
           # i+=1
        # if not flag:
        #     flag = 1
        #     with open(floc, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         for i in range(256):
        #             cnt = 0
        #             px = []
        #             loc = []
        #             for j in range(28):
        #                 for k in range(28):
        #                     if (y[i][j][k] > 0.5):
        #                         cnt += 1
        #                         px.append(y[i][j][k])
        #                         loc.append((j, k))
        #             # writer.writerow(cnt)
        #             writer.writerow(px)
        #             writer.writerow(loc)
        #             # print(cnt)
        #             # print(px)
        #             # print(loc)
        feat = self.DA(x)
        #
        x = 0.5 * x + 0.5 * feat

        if self.enable_CAM != False or self.enable_PAM != False:
            x = self.PAM(x, self.enable_PAM, self.enable_CAM)

        self.parent_map = x

        x = self.features[7](x)


        # for i in range(16):
            # fig = plt.figure(figsize=(13, 13))
            # for j in range(32):
                # fig.add_subplot(8, 4, j + 1)
                # plt.imshow(x[0][i * 32 + j].cpu().detach().numpy())
                # plt.axis('off')
            # plt.savefig(f"D:\\CODE_AND_RESULTS\\new_methods\\utils\\dcam_maps\\{i}.png")
        with open(f"D:\\CODE_AND_RESULTS\\new_methods\\utils\\dcam_pkl\\dcam{num}_file.pkl", 'wb') as f:  # open a text file
            # pickle.dump(x, f)  # serialize the list
            num += 1
        # print(x.size())
        self.dcam = x
        x = self.cls(x)
        # fig = plt.figure(figsize=(13, 13))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(x[0][0].cpu().detach().numpy(), cmap='gray')
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(x[0][1].cpu().detach().numpy(), cmap='gray')
        # print(x[0][1])
        # fig = plt.figure(figsize=(13, 13))
        self.salience_maps = x

        peak_list, aggregation_child = None, F.adaptive_avg_pool2d(x, 1).squeeze(2).squeeze(2)

        return aggregation_child

    def get_loss(self, logits, gt_labels):
        # import numpy as np
        # gt_label = np.zeros((len(gt_labels), 2))
        #
        # for i in range(len(gt_labels)):
        #     gt_label[i][gt_labels[i]] = 1
        # loss_cos = self.DA.get_loss_(torch.from_numpy(gt_label).to(device))

        # loss_cos = self.DA.get_loss(gt_labels)
        # loss_cos = self.DA.get_loss_()
        loss_cls = self.CrossEntropyLoss(logits, gt_labels.long())

        # loss_cls = F.multilabel_soft_margin_loss(logits, gt_labels)
        loss_val = loss_cls  # + self.cos_alpha * loss_cos

        return loss_val  # , loss_cls, loss_cos

    def get_salience_maps(self):
        # salience_maps = self.DA.get_salience_maps()
        return self.parent_map, self.salience_maps

    def get_dcam(self):
        return self.dcam


def model(pretrained=True, num_classes=10, cos_alpha=0.01, num_maps=4, pam=True, cam=True):
    model = models.resnet34(pretrained=pretrained)
    model_ft = FC_ResNet(model, num_classes=num_classes, cos_alpha=cos_alpha, num_maps=num_maps, enable_PAM=pam,
                         enable_CAM=cam)
    return model_ft


if __name__ == '__main__':
    from new_methods.model.basenet import resnet34

    model = resnet34(pretrained=True)
    model_ft = FC_ResNet(model, num_classes=8, cos_alpha=0.01, num_maps=4).to(device)
    x = torch.randn(1, 3, 448, 448).to(device)
    output = model_ft(x)
    print(output)
    loss = model_ft.get_loss(output, torch.randn(1).to(device))
    _, cam = model_ft.get_salience_maps()
    print(loss, _.shape, cam.shape)
