import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from new_methods.expr.train import device
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes= 256, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out
class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes, cos_alpha, num_maps, enable_CBAM=True):
        super(FC_ResNet, self).__init__()
        # super parameters
        self.cos_alpha = cos_alpha
        #num=256

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
        if enable_CBAM:
            self.cbam1 = CBAM(64, 16, 7)
            self.cbam2 = CBAM(128, 16, 7)
            self.cbam3 = CBAM(256, 16, 7)
        # DA
        # self.DA = DA(256, num_maps, 448)
        # self.PAM = PAM_CAM(1024)
        # self.enable_PAM = enable_PAM
        # self.enable_CAM = enable_CAM

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
        # x = self.features[0:7](x)
        # x=self.cbam(x)
        # self.parent_map = x
        x = self.features[0:5](x)
        x=self.cbam1(x)
        x = self.features[5](x)
        x=self.cbam2(x)
        x = self.features[6](x)
        x=self.cbam3(x)
        x = self.features[7](x)
        self.parent_map = x

        self.dcam=x
        x = self.cls(x)
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
        loss_val = loss_cls #+ self.cos_alpha * loss_cos

        return loss_val  # , loss_cls, loss_cos

    def get_salience_maps(self):
        # salience_maps = self.DA.get_salience_maps()
        return self.parent_map, self.salience_maps

    def get_dcam(self):
        return self.dcam

def model(pretrained=True, num_classes=10, cos_alpha=0.01, num_maps=4, pam=True, cam=True):
    model = models.resnet34(pretrained=pretrained)
    model_ft = FC_ResNet(model, num_classes=num_classes, cos_alpha=cos_alpha, num_maps=num_maps, enable_CBAM=True)
    return model_ft

if __name__ == '__main__':
    test_cbam = CBAM(256,16,7)
