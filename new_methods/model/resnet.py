import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pickle
num=0
class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()
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

        # classifier
        # self.cls = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
        # )
        num_features = model.layer4[1].conv1.in_channels
        self.cls = self.classifier(num_features, num_classes)
        #
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
        #self.parent_map = x
        self.parent_map = x
        x = self.features[7](x)
        with open('newrdcam_file.pkl', 'wb') as f:  # open a text file
            pickle.dump(x, f)  # serialize the list
            num += 1
        out = self.cls(x)
        self.salience_maps = out
        peak_list, aggregation = None, F.adaptive_avg_pool2d(out, 1).squeeze(2).squeeze(2)
        return aggregation

    def get_loss(self, logits, gt_labels):
        # loss_cls = self.CrossEntropyLoss(logits, gt_labels.long())

        loss_cls = F.multilabel_soft_margin_loss(logits, gt_labels)
        loss_val = loss_cls
        return loss_val

    def get_salience_maps(self):
        return self.parent_map, self.salience_maps

def model(pretrained=True, num_classes=10):
    model = models.resnet34(pretrained=pretrained)
    model_ft = FC_ResNet(model, num_classes=num_classes)
    return model_ft


# import torch.optim as optim
if __name__ == '__main__':
    model_ft = model(True, 10)
    model_ft = FC_ResNet(model, num_classes=12)
