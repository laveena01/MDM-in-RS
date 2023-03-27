import torch
from torch import nn
import torch.nn.functional as F

class FCInceptionV1(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(
            in_channels=3,
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x
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
        self.parent_map = x
        x = self.features[7](x)
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
    model = models.inceptionv3(pretrained=pretrained)
    model_ft = FC_Inception(model, num_classes=num_classes)
    return model_ft
# import torch.optim as optim
if __name__ == '__main__':
    model_ft = model(True, 10)
    model_ft = FC_Inception(model, num_classes=12)
