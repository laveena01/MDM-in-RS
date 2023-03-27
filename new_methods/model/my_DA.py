import torch.nn as nn
import torch
import random
import torch.nn.functional as F
import numpy as np
from new_methods.expr.train import device
from new_methods.model.PAM_CAM import *

class DA(nn.Module):
    def __init__(self, in_dim, k, input_size):
        # k means the num of DA maps
        super(DA, self).__init__()
        # super parameters
        self.input_size = input_size
        self.in_dim = in_dim
        self.num_maps = k


        self.cls5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, in_dim * k, kernel_size=1, padding=0)
        )

    def forward(self, x):
        batch_size = x.size(0)
        size = int(self.input_size / 16)


        childResult_multimaps = self.cls5(x).view(batch_size, self.in_dim, self.num_maps, size, size)
        childResult = torch.sum(childResult_multimaps, 2).view(batch_size, self.in_dim, size, size) / self.num_maps

        self.child_map = childResult_multimaps
        self.salience_maps = childResult_multimaps
        return childResult


    def calculate_cosineloss(self, maps):
        batch_size = maps.size(0)
        num_maps = maps.size(1)
        channel_num = int(self.num_maps/2)
        eps = 1e-8
        random_seed = random.sample(range(num_maps), channel_num)
        maps = maps[:, random_seed, :, :].view(batch_size, channel_num, -1)

        # maps_max = maps.max(dim=2)[0].expand(maps.shape[-1], batch_size, channel_num).permute(1, 2, 0)
        # maps = maps/maps_max
        X1 = maps.unsqueeze(1)
        X2 = maps.unsqueeze(2)
        dot11, dot22, dot12 = (X1 * X1).sum(3), (X2 * X2).sum(3), (X1 * X2).sum(3)
        # print(dot12)
        dist = dot12 / (torch.sqrt(dot11 * dot22 + eps))
        tri_tensor = ((torch.Tensor(np.triu(np.ones([channel_num, channel_num])) - np.diag([1]*channel_num))).expand(batch_size, channel_num, channel_num))
        tri_tensor = tri_tensor.to(device)
        dist = dist.to(device)
        dist_num = abs((tri_tensor*dist).sum(1).sum(1)).sum()/(batch_size*channel_num*(channel_num-1)/2)

        return dist_num, random_seed


    def get_loss(self, gt_labels):

        gt_label_random = torch.zeros(gt_labels.size(0))
        for i in range(gt_labels.size(0)):
            gt_label_random[i] = random.sample(list(torch.nonzero(gt_labels[i])[:, 0]), 1)[0]

        batch_size = self.child_map.size(0)
        size = int(self.input_size / 16)

        label_index = []
        for i in range(gt_labels.size(0)):
            label_index.append(list(torch.nonzero(gt_labels[i])))
        map_temp = self.child_map.reshape(batch_size * self.in_dim, self.num_maps, size, size)
        temp = torch.zeros((batch_size, self.num_maps, size, size)).to(device)

        for i in range(batch_size):
            for k in label_index[i]:
                temp[i] += map_temp[i * self.in_dim + k].squeeze(0)

        # maps = torch.cat((
        #     (self.child_map.reshape(batch_size * self.num_classes, self.num_maps, size, size)[
        #      [gt_label_random[i].long() + (i * self.num_classes) for i in range(batch_size)], :, :, :]).reshape(batch_size,
        #
        #                                                         self.num_maps, size, size),), 1)

        temp = temp / len(label_index)
        maps = torch.cat((temp,), 1)

        loss_cos, random_seed = self.calculate_cosineloss(maps)

        return loss_cos



    def get_loss_(self, labels=None):


        map_temp = self.child_map.sum(axis=1) / (self.in_dim * self.num_maps)

        maps = torch.cat((map_temp,), 1)

        loss_cos, random_seed = self.calculate_cosineloss(maps)

        return loss_cos


    def get_salience_maps(self):
        return torch.mean(F.relu(self.salience_maps), dim=2)

if __name__ == '__main__':
    input = torch.randn(32, 256, 28, 28).to(device)
    model = DA(256, 4, 448)
    model = model.to(device)
    output = model(input)
    print(model.get_loss_())