import torch
import torch.nn as nn
from new_methods.model.attention import *

class PAM_CAM(nn.Module):
    def __init__(self, in_dim):
        super(PAM_CAM, self).__init__()
        # add the pam module Position attention
        self.sa = PAM_Module(in_dim)

        # add the cam module Channel attention
        self.sc = CAM_Module(in_dim)


        # add the normLayer
        self.conv51 = nn.Sequential(nn.BatchNorm2d(in_dim),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.BatchNorm2d(in_dim),
                                    nn.ReLU())
        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False))

    def forward(self, x, enble_PAM, enble_CAM):
        # PAM module
        sa_feat = self.sa(x)
        sa_conv = self.conv51(sa_feat)

        # CAM module
        sc_feat = self.sc(x)
        sc_conv = self.conv52(sc_feat)

        # merge PAM and CAM
        if enble_CAM and enble_PAM:
            feat_sum = sa_conv + sc_conv
        elif enble_CAM:
            feat_sum = sc_conv
        elif enble_PAM:
            feat_sum = sa_conv

        return feat_sum
if __name__ == '__main__':
    test_pam = PAM_CAM(12 * 8)
    input = torch.randn(1, 12 * 8, 14, 14)
    output = test_pam(input, True, False)
    print(output.shape)


