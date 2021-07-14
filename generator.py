"""

"""
import torch
import functools
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, nf=64, mid=32):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, mid, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + mid, mid, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * mid, mid, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * mid, mid, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * mid, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        out = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return out * 0.2 + x


class RRDBBlock(nn.Module):

    def __init__(self, c_in, c_out=32):
        super(RRDBBlock, self).__init__()
        self.RB1 = ResBlock(c_in, c_out)
        self.RB2 = ResBlock(c_in, c_out)
        self.RB3 = ResBlock(c_in, c_out)

    def forward(self, x):
        out = self.RB1(x)
        out = self.RB2(out)
        out = self.RB3(out)
        return out * 0.2 + x



#     save = Generator(3, 3, 64, 23, mid=32)
class Generator(nn.Module):
    def __init__(self, c_in, c_out, mid_in, mid_out=32, block_nums=23):
        super(Generator, self).__init__()
        RRDBblock = functools.partial(RRDBBlock, mid_in, mid_out)
        self.conv1 = nn.Conv2d(c_in, mid_in, 3, 1, 1, bias=True)
        # The backbone
        block = []
        for _ in range(block_nums):
            block.append(RRDBblock())
        self.backbone = nn.Sequential(*block)
        # upsample
        self.conv2 = nn.Conv2d(mid_in, mid_in, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(mid_in, mid_in, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_in, mid_in, 3, 1, 1, bias=True)
        self.conv_end = nn.Conv2d(mid_in, mid_in, 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(mid_in, c_out, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out_backbone = self.conv2(self.backbone(out1))
        # resnet
        features = out1 + out_backbone
        # upsample & convolution
        features = self.lrelu(self.upconv1(F.interpolate(features, scale_factor=2, mode='nearest')))
        features = self.lrelu(self.upconv2(F.interpolate(features, scale_factor=2, mode='nearest')))
        features = self.lrelu(self.conv_end(features))
        out = self.conv_out(features)

        return out









