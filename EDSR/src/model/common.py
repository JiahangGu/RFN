import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def fill(x):
    b, c, h, w = x.size()
    pad_h = 8 - h % 8
    pad_w = 8 - w % 8
    y = F.pad(x, [0, pad_w, 0, pad_h])
    return y


class SpaceToDepth(nn.Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)
        return x


class Denoiser(nn.Module):
    def __init__(self, conv, channel, n_feat, act=nn.ReLU(True), bn=False):
        super(Denoiser, self).__init__()
        self.down2 = SpaceToDepth(2)
        self.down4 = SpaceToDepth(4)

        self.top1 = BasicBlock(conv, channel * 16, n_feat, 3, bn=bn)
        self.top2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.top3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.bottom1 = BasicBlock(conv, channel * 4, n_feat, 3, bn=bn)
        self.bottom_gate = conv(n_feat // 4 + n_feat, n_feat, 1)
        self.bottom2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.bottom3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.main1 = BasicBlock(conv, channel, n_feat, 3, bn=bn)
        self.main_gate = conv(n_feat + n_feat // 4, n_feat, 1)
        self.main2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.main3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.end = conv(n_feat, channel, 3)

    def forward(self, x):
        b, c, h, w = x.size()
        x = fill(x)
        top_x = self.down4(x)
        bottom_x = self.down2(x)

        top_x = self.top1(top_x)
        top_x = self.top2(top_x)
        top_x = self.top3(top_x)
        top_x = F.pixel_shuffle(top_x, 2)

        bottom_x = self.bottom1(bottom_x)
        bottom_x = torch.cat((bottom_x, top_x), 1)
        bottom_x = self.bottom_gate(bottom_x)
        bottom_x = self.bottom2(bottom_x)
        bottom_x = self.bottom3(bottom_x)
        bottom_x = F.pixel_shuffle(bottom_x, 2)

        x = self.main1(x)
        x = torch.cat((x, bottom_x), 1)
        x = self.main_gate(x)
        x = self.main2(x)
        x = self.main3(x)

        x = self.end(x)
        x = x[:, :, :h, :w]
        return x


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class SubMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(SubMeanStd, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class AddMeanStd(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=1):
        super(AddMeanStd, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) * std.view(3, 1, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean)
        for p in self.parameters():
            p.requires_grad = False


def get_mean_std(data):
    r = data[:, 0, :, :]
    g = data[:, 1, :, :]
    b = data[:, 2, :, :]
    r_std, r_mean = torch.std_mean(r)
    g_std, g_mean = torch.std_mean(g)
    b_std, b_mean = torch.std_mean(b)
    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)


class BasicBlock(nn.Module):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):
        super(BasicBlock, self).__init__()
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

