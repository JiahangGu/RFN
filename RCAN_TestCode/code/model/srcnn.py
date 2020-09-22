#Deep Residual Down and Up Scale Super Resolution
#concat is more efficient than normal

import torch
import torch.nn as nn
from model import common
import torch.nn.functional as F
from functools import reduce

def make_model(args, parent=False):
    return SRCNN(args)


class SRB(nn.Module):
    def __init__(self):
        super(SRB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, 5, padding=2)
        self.act = nn.ReLU(True)

    def forward(self, x):
        o1 = self.act(self.conv1(x))
        o2 = self.act(self.conv2(o1))
        o3 = self.conv3(o2)
        return o3


class SRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRCNN, self).__init__()

        # dug_num = args.dug_num
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        # m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     dug(
        #         conv, n_feats, n_feats, kernel_size,  block=RCAB, act=act, res_scale=args.res_scale
        #     ) for _ in range(dug_num)
        # ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        #最后一层的join层输入为log(层数)
        self.body = SRB()

        # self.dug2 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug3 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug4 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug5 = dug(conv, n_feats, n_feats, kernel_size)

        # define tail module

        # m_tail = [
        #     common.Upsampler(conv, scale, n_feats, act=False),
        #     conv(n_feats, args.n_colors, kernel_size)]
        #
        # tail = []
        # out_feats = scale * scale * args.n_colors
        # tail.append(
        #     wn(nn.Conv2d(n_feats*4, out_feats, 3, padding=1)))
        # tail.append(nn.PixelShuffle(scale))
        #
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.up = nn.Upsample()
        # self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        # self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        # x = self.head(x)
        x = F.upsample(x, scale_factor=2, mode="bilinear")

        x = self.body(x)
        # res += x

        # x = self.tail(res)

        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
