# 要直接在3维通道上操作
# 命名格式  IOXHXTX表示输入输出通道x，每个block的res_head_num为x，res_tail_num为x
import torch
import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return NEBRN(args)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


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


def up(in_channels, out_channels, kernel_size):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)


class UpBlock(nn.Module):
    def __init__(self, conv, in_ch, n_feats, kernel_size=3, act=nn.ReLU(True), scale=2, res_head_num=1, res_tail_num=1):
        super(UpBlock, self).__init__()
        self.conv_head = conv(in_ch, n_feats, 1)
        self.res_head = nn.Sequential(*[ResBlock(conv, n_feats, kernel_size, act=act) for _ in range(res_head_num)])
        self.mid = conv(n_feats, n_feats, kernel_size)
        self.up = nn.Sequential(*[conv(n_feats, n_feats*4, 1), nn.PixelShuffle(2)])
        # self.up = up(n_feats, n_feats, scale)
        self.res_tail = nn.Sequential(*[ResBlock(conv, n_feats, kernel_size, act=act) for _ in range(res_tail_num)])
        self.conv_tail = conv(n_feats, n_feats, 3)

    def forward(self, x):
        o1 = self.conv_head(x)
        o2 = self.res_head(o1)
        o3 = self.mid(o2)
        sr = self.up(o3 + o1)
        o3 = self.res_tail(sr)
        out = self.conv_tail(o3)
        return out + sr


class NEBRN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NEBRN, self).__init__()

        n_feats = 64
        kernel_size = 3
        scale = args.scale[0]
        act = nn.LeakyReLU(0.1, inplace=True)
        num_blocks = 10
        self.scale = scale

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        self.head = conv(3, n_feats, kernel_size)

        self.blocks = nn.ModuleList([UpBlock(conv, n_feats * scale * scale, n_feats, kernel_size=kernel_size, act=act, scale=scale, res_head_num=5, res_tail_num=5) for _ in range(num_blocks)])

        self.pixelUnShuffle = SpaceToDepth(scale)
        # self.pixelUnShuffle = nn.MaxPool2d(2)

        self.tail = conv(n_feats * num_blocks, 3, 3)

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        origin = torch.cat([x for _ in range(self.scale ** 2)], dim=1)
        # origin = x

        o1 = self.blocks[0](origin)
        lr1 = self.pixelUnShuffle(o1)
        res1 = origin - lr1

        o2 = self.blocks[1](res1)
        lr2 = self.pixelUnShuffle(o2)
        res2 = res1 - lr2

        o3 = self.blocks[2](res2)
        lr3 = self.pixelUnShuffle(o3)
        res3 = res2 - lr3

        o4 = self.blocks[3](res3)
        lr4 = self.pixelUnShuffle(o4)
        res4 = res3 - lr4

        o5 = self.blocks[4](res4)
        lr5 = self.pixelUnShuffle(o5)
        res5 = res4 - lr5

        o6 = self.blocks[5](res5)
        lr6 = self.pixelUnShuffle(o6)
        res6 = res5 - lr6

        o7 = self.blocks[6](res6)
        lr7 = self.pixelUnShuffle(o7)
        res7 = res6 - lr7

        o8 = self.blocks[7](res7)
        lr8 = self.pixelUnShuffle(o8)
        res8 = res7 - lr8

        o9 = self.blocks[8](res8)
        lr9 = self.pixelUnShuffle(o9)
        res9 = res8 - lr9

        o10 = self.blocks[9](res9)

        x = self.tail(torch.cat([o1, o2, o3, o4, o5, o6, o7, o8, o9, o10], dim=1))

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

