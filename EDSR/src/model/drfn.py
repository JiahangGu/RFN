# Edge-guided Residual Fractal Network for SISR

import torch
import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return DRFN(args)


## Residual Channel Attention Block (RCAB)
class RCB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, act=nn.PReLU()):

        super(RCB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class fractIn2(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCB, flag=False, act=nn.ReLU(True)):
        super(fractIn2, self).__init__()

        self.fract1 = block(conv, n_feats, kernel_size)
        self.fract2 = block(conv, n_feats, kernel_size)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res, out], dim=1)
        return res


class fractIn4(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCB, flag=False, act=nn.ReLU(True)):
        super(fractIn4, self).__init__()

        self.fract1 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 2, n_feats, 1)
        self.fract2 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res, out], dim=1)
        return res


class fractIn8(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCB, flag=False, act=nn.ReLU(True)):
        super(fractIn8, self).__init__()

        self.fract1 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 3, n_feats, 1)
        self.fract2 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res, out], dim=1)
        return res


class RFB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, edge_num=1, act=nn.PReLU()):

        super(RFB, self).__init__()
        self.fb = fractIn8(conv, n_feat, kernel_size)
        self.conv1x1 = conv(n_feat*4, n_feat, 1)
        self.conv3x3 = conv(n_feat, n_feat, 3)
        # self.conv_edge = conv(n_feat*4, 1, 1)

    def forward(self, x):
        res = self.fb(x)
        # edge = self.conv_edge(res)
        res = self.conv1x1(res)
        res = self.conv3x3(res)
        res += x
        return res


class DRFN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DRFN, self).__init__()

        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.PReLU()
        block_num = 12

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # 最后一层的join层输入为log(层数)
        self.rfb1 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb2 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb3 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb4 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb5 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb6 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb7 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb8 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb9 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb10 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb11 = RFB(conv, n_feats, kernel_size, act=act)
        self.rfb12 = RFB(conv, n_feats, kernel_size, act=act)
        # m_body.append()
        self.bottle = conv(n_feats * block_num, n_feats, 1)
        self.body = conv(n_feats, n_feats, 3)
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)
        out1 = self.rfb1(x)
        out2 = self.rfb2(out1)
        out3 = self.rfb3(out2)
        out4 = self.rfb4(out3)
        out5 = self.rfb5(out4)
        out6 = self.rfb6(out5)
        out7 = self.rfb7(out6)
        out8 = self.rfb8(out7)
        out9 = self.rfb9(out8)
        out10 = self.rfb10(out9)
        out11 = self.rfb11(out10)
        out12 = self.rfb12(out11)
        out = self.bottle(torch.cat((out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12), dim=1))
        res = self.body(out)
        res += x

        x = self.tail(res)

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

