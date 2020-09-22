#Deep Residual Down and Up Scale Super Resolution
#concat is more efficient than normal

import torch
import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return FRACTSR(args)


def make_same(a, b):
    _, _, ha, wa = a.size()
    _, _, hb, wb = b.size()
    h = min(ha, hb)
    w = min(wa, wb)
    return a[:, :, :h, :w], b[:, :, :h, :w]


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        # global average pooling: feature --> point
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out,max_out], dim=1)
        a = self.conv1(a)
        a = self.sigmoid(a)
        return a * x


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


#more 0.12M params than CALayer
class MIXLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MIXLayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                common.hsigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return x * y


#mixed attention
class mix_attention(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(mix_attention, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size))
            if i == 0: modules_body.append(act)
        modules_body.append(MIXLayer(n_feat))
        # modules_body.append(nn.Sigmoid())
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class fractIn2(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn2, self).__init__()

        self.fract1 = block(conv, n_feats, kernel_size)
        self.fract2 = block(conv, n_feats, kernel_size)
        self.fract3 = block(conv, n_feats, kernel_size)
        self.join1 = conv(n_feats * 2, n_feats, kernel_size=1)

    def forward(self, x):
        res = self.fract1(x)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+x, out], dim=1)
        return res


class fractIn4(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn4, self).__init__()

        self.fract1 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats * 2, n_feats, kernel_size=1)
        self.fract2 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        ans = self.fract2(res)
        out = self.fract3(x)
        ans = torch.cat([ans+torch.cat([x, x], dim=1), out], dim=1)
        return ans


class fractIn8(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn8, self).__init__()

        self.fract1 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats*3, n_feats, kernel_size=1)
        self.fract2 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+torch.cat([x, x, x], dim=1), out], dim=1)
        return res


class fractIn16(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn16, self).__init__()

        self.fract1 = fractIn8(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats*4, n_feats, kernel_size=1)
        self.fract2 = fractIn8(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+torch.cat([x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn32(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn32, self).__init__()

        self.fract1 = fractIn16(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats*5, n_feats, kernel_size=1)
        self.fract2 = fractIn16(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+torch.cat([x, x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn64(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn64, self).__init__()

        self.fract1 = fractIn32(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats*6, n_feats, kernel_size=1)
        self.fract2 = fractIn32(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+torch.cat([x, x, x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn128(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn128, self).__init__()

        self.fract1 = fractIn64(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats*7, n_feats, kernel_size=1)
        self.fract2 = fractIn64(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res+torch.cat([x, x, x, x, x, x, x], dim=1), out], dim=1)
        return res


class FRACTSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FRACTSR, self).__init__()

        # dug_num = args.dug_num
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     dug(
        #         conv, n_feats, n_feats, kernel_size,  block=RCAB, act=act, res_scale=args.res_scale
        #     ) for _ in range(dug_num)
        # ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        #最后一层的join层输入为log(层数)
        self.body = nn.Sequential(fractIn64(conv, n_feats, kernel_size, block=RCAB, flag=False, act=act), \
                                  conv(n_feats*7, n_feats, kernel_size=1))

        # self.dug2 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug3 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug4 = dug(conv, n_feats, n_feats, kernel_size)

        # self.dug5 = dug(conv, n_feats, n_feats, kernel_size)

        # define tail module

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        #
        # tail = []
        # out_feats = scale * scale * args.n_colors
        # tail.append(
        #     wn(nn.Conv2d(n_feats*4, out_feats, 3, padding=1)))
        # tail.append(nn.PixelShuffle(scale))
        #
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail1 = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail1(res)

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
