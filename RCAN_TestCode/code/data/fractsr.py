# Deep Residual Down and Up Scale Super Resolution
# concat is more efficient than normal

import torch
import torch.nn as nn
from model import common


def make_model(args, parent=False):
    return FRACTSR(args)


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
        # modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class SALayer(nn.Module):
    def __init__(self, channels, kernel_size=7, act=nn.ReLU(True)):
        super(SALayer, self).__init__()
        padding = kernel_size // 2
        self.conv1_1 = nn.Conv2d(channels, channels // 2, (1, kernel_size), padding=(0, padding))
        self.conv1_2 = nn.Conv2d(channels // 2, 1, (kernel_size, 1), padding=(padding, 0))
        self.conv2_1 = nn.Conv2d(channels, channels // 2, (kernel_size, 1), padding=(padding, 0))
        self.conv2_2 = nn.Conv2d(channels // 2, 1, (1, kernel_size), padding=(0, padding))
        self.conv3_1 = nn.Conv2d(channels, channels // 2, 3, padding=1)
        self.conv3_2 = nn.Conv2d(channels // 2, 1, 3, padding=2, dilation=2)
        self.conv1x1 = nn.Conv2d(3, 1, 1)
        self.act = act
        self.sig = nn.Sigmoid()

    def forward(self, x):
        attention1 = self.conv1_2(self.act(self.conv1_1(x)))
        attention2 = self.conv2_2(self.act(self.conv2_1(x)))
        attention3 = self.conv3_2(self.act(self.conv3_1(x)))
        attention = self.conv1x1(torch.cat([attention1, attention2, attention3], dim=1))
        sa = self.sig(attention)
        return x * sa


# class SALayer(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SALayer, self).__init__()
#         # global average pooling: feature --> point
#         padding = kernel_size // 2
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         a = torch.cat([avg_out,max_out], dim=1)
#         a = self.conv1(a)
#         a = self.sigmoid(a)
#         return a * x


class GCB(nn.Module):
    def __init__(self, channels, reduction=16, dilation=2, act=nn.ReLU(True)):
        super(GCB, self).__init__()
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)
        content_mask = self.conv1(x)
        content_mask = content_mask.view(b, 1, h * w)
        content_mask = self.softmax(content_mask)
        content_mask = content_mask.unsqueeze(3)
        res = torch.matmul(input_x, content_mask)
        res = res.view(b, c, 1, 1)
        ca_out = self.ca(res)
        res = x * ca_out
        return res


class RSAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act=nn.ReLU(True)):
        super(RSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size))
            if i == 0: modules_body.append(act)
        modules_body.append(SALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class fractIn2(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn2, self).__init__()

        self.fract1 = block(conv, n_feats, kernel_size)
        self.fract2 = block(conv, n_feats, kernel_size)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + x, out], dim=1)
        return res


class fractIn4(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn4, self).__init__()

        self.fract1 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 2, n_feats, 1)
        # self.join = LAB(n_feats * 2, n_feats)
        self.fract2 = fractIn2(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        ans = self.fract2(res)
        out = self.fract3(x)
        ans = torch.cat([ans + torch.cat([x, x], dim=1), out], dim=1)
        return ans


class fractIn8(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn8, self).__init__()

        self.fract1 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 3, n_feats, 1)
        # self.join = LAB(n_feats * 3, n_feats)
        self.fract2 = fractIn4(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + torch.cat([x, x, x], dim=1), out], dim=1)
        return res


class fractIn16(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn16, self).__init__()

        self.fract1 = fractIn8(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 4, n_feats, 1)
        # self.join = LAB(n_feats * 4, n_feats)
        self.fract2 = fractIn8(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + torch.cat([x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn32(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn32, self).__init__()

        self.fract1 = fractIn16(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 5, n_feats, 1)
        # self.join = LAB(n_feats * 5, n_feats)
        self.fract2 = fractIn16(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + torch.cat([x, x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn64(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn64, self).__init__()

        self.fract1 = fractIn32(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join = conv(n_feats * 6, n_feats, 1)
        # self.join = LAB(n_feats * 6, n_feats)
        self.fract2 = fractIn32(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + torch.cat([x, x, x, x, x, x], dim=1), out], dim=1)
        return res


class fractIn128(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, block=RCAB, flag=False, act=nn.ReLU(True)):
        super(fractIn128, self).__init__()

        self.fract1 = fractIn64(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.join1 = conv(n_feats * 7, n_feats, kernel_size=1)
        self.fract2 = fractIn64(conv, n_feats, kernel_size, block=block, flag=flag, act=act)
        self.fract3 = block(conv, n_feats, kernel_size)

    def forward(self, x):
        res = self.fract1(x)
        res = self.join1(res)
        res = self.fract2(res)
        out = self.fract3(x)
        res = torch.cat([res + torch.cat([x, x, x, x, x, x, x], dim=1), out], dim=1)
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

        # 最后一层的join层输入为log(层数)
        self.body = nn.Sequential(fractIn64(conv, n_feats, kernel_size, block=RCAB, flag=False, act=act), \
                                  conv(n_feats * 7, n_feats, 1))

        # define tail module

        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        # m_tail = [
        #       nn.Conv2d(n_feats*7, n_feats*scale*scale, 3, padding=1),
        #       nn.PixelShuffle(scale),
        #       nn.Conv2d(n_feats, args.n_colors, 3, padding=1)
        # ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)

        x = self.head(x)

        res = self.body(x)
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
