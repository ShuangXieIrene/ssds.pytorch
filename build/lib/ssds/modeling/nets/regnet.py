import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ssds.modeling.nets.rutils import register


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(in_w, out_w, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3 [+SE], 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self._construct(
                stem_w=kwargs["stem_w"],
                ds=kwargs["ds"],
                ws=kwargs["ws"],
                ss=kwargs["ss"],
                bms=kwargs["bms"],
                gws=kwargs["gws"],
                se_r=kwargs["se_r"],
                nc=kwargs["nc"],
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(
                name, AnyStage(prev_w, w, s, d, ResBottleneckBlock, bm, gw, se_r)
            )
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))  # ks = [0,1,2...,3...]
    ws = w_0 * np.power(w_m, ks)  # float channel for 4 stages
    ws = np.round(np.divide(ws, q)) * q  # make it divisible by 8
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(
        self,
        w_a,
        w_0,
        w_m,
        d,
        group_w,
        bot_mul,
        se_r=None,
        num_classes=1000,
        outputs=[4],
        url=None,
        **kwargs
    ):
        # Generate RegNet ws per block
        ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        # Convert to per stage format
        s_ws, s_ds = get_stages_from_blocks(ws, ws)
        # Use the same gw, bm and ss for each stage
        s_gs = [group_w for _ in range(num_stages)]
        s_bs = [bot_mul for _ in range(num_stages)]
        s_ss = [2 for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)
        # Get AnyNet arguments defining the RegNet
        kwargs = {
            "stem_w": 32,
            "ds": s_ds,
            "ws": s_ws,
            "ss": s_ss,
            "bms": s_bs,
            "gws": s_gs,
            "se_r": se_r,
            "nc": num_classes,
        }
        self.outputs = outputs
        self.url = url
        super(RegNet, self).__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url)["model_state"])

    def forward(self, x):
        x = self.stem(x)

        outputs = []
        for i, layer in enumerate([self.s1, self.s2, self.s3, self.s4]):
            level = i + 1
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


base_url = "https://dl.fbaipublicfiles.com/pycls/dds_baselines/"
model_urls = {
    "RegNetX002": "160905981/RegNetX-200MF_dds_8gpu.pyth",
    "RegNetX004": "160905967/RegNetX-400MF_dds_8gpu.pyth",
    "RegNetX006": "160906442/RegNetX-600MF_dds_8gpu.pyth",
    "RegNetX008": "160906036/RegNetX-800MF_dds_8gpu.pyth",
    "RegNetX016": "160990626/RegNetX-1.6GF_dds_8gpu.pyth",
    "RegNetX032": "160906139/RegNetX-3.2GF_dds_8gpu.pyth",
    "RegNetX040": "160906383/RegNetX-4.0GF_dds_8gpu.pyth",
    "RegNetX064": "161116590/RegNetX-6.4GF_dds_8gpu.pyth",
    "RegNetX080": "161107726/RegNetX-8.0GF_dds_8gpu.pyth",
    "RegNetX120": "160906020/RegNetX-12GF_dds_8gpu.pyth",
    "RegNetX160": "158460855/RegNetX-16GF_dds_8gpu.pyth",
    "RegNetX320": "158188473/RegNetX-32GF_dds_8gpu.pyth",
}


@register
def RegNetX002(outputs, **kwargs):
    """ s1-4: {24, 56, 368, 152} """
    model = RegNet(
        w_a=36.44,
        w_0=24,
        w_m=2.49,
        d=13,
        group_w=8,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX002"],
        **kwargs
    )
    return model


@register
def RegNetX004(outputs, **kwargs):
    """ s1-4: {32, 64, 160, 384} """
    model = RegNet(
        w_a=24.48,
        w_0=24,
        w_m=2.54,
        d=22,
        group_w=16,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX004"],
        **kwargs
    )
    return model


@register
def RegNetX006(outputs, **kwargs):
    """ s1-4: {48, 96, 240, 528} """
    model = RegNet(
        w_a=36.97,
        w_0=48,
        w_m=2.24,
        d=16,
        group_w=24,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX006"],
        **kwargs
    )
    return model


@register
def RegNetX008(outputs, **kwargs):
    """ s1-4: {64, 128, 288, 672} """
    model = RegNet(
        w_a=35.73,
        w_0=56,
        w_m=2.28,
        d=16,
        group_w=16,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX008"],
        **kwargs
    )
    return model


@register
def RegNetX016(outputs, **kwargs):
    """ s1-4: {72, 168, 408, 912} """
    model = RegNet(
        w_a=34.01,
        w_0=80,
        w_m=2.25,
        d=18,
        group_w=24,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX016"],
        **kwargs
    )
    return model


@register
def RegNetX032(outputs, **kwargs):
    """ s1-4: {96, 192, 432, 1008} """
    model = RegNet(
        w_a=26.31,
        w_0=88,
        w_m=2.25,
        d=25,
        group_w=48,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX032"],
        **kwargs
    )
    return model


@register
def RegNetX040(outputs, **kwargs):
    """ s1-4: {80, 240, 560, 1360} """
    model = RegNet(
        w_a=38.65,
        w_0=96,
        w_m=2.43,
        d=23,
        group_w=40,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX040"],
        **kwargs
    )
    return model


@register
def RegNetX064(outputs, **kwargs):
    """ s1-4: {168, 392, 784, 1624} """
    model = RegNet(
        w_a=60.83,
        w_0=184,
        w_m=2.07,
        d=17,
        group_w=56,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX064"],
        **kwargs
    )
    return model


@register
def RegNetX080(outputs, **kwargs):
    """ s1-4: {80, 240, 720, 1920} """
    model = RegNet(
        w_a=49.56,
        w_0=80,
        w_m=2.88,
        d=23,
        group_w=120,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX080"],
        **kwargs
    )
    return model


@register
def RegNetX120(outputs, **kwargs):
    """ s1-4: {224, 448, 896, 2240} """
    model = RegNet(
        w_a=73.36,
        w_0=168,
        w_m=2.37,
        d=19,
        group_w=112,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX120"],
        **kwargs
    )
    return model


@register
def RegNetX160(outputs, **kwargs):
    """ s1-4: {256, 512, 896, 2048} """
    model = RegNet(
        w_a=55.59,
        w_0=216,
        w_m=2.1,
        d=22,
        group_w=128,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX160"],
        **kwargs
    )
    return model


@register
def RegNetX320(outputs, **kwargs):
    """ s1-4: {336, 672, 1344, 2520} """
    model = RegNet(
        w_a=69.86,
        w_0=320,
        w_m=2.0,
        d=23,
        group_w=168,
        bot_mul=1,
        outputs=outputs,
        url=base_url + model_urls["RegNetX320"],
        **kwargs
    )
    return model
