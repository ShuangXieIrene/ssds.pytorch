# source https://github.com/narumiruna/efficientnet-pytorch/blob/master/efficientnet/models/efficientnet.py
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .rutils import register


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        # padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            # nn.ZeroPad2d(padding),
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            Swish(),
            # MemoryEfficientSwish(),
        )

    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            # MemoryEfficientSwish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        expand_ratio,
        kernel_size,
        stride,
        reduction_ratio=4,
        drop_connect_rate=0.2,
    ):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(
                hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim
            ),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class EfficientNet(nn.Module):
    def __init__(
        self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000
    ):
        super(EfficientNet, self).__init__()
        settings = [
            # t,  c, n, s, k
            [1, 16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112
            [6, 24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56
            [6, 40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28
            [6, 80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7
            [6, 320, 1, 1, 3],  # MBConv6_3x3, SE,   7 ->   7
        ]
        self.settings = settings

        out_channels = _round_filters(32, width_mult)
        self.conv1 = ConvBNReLU(3, out_channels, 3, stride=2)

        in_channels = out_channels
        for j, (t, c, n, s, k) in enumerate(settings):
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            stage = []
            for i in range(repeats):
                stride = s if i == 0 else 1
                stage += [
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=t,
                        stride=stride,
                        kernel_size=k,
                    )
                ]
                in_channels = out_channels
            self.add_module("stage{}".format(j + 1), nn.Sequential(*stage))

        last_channels = _round_filters(1280, width_mult)

        self.head_conv = ConvBNReLU(in_channels, last_channels, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        for j in range(len(self.setting)):
            x = getattr(self, "stage{}".format(j + 1))(x)
        x = self.head_conv(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class EfficientEx(EfficientNet):
    def __init__(
        self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, outputs=[7], url=None
    ):
        super(EfficientEx, self).__init__(
            width_mult=width_mult, depth_mult=depth_mult, dropout_rate=dropout_rate
        )
        self.url = url
        self.outputs = outputs
        self.depth_mult = depth_mult

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)
            change_dict = {"features.0.": "conv1."}
            f_idx = 1
            for j, (t, c, n, s, k) in enumerate(self.settings):
                repeats = _round_repeats(n, self.depth_mult)
                for i in range(repeats):
                    change_dict["features.{}.".format(f_idx)] = "stage{}.{}.".format(
                        j + 1, i
                    )
                    f_idx += 1
            change_dict["features.{}.".format(f_idx)] = "head_conv."
            for k, v in list(checkpoint.items()):
                for _k, _v in list(change_dict.items()):
                    if _k in k:
                        new_key = k.replace(_k, _v)
                        checkpoint[new_key] = checkpoint.pop(k)

            # stuff to remove the zero_padding
            for k, v in list(checkpoint.items()):
                if "conv" in k and "se" not in k:
                    k_list = k.split(".")
                    if k_list[-2].isdigit() and k_list[-3] != "conv":
                        k_list[-2] = str(int(k_list[-2]) - 1)
                        new_key = ".".join(k_list)
                        checkpoint[new_key] = checkpoint.pop(k)
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)

        outputs = []
        for j in range(len(self.settings)):
            level = j + 1  # only 1 conv before
            if level > max(self.outputs):
                break
            x = getattr(self, "stage{}".format(level))(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


model_urls = {
    "efficientnet_b0": "https://www.dropbox.com/s/9wigibun8n260qm/efficientnet-b0-4cfa50.pth?dl=1",
    "efficientnet_b1": "https://www.dropbox.com/s/6745ear79b1ltkh/efficientnet-b1-ef6aa7.pth?dl=1",
    "efficientnet_b2": "https://www.dropbox.com/s/0dhtv1t5wkjg0iy/efficientnet-b2-7c98aa.pth?dl=1",
    "efficientnet_b3": "https://www.dropbox.com/s/5uqok5gd33fom5p/efficientnet-b3-bdc7f4.pth?dl=1",
    "efficientnet_b4": "https://www.dropbox.com/s/y2nqt750lixs8kc/efficientnet-b4-3e4967.pth?dl=1",
    "efficientnet_b5": "https://www.dropbox.com/s/qxonlu3q02v9i47/efficientnet-b5-4c7978.pth?dl=1",
}


@register
def EfficientNetB0(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2,
        outputs=outputs,
        url=model_urls["efficientnet_b0"],
    )


@register
def EfficientNetB1(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.0,
        depth_mult=1.1,
        dropout_rate=0.2,
        outputs=outputs,
        url=model_urls["efficientnet_b1"],
    )


@register
def EfficientNetB2(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.1,
        depth_mult=1.2,
        dropout_rate=0.3,
        outputs=outputs,
        url=model_urls["efficientnet_b2"],
    )


@register
def EfficientNetB3(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.2,
        depth_mult=1.4,
        dropout_rate=0.3,
        outputs=outputs,
        url=model_urls["efficientnet_b3"],
    )


@register
def EfficientNetB4(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.4,
        depth_mult=1.8,
        dropout_rate=0.4,
        outputs=outputs,
        url=model_urls["efficientnet_b4"],
    )


@register
def EfficientNetB5(outputs, **kwargs):
    return EfficientEx(
        width_mult=1.6,
        depth_mult=2.2,
        dropout_rate=0.4,
        outputs=outputs,
        url=model_urls["efficientnet_b5"],
    )
