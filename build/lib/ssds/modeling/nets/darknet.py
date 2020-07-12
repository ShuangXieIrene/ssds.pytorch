import torch
import torch.nn as nn
from .rutils import register


def Conv3x3BNReLU(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels=nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class DarkNet(nn.Module):
    def __init__(
        self,
        layers=[1, 2, 8, 8, 4],
        outputs=[5],
        groups=1,
        width_per_group=64,
        url=None,
    ):
        super(DarkNet, self).__init__()
        self.outputs = outputs
        self.url = url

        self.conv1 = Conv3x3BNReLU(in_channels=3, out_channels=32)

        self.block1 = self._make_layers(
            in_channels=32, out_channels=64, block_num=layers[0]
        )
        self.block2 = self._make_layers(
            in_channels=64, out_channels=128, block_num=layers[1]
        )
        self.block3 = self._make_layers(
            in_channels=128, out_channels=256, block_num=layers[2]
        )
        self.block4 = self._make_layers(
            in_channels=256, out_channels=512, block_num=layers[3]
        )
        self.block5 = self._make_layers(
            in_channels=512, out_channels=1024, block_num=layers[4]
        )

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(
            Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=2)
        )
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def initialize(self):
        pass

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        for level in range(1, 6):
            if level > max(self.outputs):
                break
            x = getattr(self, "block{}".format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


@register
def DarkNet53(outputs, **kwargs):
    return DarkNet(outputs=outputs)
