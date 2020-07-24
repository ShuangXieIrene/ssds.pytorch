import torch
import torch.nn as nn
from torchvision.models import mobilenet
import torch.utils.model_zoo as model_zoo
from .rutils import register


class SepConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expand_ratio=1):
        padding = (kernel_size - 1) // 2
        super(SepConvBNReLU, self).__init__(
            # dw
            nn.Conv2d(
                in_planes,
                in_planes,
                kernel_size,
                stride,
                padding,
                groups=in_planes,
                bias=False,
            ),
            nn.BatchNorm2d(in_planes),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, version="v1", round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNet, self).__init__()

        input_channel = 32
        if version == "v2":
            settings = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
            last_channel = 1280
            layer = mobilenet.InvertedResidual
        elif version == "v1":
            settings = [
                # t, c, n, s
                [1, 64, 1, 1],
                [1, 128, 2, 2],
                [1, 256, 2, 2],
                [1, 512, 6, 2],
                [1, 1024, 2, 2],
            ]
            last_channel = 1024
            layer = SepConvBNReLU
        self.settings = settings
        self.version = version

        # building first layer
        input_channel = mobilenet._make_divisible(
            input_channel * width_mult, round_nearest
        )
        self.last_channel = mobilenet._make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        self.conv1 = mobilenet.ConvBNReLU(3, input_channel, stride=2)
        # building inverted residual blocks
        for j, (t, c, n, s) in enumerate(settings):
            output_channel = mobilenet._make_divisible(c * width_mult, round_nearest)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    layer(input_channel, output_channel, stride=stride, expand_ratio=t)
                )
                input_channel = output_channel
            self.add_module("layer{}".format(j + 1), nn.Sequential(*layers))
        # building last several layers
        if self.version == "v2":
            self.head_conv = mobilenet.ConvBNReLU(
                input_channel, self.last_channel, kernel_size=1
            )

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes),
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        for j in range(len(self.settings)):
            x = getattr(self, "layer{}".format(j + 1))(x)
        if self.version == "v2":
            x = self.head_conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


class MobileNetEx(MobileNet):
    def __init__(self, width_mult=1.0, version="v1", outputs=[7], url=None):
        super(MobileNetEx, self).__init__(width_mult=width_mult, version=version)
        self.url = url
        self.outputs = outputs

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)
            if self.version == "v2":
                change_dict = {"features.0.": "conv1."}
                f_idx = 1
                for j, (t, c, n, s) in enumerate(self.settings):
                    for i in range(n):
                        change_dict[
                            "features.{}.".format(f_idx)
                        ] = "layer{}.{}.".format(j + 1, i)
                        f_idx += 1
                change_dict["features.{}.".format(f_idx)] = "head_conv."
                for k, v in list(checkpoint.items()):
                    for _k, _v in list(change_dict.items()):
                        if _k in k:
                            new_key = k.replace(_k, _v)
                            checkpoint[new_key] = checkpoint.pop(k)
            else:
                change_dict = {"features.Conv2d_0.conv.": "conv1."}
                f_idx = 1
                for j, (t, c, n, s) in enumerate(self.settings):
                    for i in range(n):
                        for z in range(2):
                            change_dict[
                                "features.Conv2d_{}.depthwise.{}".format(f_idx, z)
                            ] = "layer{}.{}.{}".format(j + 1, i, z)
                            change_dict[
                                "features.Conv2d_{}.pointwise.{}".format(f_idx, z)
                            ] = "layer{}.{}.{}".format(j + 1, i, z + 3)
                        f_idx += 1
                for k, v in list(checkpoint.items()):
                    for _k, _v in list(change_dict.items()):
                        if _k in k:
                            new_key = k.replace(_k, _v)
                            checkpoint[new_key] = checkpoint.pop(k)

                remove_dict = ["classifier."]
                for k, v in list(checkpoint.items()):
                    for _k in remove_dict:
                        if _k in k:
                            checkpoint.pop(k)

                org_checkpoint = self.state_dict()
                org_checkpoint.update(checkpoint)
                checkpoint = org_checkpoint

            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)

        outputs = []
        for j in range(len(self.settings)):
            level = j + 1  # only 1 conv before
            if level > max(self.outputs):
                break
            x = getattr(self, "layer{}".format(level))(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


@register
def MobileNetV1(outputs, **kwargs):
    return MobileNetEx(
        width_mult=1.0,
        version="v1",
        outputs=outputs,
        url="https://www.dropbox.com/s/kygo8l6dwah3djv/mobilenet_v1_1.0_224.pth?dl=1",
    )


@register
def MobileNetV2(outputs, **kwargs):
    return MobileNetEx(
        width_mult=1.0,
        version="v2",
        outputs=outputs,
        url=mobilenet.model_urls["mobilenet_v2"],
    )
