import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet
import torch.utils.model_zoo as model_zoo
from .rutils import register


class ResNet(resnet.ResNet):
    "Deep Residual Network - https://arxiv.org/abs/1512.03385"

    def __init__(
        self,
        layers=[3, 4, 6, 3],
        bottleneck=resnet.Bottleneck,
        outputs=[5],
        groups=1,
        width_per_group=64,
        url=None,
    ):
        self.stride = 128
        self.bottleneck = bottleneck
        self.outputs = outputs
        self.url = url

        # torchvision added support for ResNeXt in version 0.3.0,
        # and introduces additional args to torchvision.models.resnet constructor
        kwargs_common = {"block": bottleneck, "layers": layers}
        kwargs_extra = (
            {"groups": groups, "width_per_group": width_per_group}
            if torchvision.__version__ > "0.2.1"
            else {}
        )
        kwargs = {**kwargs_common, **kwargs_extra}
        super().__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


@register
def ResNet18(outputs, **kwargs):
    return ResNet(
        layers=[2, 2, 2, 2],
        bottleneck=resnet.BasicBlock,
        outputs=outputs,
        url=resnet.model_urls["resnet18"],
    )


@register
def ResNet34(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 6, 3],
        bottleneck=resnet.BasicBlock,
        outputs=outputs,
        url=resnet.model_urls["resnet34"],
    )


@register
def ResNet50(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 6, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        url=resnet.model_urls["resnet50"],
    )


@register
def ResNet101(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 23, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        url=resnet.model_urls["resnet101"],
    )


@register
def ResNet152(outputs, **kwargs):
    return ResNet(
        layers=[3, 8, 36, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        url=resnet.model_urls["resnet152"],
    )


@register
def ResNeXt50_32x4d(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 6, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        groups=32,
        width_per_group=4,
        url=resnet.model_urls["resnext50_32x4d"],
    )


@register
def ResNeXt101_32x8d(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 23, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        groups=32,
        width_per_group=8,
        url=resnet.model_urls["resnext101_32x8d"],
    )


@register
def WideResNet50_2(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 6, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        width_per_group=128,
        url=resnet.model_urls["wide_resnet50_2"],
    )


@register
def WideResNet101_2(outputs, **kwargs):
    return ResNet(
        layers=[3, 4, 23, 3],
        bottleneck=resnet.Bottleneck,
        outputs=outputs,
        width_per_group=128,
        url=resnet.model_urls["wide_resnet101_2"],
    )