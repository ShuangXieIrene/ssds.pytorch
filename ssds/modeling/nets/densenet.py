import re
import torch
import torch.nn as nn
from torchvision.models import densenet
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .rutils import register


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        memory_efficient=False,
        outputs=[],
        url=None,
    ):
        super(DenseNet, self).__init__()
        self.url = url
        self.outputs = outputs
        self.block_config = block_config

        # First convolution
        self.conv1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm", nn.BatchNorm2d(num_init_features)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = densenet._Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)

            pattern = re.compile(
                r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
            )
            for key in list(checkpoint.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    checkpoint[new_key] = checkpoint[key]
                    del checkpoint[key]

            change_dict = {
                "features.conv0.": "conv1.conv.",
                "features.norm0.": "conv1.norm.",
            }
            for i, num_layers in enumerate(self.block_config):
                change_dict[
                    "features.denseblock{}.".format(i + 1)
                ] = "denseblock{}.".format(i + 1)
                change_dict[
                    "features.transition{}.".format(i + 1)
                ] = "transition{}.".format(i + 1)
            for k, v in list(checkpoint.items()):
                for _k, _v in list(change_dict.items()):
                    if _k in k:
                        new_key = k.replace(_k, _v)
                        checkpoint[new_key] = checkpoint.pop(k)

            remove_dict = ["classifier.", "features.norm5."]
            for k, v in list(checkpoint.items()):
                for _k in remove_dict:
                    if _k in k:
                        checkpoint.pop(k)
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)

        outputs = []
        for j in range(len(self.block_config)):
            level = j + 1  # only 1 conv before
            if level > max(self.outputs):
                break
            if level > 1:
                x = getattr(self, "transition{}".format(level - 1))(x)
            x = getattr(self, "denseblock{}".format(level))(x)
            if level in self.outputs:
                outputs.append(x)

        return outputs


@register
def DenseNet121(outputs, **kwargs):
    return DenseNet(
        32, (6, 12, 24, 16), 64, outputs=outputs, url=densenet.model_urls["densenet121"]
    )


# print(DenseNet121([4]))
