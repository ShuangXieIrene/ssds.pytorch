import torchvision
from torchvision.models import shufflenetv2
import torch.utils.model_zoo as model_zoo
from .rutils import register


class ShuffleNetV2(shufflenetv2.ShuffleNetV2):
    def __init__(self, stages_repeats, stages_out_channels, outputs=[4], url=None):
        super(ShuffleNetV2, self).__init__(stages_repeats, stages_out_channels)
        self.outputs = outputs
        self.url = url

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        x = self.maxpool(self.conv1(x))

        outputs = []
        for i, stage in enumerate([self.stage2, self.stage3, self.stage4]):
            level = i + 2
            if level > max(self.outputs):
                break
            x = stage(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


@register
def ShuffleNetV2_x1(outputs, **kwargs):
    return ShuffleNetV2(
        [4, 8, 4],
        [24, 116, 232, 464, 1024],
        outputs=outputs,
        url=shufflenetv2.model_urls["shufflenetv2_x1.0"],
    )


@register
def ShuffleNetV2_x2(outputs, **kwargs):
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], outputs=outputs)
