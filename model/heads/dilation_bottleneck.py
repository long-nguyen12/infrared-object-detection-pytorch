import torch
from torch import Tensor, nn


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
        )


class DilationBottleneck(nn.Module):
    def __init__(self, c1, c2=128, scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                ConvModule(
                    c1,
                    c2,
                    (3, 3),
                    1,
                    p=(1 * d + 1, 1 * d + 1),
                    d=(d + 1, d + 1),
                    g=c2,
                )
                for d in scales
            ]
        )

        self.bottleneck = ConvModule(c1 + c2 * len(scales), c2, 3, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for stage in self.stages:
            outs.append(stage(x))

        outs = [x] + outs[::-1]
        out = self.bottleneck(torch.cat(outs, dim=1))
        return out
