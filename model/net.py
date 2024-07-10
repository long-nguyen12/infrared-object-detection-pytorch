import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import *
from heads import *


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = MiT("B0")
        self.params = self.backbone.channels

        self.decoder = FaPNHead(self.params, 128, 1)

        self.output_0 = nn.Conv2d(128, 1, 1)
        self.output_1 = nn.Conv2d(128, 1, 1)
        self.output_2 = nn.Conv2d(128, 1, 1)
        self.output_3 = nn.Conv2d(128, 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def forward(self, x):
        enc_out = self.backbone(x)

        sub_masks, global_mask = self.decoder(enc_out)

        x_d1, x_d2, x_d3, x_d4 = sub_masks

        mask0 = self.output_0(x_d1)
        mask1 = self.output_1(x_d2)
        mask2 = self.output_2(x_d3)
        mask3 = self.output_3(x_d4)

        mask0 = F.interpolate(
            mask0, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        mask1 = F.interpolate(
            mask1, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        mask2 = F.interpolate(
            mask2, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        mask3 = F.interpolate(
            mask3, size=x.size()[2:], mode="bilinear", align_corners=False
        )

        output = self.final(torch.cat([mask0, mask1, mask2, mask3], dim=1))
        return [mask0, mask1, mask2, mask3], output


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
