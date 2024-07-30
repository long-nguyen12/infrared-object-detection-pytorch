import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import *
from .heads import *


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = custom_res2net50_v1b(pretrained=False)
        self.params = self.backbone.channels

        self.decoder = DilationHead(self.params, 128)

        self.output_0 = nn.Conv2d(128, 1, 1)
        self.output_1 = nn.Conv2d(128, 1, 1)
        self.output_2 = nn.Conv2d(128, 1, 1)
        self.output_3 = nn.Conv2d(128, 1, 1)

    def forward(self, x, warm_flag=False):
        enc_out = self.backbone(x)
        x0, x1, x2, x3 = enc_out

        sub_masks, global_mask = self.decoder([x0, x1, x2, x3])

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
        global_mask = F.interpolate(
            global_mask, size=x.size()[2:], mode="bilinear", align_corners=False
        )

        # return [mask0], global_mask
        return [mask0, mask1, mask2, mask3], global_mask


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
