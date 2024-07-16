import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import *
from .heads import *
from .modules.cbam import CBAM
from model.modules.cfp import CFPModule
from timm.layers import trunc_normal_
import math
from model.modules.attentions import AA_kernel


class CrossAttention(nn.Module):
    def __init__(self, lower_c, higher_c) -> None:
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.aa_1 = AA_kernel(lower_c, lower_c)
        self.aa_2 = AA_kernel(higher_c, lower_c)
        self.up = nn.Upsample(size=2, mode="bilinear")

    def forward(self, x1, x2):
        att = self.avg_pool(x1)

        x1 = self.aa_1(x1)
        x2 = self.aa_2(x2)

        out = x1 + self.up(x2)
        return att * out


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = res2net50_v1b(pretrained=False)
        self.params = self.backbone.channels

        self.cross_1 = CrossAttention(64, 128)
        self.cross_2 = CrossAttention(128, 256)
        self.cross_3 = CrossAttention(256, 512)

        self.decoder = UPerHead(self.params, 128, 1)

        self.output_0 = nn.Conv2d(128, 1, 1)
        self.output_1 = nn.Conv2d(128, 1, 1)
        self.output_2 = nn.Conv2d(128, 1, 1)
        self.output_3 = nn.Conv2d(128, 1, 1)

        self.final = nn.Conv2d(5, 1, 3, 1, 1)

    def forward(self, x, warm_flag):
        enc_out = self.backbone(x)
        x0, x1, x2, x3 = enc_out

        x0 = self.cross_1(x0, x1)
        x1 = self.cross_1(x1, x2)
        x2 = self.cross_1(x2, x3)

        sub_masks, global_mask = self.decoder([x0, x1, x2, x3])

        if warm_flag:

            x_d1, x_d2, x_d3, x_d4 = sub_masks

            mask0 = self.output_0(x_d1)
            mask1 = self.output_1(x_d2)
            mask2 = self.output_2(x_d3)
            mask3 = self.output_3(x_d4)

            # global_mask = self.cfr(global_mask)

            mask0 = F.interpolate(
                mask0, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            mask1 = F.interpolate(
                mask1, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            mask2 = F.interpolate(
                mask2, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            mask3 = F.interpolate(
                mask3, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            global_mask = F.interpolate(
                global_mask, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            output = self.final(
                torch.cat([mask0, mask1, mask2, mask3, global_mask], dim=1)
            )
            return [mask0, mask1, mask2, mask3, global_mask], output

        else:
            global_mask = F.interpolate(
                global_mask, size=x.size()[2:], mode="bicubic", align_corners=False
            )
            return [], global_mask


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
