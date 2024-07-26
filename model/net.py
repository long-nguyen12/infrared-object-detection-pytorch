import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import *
from .heads import *
from .modules.cbam import CBAM
from model.modules.cfp import CFPModule
from timm.layers import trunc_normal_
import math
from model.modules.attentions import AA_kernel, ECA, SELayer
from model.modules.conv_layers import Conv


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CrossAttention(nn.Module):
    def __init__(self, lower_c, higher_c) -> None:
        super().__init__()
        self.aa_1 = SpatialAttention()
        self.aa_2 = SpatialAttention()
        self.conv = Conv(higher_c, lower_c, 1, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * lower_c,
                out_channels=lower_c,
                kernel_size=1,
                groups=lower_c,
            ),
            nn.BatchNorm2d(lower_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode="bilinear")
        
        # x1 = self.aa_1(x1) * x1
        # x2 = self.aa_2(x2) * x2

        x = torch.cat([x1, x2], dim=1)
        x = self.down(x)
        # out = x1 * x + x2 * x
        return x


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = custom_res2net50_v1b(pretrained=False)
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
        
        # _x0 = self.cross_1(x0, x1)
        # _x1 = self.cross_2(x1, x2)
        # _x2 = self.cross_3(x2, x3)
        _x0 = x0
        _x1 = x1
        _x2 = x2

        sub_masks, global_mask = self.decoder([_x0, _x1, _x2, x3])

        # global_mask = self.refine(global_mask)

        if warm_flag:

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
            output = self.final(
                torch.cat([mask0, mask1, mask2, mask3, global_mask], dim=1)
            )
            return [mask0, mask1, mask2, mask3, global_mask], output

        else:
            
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
            
            return [mask0, mask1, mask2, mask3], global_mask
        


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
