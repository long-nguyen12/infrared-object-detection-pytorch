import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import *
from .heads import *
from .modules.cbam import CBAM
from model.modules.cfp import CFPModule
from timm.layers import trunc_normal_
import math
from model.modules.attentions import AA_kernel, ECA
from model.modules.conv_layers import Conv


class RefUnet(nn.Module):
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual


class CrossAttention(nn.Module):
    def __init__(self, lower_c, higher_c) -> None:
        super().__init__()
        self.aa_1 = ECA(lower_c)
        self.aa_2 = ECA(higher_c)
        self.conv = Conv(lower_c, higher_c, 1, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels=higher_c,
                out_channels=lower_c,
                kernel_size=1,
                groups=lower_c,
            ),
            nn.BatchNorm2d(lower_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        # x1 = self.conv(x1)
        # x1 = F.interpolate(x1, size=x2.size()[2:], mode="bilinear")
        # x1 = self.aa_1()
        x1 = self.aa_1(x1)
        x2 = self.aa_2(x2)
        
        x2 = self.down(x2)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode="bilinear")
        
        out = torch.cat([x1, x2], dim=1)

        return out


class SegmentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = custom_res2net50_v1b(pretrained=False)
        self.params = self.backbone.channels

        # self.cross_1 = CrossAttention(64, 128)
        # self.cross_2 = CrossAttention(128, 256)
        # self.cross_3 = CrossAttention(256, 512)

        self.decoder = UPerHead(self.params, 128, 1)
        
        self.refine_0 = RefUnet(1, 64)
        self.refine_1 = RefUnet(1, 64)
        self.refine_2 = RefUnet(1, 64)
        self.refine_3 = RefUnet(1, 64)

        self.output_0 = nn.Conv2d(128, 1, 1)
        self.output_1 = nn.Conv2d(128, 1, 1)
        self.output_2 = nn.Conv2d(128, 1, 1)
        self.output_3 = nn.Conv2d(128, 1, 1)

        self.final = nn.Conv2d(5, 1, 3, 1, 1)

    def forward(self, x, warm_flag):
        enc_out = self.backbone(x)
        x0, x1, x2, x3 = enc_out

        # x1 = self.cross_1(x0, x1)
        # x2 = self.cross_2(x1, x2)
        # x3 = self.cross_3(x2, x3)

        sub_masks, global_mask = self.decoder([x0, x1, x2, x3])

        # global_mask = self.refine(global_mask)

        if warm_flag:

            x_d1, x_d2, x_d3, x_d4 = sub_masks

            mask0 = self.output_0(x_d1)
            mask1 = self.output_1(x_d2)
            mask2 = self.output_2(x_d3)
            mask3 = self.output_3(x_d4)
            
            mask0 = self.refine_0(mask0)
            mask1 = self.refine_1(mask1)
            mask2 = self.refine_2(mask2)
            mask3 = self.refine_3(mask3)

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
            global_mask = F.interpolate(
                global_mask, size=x.size()[2:], mode="bilinear", align_corners=False
            )
            return [], global_mask


if __name__ == "__main__":
    model = SegmentNet()
    x = torch.randn(1, 3, 256, 256)
    outs, out = model(x)
    print(out.shape)
