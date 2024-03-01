import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vit import VisionTransformer, Reconstruct
from .pixlevel import PixLevelModule

class PPCDblock(nn.Module):
    def __init__(self, channel):
        super(PPCDblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.patch = nn.Sequential(nn.Conv2d(channel + 4, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                    nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, padding=0)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)
        out = self.patch(out)
        dilate1_out = self.conv1x1(out)
        dilate1_out = self.bn(dilate1_out)
        dilate1_out = self.relu(dilate1_out)
        dilate2_out = self.conv3x3(self.conv1x1(out))
        dilate2_out = self.bn(dilate2_out)
        dilate2_out = self.relu(dilate2_out)
        dilate3_out = self.conv5x5(self.conv1x1(out))
        dilate3_out = self.bn(dilate3_out)
        dilate3_out = self.relu(dilate3_out)
        dilate4_out = self.pooling(out)
        out_final = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out_final
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN1(nn.Module):
    def __init__(self, channel, map_size, pad):
        super(CNN1, self).__init__()
        self.conv = nn.Conv2d(channel, channel,
                              kernel_size=map_size, padding=pad)
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.relu(out)

class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class MTGT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.pix_module1 = PixLevelModule(64)
        self.pix_module2 = PixLevelModule(128)
        self.pix_module3 = PixLevelModule(256)
        self.pix_module4 = PixLevelModule(512)
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.PPCD_64 = PPCDblock(64)
        self.PPCD_128 = PPCDblock(128)
        self.PPCD_256 = PPCDblock(256)
        self.PPCD_512 = PPCDblock(512)
        self.x4_to_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_to_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_to_1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x1_to_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.conv_3 = CNN1(64, 3, 1)
        self.conv_5 = CNN1(64, 5, 2)
        self.x1_to_2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))
        self.x1_to_3 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))
        self.x1_to_4 = nn.Sequential(nn.Conv2d(64, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
                                      nn.ReLU(inplace=True))

        self.y5_y4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.y4_y3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.y3_y2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.y5_y4_y3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y4_y3_y2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.y5_y4_y3_y2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))

    def forward(self, x, text):
        x = x.float()  # x [2,3,224,224]
        x1 = self.inc(x)  # x1 [2, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)  # text4 [2 ,10, 512]
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)  # text4 [2 ,10, 256]
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)  # text4 [2 ,10, 128]
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # text4 [2 ,10, 64]
        y1 = self.downVit(x1, x1, text1)  # y1 [2 ,196, 64]
        x2 = self.down1(x1)  # x2 [2, 128, 122, 122]
        y2 = self.downVit1(x2, y1, text2)  # y2 [2 ,196, 128]
        x3 = self.down2(x2)  # x3 [2, 256, 56, 56]
        y3 = self.downVit2(x3, y2, text3)  # y3 [2 ,196, 256]
        x4 = self.down3(x3)  # x4 [2, 512, 28, 28]
        y4 = self.downVit3(x4, y3, text4)  # y4 [2 ,196, 512]
        x5 = self.down4(x4)  # x4 [2, 512, 14, 14]
        y4 = self.upVit3(y4, y4, text4, True)  # y4 [2 ,196, 512]
        y3 = self.upVit2(y3, y4, text3, True)  # y3 [2 ,196, 256]
        y2 = self.upVit1(y2, y3, text2, True)  # y2 [2 ,196, 128]
        y1 = self.upVit(y1, y2, text1, True)  # y1 [2 ,196, 64]
        y1_r = self.reconstruct1(y1)  # [2, 64, 224, 224]
        y2_r = self.reconstruct2(y2)  # [2, 128, 112, 112]
        y3_r = self.reconstruct3(y3)  # [2, 256, 56, 56]
        y4_r = self.reconstruct4(y4)  # [2, 512, 28, 28]
        y1_r = self.PPCD_64(y1_r)
        y2_r = self.PPCD_128(y2_r)
        y3_r = self.PPCD_256(y3_r)
        y4_r = self.PPCD_512(y4_r)
        y4_r_to_1 = self.x4_to_1(y4_r)  # [2, 64, 28, 28]
        y3_r_to_1 = self.x3_to_1(y3_r)  # [2, 64, 56, 56]
        y2_r_to_1 = self.x2_to_1(y2_r)  # [2, 64, 112, 112]
        y1_r_to_1 = self.x1_to_1(y1_r)  # [2, 64, 224, 224]

        y4_r_to_1_up = F.upsample(y4_r_to_1, size=y3_r.size()[2:], mode='bilinear')
        y4_r_to_1_up_map1 = self.conv_3(y4_r_to_1_up)
        y3_r_to_1_map1 = self.conv_3(y3_r_to_1)
        y4_r_to_1_up_map2 = self.conv_5(y4_r_to_1_up)
        y3_r_to_1_map2 = self.conv_5(y3_r_to_1)
        y5_4 = self.y5_y4(
            abs(y4_r_to_1_up - y3_r_to_1) + abs(y4_r_to_1_up_map1 - y3_r_to_1_map1) + abs(y4_r_to_1_up_map2 - y3_r_to_1_map2))  # [2, 64, 56, 56]

        y3_r_to_1_up = F.upsample(y3_r_to_1, size=y2_r.size()[2:], mode='bilinear')
        y3_r_to_1_up_map1 = self.conv_3(y3_r_to_1_up)
        y2_r_to_1_map1 = self.conv_3(y2_r_to_1)
        y3_r_to_1_up_map2 = self.conv_5(y3_r_to_1_up)
        y2_r_to_1_map2 = self.conv_5(y2_r_to_1)
        y4_3 = self.y4_y3(
            abs(y3_r_to_1_up - y2_r_to_1) + abs(y3_r_to_1_up_map1 - y2_r_to_1_map1) + abs(y3_r_to_1_up_map2 - y2_r_to_1_map2))  # [2, 64, 112, 112]

        y2_r_to_1_up = F.upsample(y2_r_to_1, size=y1_r.size()[2:], mode='bilinear')
        y2_r_to_1_up_map1 = self.conv_3(y2_r_to_1_up)
        y1_r_to_1_map1 = self.conv_3(y1_r_to_1)
        y2_r_to_1_up_map2 = self.conv_5(y2_r_to_1_up)
        y1_r_to_1_map2 = self.conv_5(y1_r_to_1)
        y3_2 = self.y3_y2(
            abs(y2_r_to_1_up - y1_r_to_1) + abs(y2_r_to_1_up_map1 - y1_r_to_1_map1) + abs(y2_r_to_1_up_map2 - y1_r_to_1_map2))  # [2, 64, 224, 224]

        y5_4_up = F.upsample(y5_4, size=y4_3.size()[2:], mode='bilinear')
        y5_4_up_map1 = self.conv_3(y5_4_up)
        y4_3_map1 = self.conv_3(y4_3)
        y5_4_up_map2 = self.conv_5(y5_4_up)
        y4_3_map2 = self.conv_5(y4_3)
        y5_4_3 = self.y5_y4_y3(abs(y5_4_up - y4_3) + abs(y5_4_up_map1 - y4_3_map1) + abs(y5_4_up_map2 - y4_3_map2))  # [2, 64, 112, 112]

        y4_3_up = F.upsample(y4_3, size=y3_2.size()[2:], mode='bilinear')
        y4_3_up_map1 = self.conv_3(y4_3_up)
        y3_2_map1 = self.conv_3(y3_2)
        y4_3_up_map2 = self.conv_5(y4_3_up)
        y3_2_map2 = self.conv_5(y3_2)
        y4_3_2 = self.y4_y3_y2(abs(y4_3_up - y3_2) + abs(y4_3_up_map1 - y3_2_map1) + abs(y4_3_up_map2 - y3_2_map2))  # [2, 64, 224, 224]

        y5_4_3_up = F.upsample(y5_4_3, size=y4_3_2.size()[2:], mode='bilinear')
        y5_4_3_up_map1 = self.conv_3(y5_4_3_up)
        y4_3_2_map1 = self.conv_3(y4_3_2)
        y5_4_3_up_map2 = self.conv_5(y5_4_3_up)
        y4_3_2_map2 = self.conv_5(y4_3_2)
        y5_4_3_2 = self.y5_y4_y3_y2(
            abs(y5_4_3_up - y4_3_2) + abs(y5_4_3_up_map1 - y4_3_2_map1) + abs(y5_4_3_up_map2 - y4_3_2_map2))  # [2, 64, 224, 224]

        subtraction224 = self.x1_to_1(y5_4_3_2 + y4_3_2 + y3_2)  # [2, 64, 224, 224]
        subtraction112 = self.x1_to_2(y5_4_3 + y4_3)  # [2, 128, 122, 122]
        subtraction56 = self.x1_to_3(y5_4)  # [2, 256, 56, 56]

        x1 = y1_r + x1 + subtraction224  # x1 [2, 64, 224, 224]
        x2 = y2_r + x2 + subtraction112  # x2 [2, 128, 122, 122]
        x3 = y3_r + x3 + subtraction56  # x3 [2, 256, 56, 56]
        x4 = y4_r + x4 # x4 [2, 512, 28, 28]
        x = self.up4(x5, x4)  # x [2, 256, 28, 28]
        x = self.up3(x, x3)  # x [2, 128, 56, 56]
        x = self.up2(x, x2)  # x [2, 64, 112, 112]
        x = self.up1(x, x1)  # x [2, 64, 224, 224]
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1  self.outc(x) [2, 1, 224, 224]
        return logits


