from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torchstat import stat


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=11, stride=1, padding=10, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class multi_scaled_dilation_conv_block(nn.Module):
    # 多尺度预处理kernel
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1):
        super(multi_scaled_dilation_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(int((kernel_size - 1) / 2 * dilation)),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class bias_convolution(nn.Module):
    # 多方向的空洞卷积，提供每个像素不同方向的情况
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1, direction=''):
        # default is normal convolution
        super(bias_convolution, self).__init__()
        self.direction = direction
        self.padding_size = int((kernel_size - 1) * dilation)
        # self.direction_padding = nn.ReflectionPad2d(self.padding_size)
        self.direction_padding_LU = nn.ReflectionPad2d((self.padding_size, 0, self.padding_size, 0))
        self.direction_padding_RU = nn.ReflectionPad2d((0, self.padding_size, self.padding_size, 0))
        self.direction_padding_LD = nn.ReflectionPad2d((self.padding_size, 0, 0, self.padding_size))
        self.direction_padding_RD = nn.ReflectionPad2d((0, self.padding_size, 0, self.padding_size))

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # print(self.padding_size)
        # x = self.direction_padding(x)
        x_LU = self.direction_padding_LU(x)
        x_RU = self.direction_padding_RU(x)
        x_LD = self.direction_padding_LD(x)
        x_RD = self.direction_padding_RD(x)

        if self.direction == 'LU':
            # padding to left up
            return self.conv(x_LU)

        elif self.direction == 'LD':
            # padding to left down
            return self.conv(x_LD)

        elif self.direction == 'RU':
            # padding to right up
            return self.conv(x_RU)

        elif self.direction == 'RD':
            # padding to right down
            return self.conv(x_RD)

        else:
            # normal padding
            return self.conv(x)


class offset_convolution(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(offset_convolution, self).__init__()
        self.LU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LU')
        self.LD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='LD')
        self.RU_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RU')
        self.RD_bias_convolution = bias_convolution(ch_in=ch_in, ch_out=ch_out, kernel_size=7, dilation=1,
                                                    direction='RD')
        self.final_conv = nn.Conv2d(ch_out * 4, ch_out, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(ch_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        LU_BC = self.LU_bias_convolution(x)
        LD_BC = self.LD_bias_convolution(x)
        RU_BC = self.RU_bias_convolution(x)
        RD_BC = self.RD_bias_convolution(x)
        d = torch.cat((LU_BC, LD_BC, RU_BC, RD_BC), dim=1)
        d = self.final_conv(d)
        d = self.BN(d)
        d = self.activation(d)
        return d



class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积, here gate is query.
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1,并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class my_model4(nn.Module):
    # MDOAU-net
    def __init__(self, img_ch=1, output_ch=1):
        super(my_model4, self).__init__()
        # offset_convolution()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.multi_scale_1 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=3, dilation=1)
        self.multi_scale_2 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=5, dilation=1)
        self.multi_scale_3 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=7, dilation=2)
        self.multi_scale_4 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=11, dilation=2)
        self.multi_scale_5 = multi_scaled_dilation_conv_block(img_ch, 16, kernel_size=15, dilation=3)

        self.Conv1 = conv_block(ch_in=16 * 5, ch_out=8)
        self.Conv2 = conv_block(ch_in=8, ch_out=16)
        self.Conv3 = conv_block(ch_in=16, ch_out=32)
        self.Conv4 = conv_block(ch_in=32, ch_out=64)
        self.Conv5 = conv_block(ch_in=64, ch_out=128)

        self.o1 = offset_convolution(ch_in=8, ch_out=8)
        self.o2 = offset_convolution(ch_in=16, ch_out=16)
        self.o3 = offset_convolution(ch_in=32, ch_out=32)
        self.o4 = offset_convolution(ch_in=64, ch_out=64)

        self.Up5 = up_conv(ch_in=128, ch_out=64)
        self.Att5 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64, ch_out=32)
        self.Att4 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32, ch_out=16)
        self.Att3 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16, ch_out=8)
        self.Att2 = Attention_block(F_g=8, F_l=8, F_int=4)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.Conv_1x1_1 = nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, train_flag=False):
        # multi_scale_generator
        x_pre_1 = self.multi_scale_1(x)
        x_pre_2 = self.multi_scale_2(x)
        x_pre_3 = self.multi_scale_3(x)
        x_pre_4 = self.multi_scale_4(x)
        x_pre_5 = self.multi_scale_5(x)
        muti_scale_x = torch.cat((x_pre_1, x_pre_2, x_pre_3, x_pre_4, x_pre_5), dim=1)

        # encoding path
        x1 = self.Conv1(muti_scale_x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # offset convolution
        o1 = self.o1(x1)
        o2 = self.o2(x2)
        o3 = self.o3(x3)
        o4 = self.o4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=o4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=o3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=o2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=o1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        if train_flag:
            return d1
        else:
            return self.sigmoid(d1)


# model = my_model4(1, 1)
# # print(stat(model, (1, 512, 512))) # my_model4 = 4,086,649
# test_x = torch.rand(2, 1, 512, 512)
# print(model(test_x))
# print(model(test_x).shape)



