import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.resnet import resnet101


class resnet_backbone(nn.Module):
    def __init__(self, num_class = 1, input_channel = 3, output_stride=16):
        super(resnet_backbone, self).__init__()
        if output_stride != 8 and output_stride != 16:
            raise ValueError("output stride can only be 8 or 16 for now")

        self.resnet = resnet101(pretrained=True, output_stride=output_stride)
        if input_channel == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3, bias=False)
        elif input_channel == 3:
            self.conv1 = self.resnet.conv1
        else:
            raise ValueError("input channel should be 3 or 1")

    def forward(self, x):

        x = self.conv1(x) #1, 320*320
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) #4, 80*80

        x = self.resnet.layer1(x)
        low_feat = x
        x = self.resnet.layer2(x) #8, 40*40
        x = self.resnet.layer3(x) #16, 20*20
        x = self.resnet.layer4(x) #32, 10*10

        return x, low_feat

class ASPP(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)

        return x

#ResNet101 as backbone 
class deeplabv3p(nn.Module):
    def __init__(self, input_channel=3, num_class=21, output_stride=16 ):
        super(deeplabv3p, self).__init__()
        self.feature_extractor = resnet_backbone(num_class=num_class, input_channel=input_channel, output_stride=output_stride)
        self.output_stride = output_stride

        #ASPP
        rates = [6, 12, 18]
        self.aspp1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.aspp1_bn = nn.BatchNorm2d(256)

        self.aspp2 = ASPP(2048, 256, rate=rates[0])
        self.aspp3 = ASPP(2048, 256, rate=rates[1])
        self.aspp4 = ASPP(2048, 256, rate=rates[2])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                        nn.Conv2d(2048, 256, 1, stride=1),
                                        nn.BatchNorm2d(256)
                                        )

        self.conv1 = nn.Conv2d(1280, 256, 1)
        self.bn1 = nn.BatchNorm2d(256)

        #channel reduction to 48.
        self.conv2 = nn.Conv2d(256, 48, 1)
        self.bn2 = nn.BatchNorm2d(48)


        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, num_class, kernel_size=1, stride=1))


    def forward(self, x):

        x, low_level_features = self.feature_extractor(x)
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)

        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)

        x = F.interpolate(x, scale_factor= self.output_stride//4, mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, scale_factor= self.output_stride//(self.output_stride//4), mode='bilinear', align_corners=True)

        return x


if __name__ == "__main__":
    model = DeepLabv3p(input_channel=3, num_class=1, output_stride=16).cuda()
    image = torch.randn(10, 3, 320, 320 ).cuda()
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())
