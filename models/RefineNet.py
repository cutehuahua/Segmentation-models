import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class RCU(nn.Module):

    def __init__(self, features):
        super(RCU, self).__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MRF(nn.Module):

    def __init__(self, out_feats, *shapes):
        super(MRF, self).__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        for i, shape in enumerate(shapes):
            feat, size = shape

            scale_factor = max_size // size
            if scale_factor != 1:
                self.add_module("resolve{}".format(i), nn.Sequential(
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
                ))
            else:
                self.add_module(
                    "resolve{}".format(i),
                    nn.Conv2d(feat, out_feats, kernel_size=3,
                              stride=1, padding=1, bias=False)
                )

    def forward(self, *xs):

        output = self.resolve0(xs[0])

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__("resolve{}".format(i))(x)

        return output


class CRP(nn.Module):

    def __init__(self, feats):
        super(CRP, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module("block{}".format(i), nn.Sequential(
                nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(feats, feats, kernel_size=3, stride=1, padding=1, bias=False)
            ))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class RefineNet_block(nn.Module):

    def __init__(self, features, *shapes):
        super(RefineNet_block, self).__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module("rcu{}".format(i), nn.Sequential(
                RCU(feats),
                RCU(feats)
            ))

        if len(shapes) != 1:
            self.mrf = MRF(features, *shapes)
        else:
            self.mrf = None

        self.crp = CRP(features)
        self.output_conv = RCU(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))
        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]
        out = self.crp(out)
        return self.output_conv(out)

class RefineNet(nn.Module):

    def __init__(self, input_shape, in_channel=3, num_classes=1, features=256,resnet_factory=models.resnet101, pretrained=True):

        super(RefineNet, self).__init__()

        input_channel, input_size = input_shape

        resnet = resnet_factory(pretrained=pretrained)

        #grayscale input

        if in_channel == 3:
            self.layer1 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )
        elif in_channel == 1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1
            )

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNet_block( 2 * features, (2 * features, input_size // 32))
        self.refinenet3 = RefineNet_block( features, (2 * features, input_size // 32), (features, input_size // 16))
        self.refinenet2 = RefineNet_block( features, (features, input_size // 16), (features, input_size // 8))
        self.refinenet1 = RefineNet_block( features, (features, input_size // 8), (features, input_size // 4))

        self.output_conv = nn.Sequential(
            RCU(features),
            RCU(features),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        return out
