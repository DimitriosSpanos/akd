import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from scipy.stats import norm
import scipy

import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        pre_relu_out = out
        out = self.relu(out)

        return out, pre_relu_out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        pre_relu_out = out
        out = self.relu(out)

        return out, pre_relu_out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, low_dim=None, marginal_relu=False):
        super(ResNet, self).__init__()
        zero_init_residual = False
        groups = 1
        width_per_group = 64
        replace_stride_with_dilation = None
        norm_layer = None
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.list = [2048]
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if low_dim>0:
            self.low_dim = low_dim
            self.fea_dim = low_dim
            self.fc = nn.Linear(512 * block.expansion, low_dim)
        else:
            self.fea_dim = 512 * block.expansion
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.marginal_relu = marginal_relu
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, feat_s=None, preact=False):
        if not feat_s is None:
            x = self.avgpool(feat_s)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x, pre_relu_1 = self.layer1(x)
            x, pre_relu_2 = self.layer2(x)
            x, pre_relu_3 = self.layer3(x)
            fea3 = x
            x, pre_relu_4 = self.layer4(x)
            if self.marginal_relu:
                if isinstance(self.layer1[0], Bottleneck):
                    bn4 = self.layer4[-1].bn3
                elif isinstance(self.layer1[0], BasicBlock):
                    bn4 = self.layer4[-1].bn2
                margin4 = self.get_margin_from_BN(bn4)
                margin4 = margin4.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach()
                fea4 = torch.max(x, margin4)
            else:
                fea4 = x

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            if is_feat:
                return [fea3, fea4, ], x
            else:
                return x

    def get_margin_from_BN(self, bn):
        margin = []
        std = bn.weight.data
        mean = bn.bias.data
        for (s, m) in zip(std, mean):
            s = abs(s.item())
            m = m.item()
            if norm.cdf(-m / s) > 0.001:
                margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
            else:
                margin.append(-3 * s)

        return torch.FloatTensor(margin).to(std.device)

def resnet18T(num_classes=1000, low_dim=0, model_path=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, low_dim=low_dim, **kwargs)

    if model_path:
        w = torch.load(model_path)['model']
        new_w = OrderedDict()
        for k, v in w.items():
            new_w[k.replace('module.', '')] = v
        model.load_state_dict(new_w, strict=False)
        print('load weights from:', model_path)
    else:# if define model_path reload weights
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        model.load_state_dict(state_dict, strict=False)
        print('loaded weights from model zoo!')

    return model

def resnet34T(num_classes=1000, low_dim=0,model_path=None,**kargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, low_dim=low_dim, **kargs)
    state_dict = load_state_dict_from_url(model_urls['resnet34'], progress=True)
    model.load_state_dict(state_dict)
    return model


def resnet50T(num_classes=1000, low_dim=0,model_path=None,**kargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, low_dim=low_dim, **kargs)
    state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
    model.load_state_dict(state_dict)
    #w = torch.load('/home/xiaolongliu/project/Sub-KD-ImageNet/save/models/resnet50S_best_cos_decay_76.94.pth')
    new_w = OrderedDict()
    for k,v in state_dict.items():
        new_w[k.replace('module.', '')] = v
    model.load_state_dict(new_w, strict=False)
    return model

if __name__ == '__main__':
    net_G = resnet34T()
    sub_params = sum(p.numel() for p in net_G.parameters())
    print(sub_params)