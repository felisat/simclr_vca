import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock



def resnet8():
    return ResNet(BasicBlock, [1,1,1,1], num_classes=10)

def apply_gn(model):
    for n, c in model.named_children():

        
        if isinstance(c, nn.Sequential) or \
                isinstance(c, torch.nn.modules.container.Sequential) or \
                isinstance(c, torchvision.models.resnet.BasicBlock):
            print("-->", n)
            apply_gn(c)
            
        if isinstance(c, nn.BatchNorm2d):
            print(n, c.num_features)
            setattr(model, n, torch.nn.GroupNorm(num_groups=2, num_channels=c.num_features))  


class ResNet8(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False):
        super(ResNet8, self).__init__()

        self.f = []
        for name, module in resnet8().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        if group_norm:
            apply_gn(self)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out



class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    def __init__(self, feature_dim, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=2):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.norm_layer = norm_layer
        self.cfg = [(1,  16//shrink, 1, 1),
                   (6,  24//shrink, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                   (6,  32//shrink, 3, 2),
                   (6,  64//shrink, 4, 2),
                   (6,  96//shrink, 3, 1),
                   (6, 160//shrink, 3, 2),
                   (6, 320//shrink, 1, 1)]


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = self.norm_layer(1280//shrink)
        self.linear = nn.Linear(1280//shrink, num_classes)

        self.g = nn.Sequential(nn.Linear(1280//shrink, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))



    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def f(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)










class LeNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 512)

        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), #nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def f(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class LeNet2(nn.Module):
    def __init__(self, feature_dim=128):
        super(LeNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512, 512)

        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), #nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def f(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class VGG9new(nn.Module):
    def __init__(self, feature_dim=128):
        super(VGG9new, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512, 512)

        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), #nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def f(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x)) 
        x = self.pool(F.relu(self.conv4(x)))

        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))

        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)




class VGG11(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False, batch_norm=False):
        super(VGG11, self).__init__()


        self.f = self.make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], batch_norm=batch_norm)
        # projection head


        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), #nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        if group_norm:
            apply_gn(self)


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)



class VGG16(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False, batch_norm=True):
        super(VGG16, self).__init__()


        self.f = self.make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        if group_norm:
            apply_gn(self)


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

