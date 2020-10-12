import torch
import torch.nn.functional as F
from torch import nn


def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.1)
        if m.bias is not None:
            m.bias.data.fill_(0)
    else:
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


class FCNet(nn.Module):
    def __init__(self, image_size, channels, num_layers, layer_size, classes):
        super().__init__()
        assert image_size > 1
        assert channels >= 1
        assert num_layers > 1
        assert layer_size > 1
        assert classes > 1
        self.image_size = image_size
        self.channels = channels
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.classes = classes
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(
            nn.Linear(self.image_size * self.channels, self.layer_size))
        num_layers -= 1
        # remaining layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size))
        self.layers.append(nn.Linear(self.layer_size, self.classes))

    def forward(self, x):
        x = x.view(-1, self.image_size * self.channels)
        for fc_layer in self.layers[:-1]:
            x = torch.relu(fc_layer(x))
        return torch.log_softmax(self.layers[-1](x), dim=1)


class DCNet(nn.Module):
    def __init__(self,
                 image_size,
                 channels,
                 num_layers,
                 num_filters,
                 kernel_size,
                 classes,
                 batchnorm=True):
        super().__init__()
        assert image_size > 1
        assert channels >= 1
        assert classes > 1
        assert num_layers >= 1
        self.image_size = image_size
        self.channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.classes = classes
        self.batchnorm = batchnorm
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.bn_layers = nn.ModuleList()
        # assume, for simplicity, that we only use 'same' padding and stride 1
        padding = (self.kernel_size - 1) // 2
        c_in = self.channels
        c_out = self.num_filters
        for layer in range(self.num_layers):
            self.layers.append(
                nn.Conv2d(c_in,
                          c_out,
                          kernel_size=self.kernel_size,
                          stride=1,
                          padding=padding))
            c_in, c_out = c_out, c_out
            # c_in, c_out = c_out, c_out + self.filters_inc
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(c_out))
        self.layers.append(nn.Linear(c_out, self.classes))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorm:
                x = self.bn_layers[i](x)
        x_transformed = nn.functional.max_pool2d(x,
                                                 (x.size(2), x.size(3))).view(
                                                     x.size(0), -1)
        last_activations = self.layers[-1](x_transformed)
        return torch.log_softmax(last_activations, dim=1)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.classes = num_classes
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        self.current_planes = 16
        # self.current_size = 32

        self.conv1 = nn.Conv2d(3,
                               self.current_planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.current_planes)
        self.group1 = self._make_layer(block,
                                       self.planes[0],
                                       num_blocks=num_blocks[0],
                                       stride=self.strides[0])
        self.group2 = self._make_layer(block,
                                       self.planes[1],
                                       num_blocks=num_blocks[1],
                                       stride=self.strides[1])
        self.group3 = self._make_layer(block,
                                       self.planes[2],
                                       num_blocks=num_blocks[2],
                                       stride=self.strides[2])
        self.linear = nn.Linear(self.planes[2], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.current_planes, planes * block.expansion, stride))
            self.current_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.avg_pool2d(x, x.size(3))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return torch.log_softmax(x, dim=1)


def ResNet56():
    return ResNet(BasicBlock, [9, 9, 9])


def ResNet110():
    return ResNet(BasicBlock, [18, 18, 18])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
