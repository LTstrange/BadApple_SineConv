import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class BAN_encoder(nn.Module):
    def __init__(self):
        super(BAN_encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(4, 8, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(8, 8, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(8, 8, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(8, 8, (3, 3), padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        x = x.view(-1, 2048)

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        return x


class SineLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.fc = nn.Linear(self.in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.fc.weight.uniform_(-1 / self.in_features,
                                        1 / self.in_features)
            else:
                self.fc.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.fc(x))


class SineConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: tuple, padding=0, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.conv = nn.Conv2d(self.in_channels, out_channels, kernel_size, padding=padding, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.conv.weight.uniform_(-1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]),
                                          1 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
            else:
                self.conv.weight.uniform_(
                    -np.sqrt(6 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])) / self.omega_0,
                    np.sqrt(6 / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.conv(x))


class Sine_decoder(nn.Module):
    def __init__(self):
        super(Sine_decoder, self).__init__()

        self.fc1 = SineLinear(128, 512, is_first=True)
        self.fc2 = SineLinear(512, 2048, is_first=False)

        self.net = []
        self.net.append(SineConv(8, 16, (5, 5), padding=2, is_first=False))
        self.net.append(nn.PixelShuffle(2))
        # self.net.append(nn.BatchNorm2d(4))
        for i in range(3):
            self.net.append(SineConv(4, 16, (5, 5), padding=2, is_first=False))
            self.net.append(nn.PixelShuffle(2))
            # self.net.append(nn.BatchNorm2d(4))

        self.net.append(SineConv(4, 32, (5, 5), padding=2, is_first=False))
        self.net.append(nn.PixelShuffle(2))
        # self.net.append(nn.BatchNorm2d(8))
        self.net = nn.Sequential(*self.net)

        self.outer_conv = nn.Conv2d(8, 1, (7, 7), padding=3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 8, 16, 16)

        x = self.net(x)

        output = torch.sigmoid(self.outer_conv(x))

        return output


class BAN_decoder(nn.Module):
    def __init__(self):
        super(BAN_decoder, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 2048)

        self.net = []
        self.net.append(nn.Conv2d(8, 16, (5, 5), padding=2))
        self.net.append(nn.PixelShuffle(2))
        self.net.append(nn.BatchNorm2d(4))
        for i in range(3):
            self.net.append(nn.Conv2d(4, 16, (5, 5), padding=2))
            self.net.append(nn.PixelShuffle(2))
            self.net.append(nn.BatchNorm2d(4))
        self.net.append(nn.Conv2d(4, 32, (5, 5), padding=2))
        self.net.append(nn.PixelShuffle(2))
        self.net.append(nn.BatchNorm2d(8))
        self.net = nn.Sequential(*self.net)

        self.conv6 = nn.Conv2d(8, 1, (7, 7), padding=3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = x.view(-1, 8, 16, 16)
        x = self.net(x)

        x = torch.sigmoid(self.conv6(x))
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = BAN_encoder()
        self.decoder = BAN_decoder()

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return x, decode
