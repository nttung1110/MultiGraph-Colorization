import time
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional
from PIL import Image
from rules.component_wrapper import ComponentWrapper
from models.utils import count_parameters, get_coord_features, gather_by_label, l2_normalize


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=True):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = functional.interpolate(
            x, size=self.size, scale_factor=self.scale_factor,
            mode=self.mode, align_corners=self.align_corners)
        return out


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, rate=1):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel,
            stride=stride, padding=padding, dilation=rate)
        self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0):
        super(TransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, features_dim, kernel=3, padding=1):
        super(DownBlock, self).__init__()
        self.features_dim = features_dim

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = ConvolutionBlock(in_channels, features_dim, 3, stride=2, padding=1)

        layers = nn.ModuleList()
        layers.append(ConvolutionBlock(features_dim, features_dim, kernel, padding=padding))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample(x)
        x = self.relu(x)

        out = self.conv(x)
        out = out + x
        out = self.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, features_dim, kernel=3, padding=1):
        super(UpBlock, self).__init__()
        self.features_dim = features_dim

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = TransposeBlock(in_channels, features_dim, 2, stride=2, padding=0)

        self.merge = ConvolutionBlock(features_dim, features_dim, 1)
        layers = nn.ModuleList()
        layers.append(ConvolutionBlock(features_dim, features_dim, kernel, padding=padding))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.upsample(x)
        z = x + y
        z = self.merge(z)
        z = self.relu(z)

        out = self.conv(z)
        out = out + z
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, dropout):
        super(UNet, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = ConvolutionBlock(in_channels + 2, 32, 5, padding=4, stride=1, rate=2)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        self.conv2 = ConvolutionBlock(256, 256, 3, padding=1)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up1 = UpBlock(64, 32)

        self.prediction = nn.Conv2d(32, 64, 5, padding=2)

    def forward(self, inputs, label):
        inputs = torch.cat([inputs, get_coord_features(inputs)], dim=1)
        out = self.relu(self.conv1(inputs))

        y1 = out
        out = y2 = self.down1(out)
        out = y3 = self.down2(out)
        out = self.down3(out)

        out = self.relu(self.conv2(out))
        out = self.dropout(out)

        out = self.up3(out, y3)
        out = self.up2(out, y2)
        out = self.up1(out, y1)

        out = self.prediction(out)
        out = gather_by_label(out, label)
        out = l2_normalize(out)
        assert not torch.isnan(out).any()
        return out


if __name__ == "__main__":
    torch.cuda.benchmark = True

    paths = ["D:/Data/GeekToys/coloring_data/simple_data/E/sketch_v3/A0001.png",
             "D:/Data/GeekToys/coloring_data/simple_data/E/color/A0001.tga"]
    sketch_image = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(np.array(Image.open(paths[1]).convert("RGB")), cv2.COLOR_RGB2BGR)

    color_image = np.where(
        np.stack([sketch_image == 0] * 3, axis=-1), np.zeros_like(color_image),
        color_image)
    color_image = cv2.resize(color_image, (768, 512), interpolation=cv2.INTER_NEAREST)

    # extract components
    component_wrapper = ComponentWrapper()
    input_label = component_wrapper.process(color_image)[0]
    input_label = torch.tensor(input_label).long().unsqueeze(0)

    model = UNet(8, 0.5)
    model.eval()
    print(count_parameters(model))

    start = time.time()
    print(input_label.shape, input_label.max())
    output = model(torch.zeros([1, 8, 512, 768]), input_label)
    print(output.size(), torch.mean(output))
    print(time.time() - start)
