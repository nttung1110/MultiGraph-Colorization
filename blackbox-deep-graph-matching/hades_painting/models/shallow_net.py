import time
import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image
import sys
import sys 
sys.path.append("/mnt/ai_filestore/home/zed/multi-graph-matching/hades_painting")
from hades_painting.rules.component_wrapper import ComponentWrapper, resize_mask, get_component_color
from hades_painting.models.utils import count_parameters, get_coord_features, gather_by_label, gather_by_label_matrix
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, rate=1):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel, bias=True,
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

        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolutionBlock(in_channels, features_dim, 1))
        self.conv = ConvolutionBlock(features_dim, features_dim, kernel, padding=padding)

    def forward(self, x):
        x = self.downsample(x)
        x = self.relu(x)

        out = self.conv(x)
        out = self.relu(out + x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, features_dim, kernel=3, padding=1):
        super(UpBlock, self).__init__()
        self.features_dim = features_dim
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvolutionBlock(in_channels, features_dim, 1))
        self.merge = ConvolutionBlock(features_dim, features_dim, 1)
        self.conv = ConvolutionBlock(features_dim, features_dim, kernel, padding=padding)

    def forward(self, x, y):
        x = self.upsample(x)
        z = x + y
        z = self.merge(z)
        z = self.relu(z)

        out = self.conv(z)
        out = self.relu(out + z)
        return out


class GlobalUp(nn.Module):
    def __init__(self, in_channels, features_dim, out_channels, scale_factor):
        super(GlobalUp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.reduce1 = ConvolutionBlock(in_channels, features_dim, 3, padding=1)
        self.conv1 = ConvolutionBlock(features_dim, features_dim, 3, padding=1)

        self.reduce2 = ConvolutionBlock(features_dim, out_channels, 3, padding=1)
        self.conv2 = ConvolutionBlock(out_channels, out_channels, 3, padding=1)
        self.conv3 = ConvolutionBlock(out_channels, out_channels, 3, padding=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        out = self.relu(self.reduce1(x))
        out = self.relu(self.conv1(out))

        x = self.relu(self.reduce2(out))
        out = self.relu(self.conv2(x))
        out = self.conv3(out)
        out = self.relu(out + x)

        out = self.upsample(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, dropout, mode="infer"):
        super(UNet, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        assert self.mode in ["train", "infer"]

        self.conv1 = ConvolutionBlock(in_channels + 2, 64, 7, padding=3, stride=2)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)

        self.conv2 = ConvolutionBlock(256, 256, 3, padding=1)
        self.up3 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)

        self.upsample = GlobalUp(256, 128, 64, 8)
        self.gamma = nn.Parameter(torch.full([1], 0.01), requires_grad=True)

        self.up1 = nn.Sequential(
            TransposeBlock(64, 64, 2, stride=2),
            ConvolutionBlock(64, 64, 3, padding=1))
        self.prediction = nn.Conv2d(64, 64, 3, padding=1)

    def merge_with_global(self, x, y):
        y = self.upsample(y)
        out = x + self.gamma * y
        return out, x, y

    def forward(self, inputs, label):
        inputs = torch.cat([inputs, get_coord_features(inputs)], dim=1)

        out = y1 = self.relu(self.conv1(inputs))
        out = y2 = self.down1(out)
        out = self.down2(out)

        out = self.relu(self.conv2(out))
        out = y3 = self.dropout(out)

        out = self.up3(out, y2)
        out = self.up2(out, y1)
        out = self.up1(out)

        out, x, y = self.merge_with_global(out, y3)
        out = self.prediction(out)

        # out = gather_by_label_matrix(out, label)
        out = gather_by_label(out, label)
        assert not torch.isnan(out).any()

        if self.mode == "train":
            x = gather_by_label(x, label)
            y = gather_by_label(y, label)
            return out, x, y
        return out


if __name__ == "__main__":
    torch.cuda.benchmark = True

    paths = ["/mnt/ai_filestore/home/zed/multi-graph-matching/blackbox-deep-graph-matching/data/Geek_data/HOR02_Full_Formated/hor02_127/sketch/A0001.tga",
             "/mnt/ai_filestore/home/zed/multi-graph-matching/blackbox-deep-graph-matching/data/Geek_data/HOR02_Full_Formated/hor02_127/color/A0001.tga"]
    # sketch_image = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(np.array(Image.open(paths[1]).convert("RGB")), cv2.COLOR_RGB2BGR)

    # extract components
    component_wrapper = ComponentWrapper()
    input_label, input_components = component_wrapper.process(color_image, None, ComponentWrapper.EXTRACT_COLOR)
    input_label = resize_mask(input_label, input_components, (768, 512)).astype(np.int32)
    print(input_label.shape, input_label.dtype, input_label.min(), input_label.max())
    
    # cv2.imwrite("result_input_label.jpg", input_label)
    input_label = torch.from_numpy(input_label).long().unsqueeze(0)

    get_component_color(input_components, color_image, ComponentWrapper.EXTRACT_COLOR)

    # print(input_components)
    model = UNet(8, 0.5, "train")
    model.eval()
    print(count_parameters(model))

    start = time.time()
    output = model(torch.zeros([1, 8, 512, 768]), input_label)[0]
    print(output.size(), torch.mean(output))
    print(time.time() - start)
