import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from models.shallow_net import count_parameters, get_coord_features
from rules.component_wrapper import ComponentWrapper


BN_MOMENTUM = 0.1


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)
        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_in_channels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_in_channels, num_channels)

        self.num_in_channels = num_in_channels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, num_blocks, num_in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_in_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_IN_CHANNELS({})".format(num_branches, len(num_in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        layers = list()
        layers.append(block(
            self.num_in_channels[branch_index], num_channels[branch_index],
            stride, None))

        self.num_in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(
                self.num_in_channels[branch_index],
                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_in_channels = self.num_in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_in_channels[j], num_in_channels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(num_in_channels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode="nearest")
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_out_channels_conv3x3 = num_in_channels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_in_channels[j], num_out_channels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_out_channels_conv3x3)
                                )
                            )
                        else:
                            num_out_channels_conv3x3 = num_in_channels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_in_channels[j], num_out_channels_conv3x3, 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_out_channels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_in_channels(self):
        return self.num_in_channels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck
}


class HighResolutionNet(nn.Module):
    def __init__(self, config):
        super(HighResolutionNet, self).__init__()
        self.in_channels = 32
        self.relu = nn.ReLU(inplace=True)

        # stem net
        self.conv1 = nn.Conv2d(9, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM)

        # stage 2
        self.stage2_config = config["STAGE2"]
        num_channels = self.stage2_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition1 = self._make_transition_layer([self.in_channels], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_config, num_channels)

        # stage 3
        self.stage3_config = config["STAGE3"]
        num_channels = self.stage3_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_config, num_channels)

        # stage 4
        self.stage4_config = config["STAGE4"]
        num_channels = self.stage4_config["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_config["BLOCK"]]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_config, num_channels, multi_scale_output=False)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=config["OUT_CHANNELS"],
            kernel_size=config["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if config["FINAL_CONV_KERNEL"] == 3 else 0)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i-num_branches_pre else in_channels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_in_channels,
                    multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            reset_multi_scale_output = True
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False

            modules.append(
                HighResolutionModule(
                    num_branches, block, num_blocks, num_in_channels, num_channels,
                    fuse_method, reset_multi_scale_output))
            num_in_channels = modules[-1].get_num_in_channels()

        return nn.Sequential(*modules), num_in_channels

    def gather_by_label(self, out, labels):
        max_label = torch.max(labels)
        labels = torch.unsqueeze(labels, dim=1)
        one_hot = torch.zeros([labels.shape[0], max_label + 1, labels.shape[2], labels.shape[3]]).to(out.device)
        one_hot = torch.scatter(one_hot, 1, labels, 1)[:, 1:, :, :]
        one_hot = one_hot.view(one_hot.shape[0], one_hot.shape[1], one_hot.shape[2] * one_hot.shape[3])

        out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3]).permute(0, 2, 1)
        out = torch.bmm(one_hot, out)
        norm = torch.sum(one_hot, dim=-1).unsqueeze(-1)
        out = torch.div(out, norm)
        return out

    def forward(self, x, label):
        x = torch.cat([x, get_coord_features(x)], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_list = []
        for i in range(self.stage2_config["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_config["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_config["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        out = self.final_layer(y_list[0])
        out = self.gather_by_label(out, label)
        return out


def get_config():
    config = {
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "NUM_BLOCKS": [1, 1],
            "NUM_CHANNELS": [32, 64],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "NUM_BLOCKS": [1, 1, 1],
            "NUM_CHANNELS": [32, 64, 128],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "NUM_BLOCKS": [1, 1, 1, 1],
            "NUM_CHANNELS": [32, 64, 128, 256],
            "BLOCK": "BASIC",
            "FUSE_METHOD": "SUM",
        },
        "FINAL_CONV_KERNEL": 3,
        "OUT_CHANNELS": 64,
    }
    return config


if __name__ == "__main__":
    torch.cuda.benchmark = True

    # Read data
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
    print(input_label.shape, input_label.max())
    model = HighResolutionNet(get_config())

    start = time.time()
    output = model(torch.zeros([1, 7, 512, 768]), input_label)
    print(count_parameters(model))
    print(output.shape)
    print(time.time() - start)
