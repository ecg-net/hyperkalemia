from torch import nn
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, List
from torch import Tensor


class EffNet(nn.Module):
    # lightly retouched version of John's EffNet to add clean support for multiple output
    # layer designs as well as single-lead inputs
    def __init__(
        self,
        num_extra_inputs: int = 0,
        output_neurons: int = 1,
        channels: List[int] = (32, 16, 24, 40, 80, 112, 192, 320, 1280), 
        depth: List[int] = (1, 2, 2, 3, 3, 3, 3),
        dilation: int = 2,
        stride: int = 8,
        expansion: int = 6,
        embedding_hook: bool = False,
        input_channels: int = 1, 
        verbose: bool = False,
        embedding_shift: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.output_nerons = output_neurons

        # backwards compatibility change to prevent the addition of the output_neurons param
        # from breaking people's existing EffNet initializations
        if len(self.channels) == 10:
            self.output_nerons = self.channels[9]
            print(
                "DEPRECATION WARNING: instead of controlling the number of output neurons by changing the 10th item in the channels parameter, use the new output_neurons parameter instead."
            )

        self.depth = depth
        self.expansion = expansion
        self.stride = stride
        self.dilation = dilation
        self.embedding_hook = embedding_hook
        self.embedding_shift = embedding_shift

        if verbose:
            print("\nEffNet Parameters:")
            print(f"{self.input_channels=}")
            print(f"{self.channels=}")
            print(f"{self.output_nerons=}")
            print(f"{self.depth=}")
            print(f"{self.expansion=}")
            print(f"{self.stride=}")
            print(f"{self.dilation=}")
            print(f"{self.embedding_hook=}")
            print("\n")

        self.stage1 = nn.Conv1d(
            self.input_channels,
            self.channels[0],
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
        )  # 1 conv

        self.b0 = nn.BatchNorm1d(self.channels[0])

        self.stage2 = MBConv(
            self.channels[0], self.channels[1], self.expansion, self.depth[0], stride=2
        )

        self.stage3 = MBConv(
            self.channels[1], self.channels[2], self.expansion, self.depth[1], stride=2
        )

        self.Pool = nn.MaxPool1d(3, stride=1, padding=1)

        self.stage4 = MBConv(
            self.channels[2], self.channels[3], self.expansion, self.depth[2], stride=2
        )

        self.stage5 = MBConv(
            self.channels[3], self.channels[4], self.expansion, self.depth[3], stride=2
        )

        self.stage6 = MBConv(
            self.channels[4], self.channels[5], self.expansion, self.depth[4], stride=2
        )

        self.stage7 = MBConv(
            self.channels[5], self.channels[6], self.expansion, self.depth[5], stride=2
        )

        self.stage8 = MBConv(
            self.channels[6], self.channels[7], self.expansion, self.depth[6], stride=2
        )

        self.stage9 = nn.Conv1d(self.channels[7], self.channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.num_extra_inputs = num_extra_inputs

        self.fc = nn.Linear(self.channels[5] + num_extra_inputs, self.output_nerons)
        self.fc = nn.Linear(self.channels[8] + num_extra_inputs, self.output_nerons)
        
        
        self.fc.bias.data[0] = 0.275

    def forward(self, x: Tensor) -> Tensor:
        if self.num_extra_inputs > 0:
            x, extra_inputs = x

        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.Pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.Pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.AAP(x)[:, :, 0])
        if self.embedding_hook:
            return x
        
        else:
        
            if self.embedding_shift:
                delta_embedding_array = np.load('/workspace/imin/applewatch_potassium/delta_embedding_poolaverage_5second_to_5second.npy')
                delta_embedding_tensor = torch.tensor(delta_embedding_array, device='cuda')
                x += delta_embedding_tensor

            x = self.drop(x)

            if self.num_extra_inputs > 0:
                x = torch.cat((x, extra_inputs), 1)

            x = self.fc(x)
            return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expansion: int,
        activation: Callable,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.stride = stride
        self.conv1 = nn.Conv1d(in_channel, in_channel * expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channel * expansion,
            in_channel * expansion,
            kernel_size=3,
            groups=in_channel * expansion,
            padding=padding,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channel * expansion, out_channel, kernel_size=1, stride=1
        )
        self.b0 = nn.BatchNorm1d(in_channel * expansion)
        self.b1 = nn.BatchNorm1d(in_channel * expansion)
        self.d = nn.Dropout()
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x + y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y


class MBConv(nn.Module):
    def __init__(
        self, in_channel, out_channels, expansion, layers, activation=nn.ReLU6, stride=2
    ):
        super().__init__()

        self.stack = OrderedDict()
        for i in range(0, layers - 1):
            self.stack["s" + str(i)] = Bottleneck(
                in_channel, in_channel, expansion, activation
            )

        self.stack["s" + str(layers + 1)] = Bottleneck(
            in_channel, out_channels, expansion, activation, stride=stride
        )

        self.stack = nn.Sequential(self.stack)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stack(x)
        return self.bn(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out
