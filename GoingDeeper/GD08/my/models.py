# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, filters, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.downsample = downsample
        # 만약 downsample이라면 identity branch에 1x1 conv 적용하여 채널 수와 spatial size 조정
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False)

        # main branch
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, filters // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        # kernel_size=3, padding=1로 'same' padding 효과
        self.conv2 = nn.Conv2d(filters // 2, filters // 2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(filters // 2, momentum=0.9)
        self.conv3 = nn.Conv2d(filters // 2, filters, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample_conv(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity
        return out
    
class HourglassModule(nn.Module):
    def __init__(self, order, filters, num_residual):
        super(HourglassModule, self).__init__()
        self.order = order

        # Up branch: BottleneckBlock 1회 + num_residual회 반복
        self.up1_0 = BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
        self.up1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # Low branch: MaxPool + num_residual BottleneckBlock
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # Recursive hourglass or additional BottleneckBlocks
        if order > 1:
            self.low2 = HourglassModule(order - 1, filters, num_residual)
        else:
            self.low2_blocks = nn.Sequential(*[
                BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
                for _ in range(num_residual)
            ])

        # 후처리 BottleneckBlock 반복
        self.low3_blocks = nn.Sequential(*[
            BottleneckBlock(in_channels=filters, filters=filters, stride=1, downsample=False)
            for _ in range(num_residual)
        ])

        # UpSampling (최근접 보간법)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # up branch
        up1 = self.up1_0(x)
        up1 = self.up1_blocks(up1)

        # low branch
        low1 = self.pool(x)
        low1 = self.low1_blocks(low1)
        if self.order > 1:
            low2 = self.low2(low1)
        else:
            low2 = self.low2_blocks(low1)
        low3 = self.low3_blocks(low2)
        up2 = self.upsample(low3)

        return up2 + up1

class LinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        # He (Kaiming) 초기화 적용
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackedHourglassNetwork(nn.Module):
    def __init__(self, input_shape=(256, 256, 3), num_stack=4, num_residual=1, num_heatmap=16):
        super(StackedHourglassNetwork, self).__init__()
        self.num_stack = num_stack

        in_channels = input_shape[2]  # 3
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)

        # Bottleneck blocks 초기화
        # BottleneckBlock의 첫번째 호출: 64 → 128, downsample=True
        self.bottleneck1 = BottleneckBlock(in_channels=64, filters=128, stride=1, downsample=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 두 번째: 128 → 128, downsample=False
        self.bottleneck2 = BottleneckBlock(in_channels=128, filters=128, stride=1, downsample=False)
        # 세 번째: 128 → 256, downsample=True
        self.bottleneck3 = BottleneckBlock(in_channels=128, filters=256, stride=1, downsample=True)

        # 스택 구성 요소들
        self.hourglass_modules = nn.ModuleList()
        self.residual_modules = nn.ModuleList()  # hourglass 후 residual block들 (num_residual회)
        self.linear_layers = nn.ModuleList()
        self.heatmap_convs = nn.ModuleList()
        # 마지막 스택을 제외한 중간 피쳐 결합용 1x1 conv
        self.intermediate_convs = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        for i in range(num_stack):
            # order=4인 hourglass 모듈 (앞에서 정의한 HourglassModule 사용)
            self.hourglass_modules.append(HourglassModule(order=4, filters=256, num_residual=num_residual))
            # hourglass 후 residual block들
            self.residual_modules.append(nn.Sequential(*[
                BottleneckBlock(in_channels=256, filters=256, stride=1, downsample=False)
                for _ in range(num_residual)
            ]))
            # Linear layer: 1x1 conv + BN + ReLU (앞에서 정의한 LinearLayer 사용)
            self.linear_layers.append(LinearLayer(in_channels=256, out_channels=256))
            # 최종 heatmap을 생성하는 1x1 conv
            self.heatmap_convs.append(nn.Conv2d(256, num_heatmap, kernel_size=1, stride=1, padding=0))

            if i < num_stack - 1:
                self.intermediate_convs.append(nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.intermediate_outs.append(nn.Conv2d(num_heatmap, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.bottleneck1(x)
        x = self.pool(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)

        outputs = []
        for i in range(self.num_stack):
            hg = self.hourglass_modules[i](x)
            res = self.residual_modules[i](hg)
            lin = self.linear_layers[i](res)
            heatmap = self.heatmap_convs[i](lin)
            outputs.append(heatmap)

            if i < self.num_stack - 1:
                inter1 = self.intermediate_convs[i](lin)
                inter2 = self.intermediate_outs[i](heatmap)
                x = inter1 + inter2  # 다음 스택의 입력으로 사용

        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SimpleBaselinePose(nn.Module):
    """
    Simple Baselines for Human Pose Estimation and Tracking
    (Xiao et al., ECCV 2018) 구현.

    - Backbone: ResNet-50 / 101 / 152
    - Deconv: 3개의 ConvTranspose2d (kernel=4, stride=2, padding=1)
              채널 수는 모두 256
    - Final: 1x1 Conv → num_joints heatmap
    """

    def __init__(
        self,
        num_joints: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        in_channels: int = 3,
        deconv_with_bias: bool = False,
    ):
        super(SimpleBaselinePose, self).__init__()
        self.inplanes = 64
        self.num_joints = num_joints
        self.deconv_with_bias = deconv_with_bias

        # -----------------------
        # 1. ResNet backbone
        # -----------------------
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            resnet = models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 입력 채널이 3이 아닐 수도 있는 경우를 대비
        if in_channels != 3:
            # 기존 weight 형태: (64, 3, 7, 7)
            old_weight = resnet.conv1.weight
            resnet.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            # 단순하게 채널 수 맞춰서 평균 복제 (필요하면 더 정교하게 바꿔도 됨)
            if in_channels > 3:
                resnet.conv1.weight.data[:, :3, :, :] = old_weight.data
                if in_channels > 3:
                    for c in range(3, in_channels):
                        resnet.conv1.weight.data[:, c:c+1, :, :] = old_weight.data[:, :1, :, :]
            else:
                # in_channels < 3 인 경우: 앞 in_channels만 사용
                resnet.conv1.weight.data = old_weight.data[:, :in_channels, :, :]

        # ResNet의 stem + layer1~4 그대로 사용
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # layer4의 output 채널 수는 ResNet-50/101/152 모두 2048
        self.inplanes = 2048

        # -----------------------
        # 2. Deconv layers (3개)
        # -----------------------
        # 논문 구현은 kernel_size = 4, stride = 2, padding = 1 로 3단 업샘플링
        # => 최종적으로 입력 해상도의 1/4 크기 heatmap
        num_deconv_layers = 3
        num_deconv_filters = [256, 256, 256]
        num_deconv_kernels = [4, 4, 4]

        self.deconv_layers = self._make_deconv_layer(
            num_layers=num_deconv_layers,
            num_filters=num_deconv_filters,
            num_kernels=num_deconv_kernels,
        )

        # -----------------------
        # 3. Final 1x1 conv → heatmaps
        # -----------------------
        self.final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=num_joints,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # (옵션) weight 초기화 – 필요하면 더 정교하게 할 수 있음
        self._init_weights()

    # --------------------------------------------------------------------- #
    #  Helper methods
    # --------------------------------------------------------------------- #
    def _get_deconv_cfg(self, deconv_kernel):
        """kernel 사이즈에 따라 padding, output_padding 설정"""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Unsupported deconv_kernel: {deconv_kernel}")
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """3개의 deconv layer 쌓기"""
        assert num_layers == len(num_filters)
        assert num_layers == len(num_kernels)

        layers = []
        in_channels = self.inplanes

        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
            out_channels = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _init_weights(self):
        """deconv와 final_layer 초기화 (backbone은 torchvision이 함)"""
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # --------------------------------------------------------------------- #
    #  Forward
    # --------------------------------------------------------------------- #
    def forward(self, x):
        """
        x: (B, C, H, W)  예: (B, 3, 256, 192) or (B, 3, 256, 256)
        출력: (B, num_joints, H/4, W/4)
        """

        # ResNet stem
        x = self.conv1(x)      # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # /4

        # ResNet backbone
        x = self.layer1(x)     # /4
        x = self.layer2(x)     # /8
        x = self.layer3(x)     # /16
        x = self.layer4(x)     # /32

        # 3× deconv: /32 → /16 → /8 → /4
        x = self.deconv_layers(x)

        # 최종 heatmap
        x = self.final_layer(x)  # (B, num_joints, H/4, W/4)

        return x
