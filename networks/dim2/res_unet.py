import torch
import torch.nn as nn

from networks.dim2.modules import ResidualConv, Upsample


class ResUnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.residual_conv_1 = ResidualConv(input_channels, filters[0], 1)
        self.residual_conv_2 = ResidualConv(filters[0], filters[1], 2)
        self.residual_conv_3 = ResidualConv(filters[1], filters[2], 2)

        self.bridge = ResidualConv(filters[2], filters[3], 2)

        self.upsample_1 = Upsample(filters[3], filters[2], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[2] + filters[2], filters[2], 1)

        self.upsample_2 = Upsample(filters[2], filters[1], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[1] + filters[1], filters[1], 1)

        self.upsample_3 = Upsample(filters[1], filters[0], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[0] + filters[0], filters[0], 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encode
        x1 = self.residual_conv_1(x)
        x2 = self.residual_conv_2(x1)
        x3 = self.residual_conv_3(x2)

        # # Bridge
        x4 = self.bridge(x3)

        # # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


if __name__ == "__main__":
    x = torch.randn((2, 1, 272, 272))

    print(x.dtype)
    model = ResUnet(
        input_channels=1,
        num_classes=4,
    )

    output = model(x)
    print(output.shape)

    # for p in model.parameters():
    #     print(p.data.std(), p.ndim)
