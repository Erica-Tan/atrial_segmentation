import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),

            nn.Conv2d(
                output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            # nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size=kernel, stride=stride, bias=False
            )

    def forward(self, x):
        return self.upsample(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out