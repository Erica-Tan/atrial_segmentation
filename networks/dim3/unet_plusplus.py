from torch import nn
from monai.networks.nets import BasicUNetPlusPlus


class NestedUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, features, norm):
        super(NestedUNet, self).__init__()

        self.model = BasicUNetPlusPlus(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            norm=norm
        )

    def forward(self, x):
        return self.model(x)[0]