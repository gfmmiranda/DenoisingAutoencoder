import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, depth=3, dropout_rate=0.3, activation=nn.ReLU):
        super(UNet, self).__init__()
        self.depth = depth
        self.pool = nn.MaxPool2d(2)
        self.dropout_rate = dropout_rate
        self.activation = activation

        # Encoder blocks
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for d in range(depth):
            out_channels_d = base_filters * (2 ** d)
            self.encoders.append(self.conv_block(prev_channels, out_channels_d))
            prev_channels = out_channels_d

        # Bottleneck
        self.bottleneck = self.conv_block(prev_channels, prev_channels * 2)
        self.bottleneck_channels = prev_channels * 2

        # Decoder blocks
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for d in reversed(range(depth)):
            out_channels_d = base_filters * (2 ** d)
            self.upconvs.append(nn.ConvTranspose2d(self.bottleneck_channels, out_channels_d, kernel_size=2, stride=2))
            self.decoders.append(self.conv_block(self.bottleneck_channels, out_channels_d))
            self.bottleneck_channels = out_channels_d

        # Final output layer
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            self.activation(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            self.activation(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        enc_features = []

        # Encoder forward
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        self._latent_shape = x.shape  # Store latent shape for later reference

        # Decoder forward
        for i in range(self.depth):
            skip_feat = enc_features[-(i+1)]
            x = self.upconvs[i](x)
            x = self._match_and_concat(skip_feat, x)
            x = self.decoders[i](x)

        return self.final(x)

    def _match_and_concat(self, enc_feat, x):
        if enc_feat.shape[2:] != x.shape[2:]:
            x = F.interpolate(x, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, enc_feat], dim=1)

    def summary(self, input_size):
        return summary(self, input_size=input_size)

    def latent_space_size(self, input_tensor):
        """Compute latent shape given an input tensor."""
        with torch.no_grad():
            _ = self.forward(input_tensor)
        return self._latent_shape
