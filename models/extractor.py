import math
import torch.nn as nn


def get_extractor(args):
    """
    Get the proper extractor type.
    :param args: argparse.Namespace
    :return: FeatureExtractor
    """
    if args.extractor == "danq":
        return DanQ(
            args.in_channels,
            args.out_channels,
            args.filter_length,
            args.pool_length,
            args.pool_stride,
            args.input_length,
            lstm=args.lstm
        )

    elif args.extractor == "sae":
        return SAE(
            args.in_channels,
            args.intermediate_size,
            args.filter_length,
            args.pool_length,
            args.pool_stride
        )

    raise NotImplementedError("{} not implemented.".format(args.extractor))


class FeatureExtractor:
    def initialize_weights(self):
        raise NotImplementedError("!")


class DanQ(FeatureExtractor, nn.Module):
    """
    See original at: https://github.com/uci-cbcl/DanQ
    """
    def __init__(self, in_channels, out_channels, filter_length, pool_length, pool_stride, input_length, lstm=True):
        """
        Construcotr.
        :param in_channels: int, number of input channels to 1d convolution.
            - SNPs should be 3 channels.
            - Full sequence should be 4 channels.
        :param out_channels: int, desired output channels from 1d convolution.
        :param filter_length: int, length of the 1d convolution filter.
        :param pool_length: int, length of the pooling kernel.
        :param pool_stride: int, stride of the pooling kernel.
        :param input_length: int, the length of the input vector.
        :param lstm: bool
        """
        super(DanQ, self).__init__()

        computed_length = input_length - filter_length + 1
        computed_length = math.ceil((computed_length - pool_length + 1) / pool_stride)

        self.use_lstm = lstm

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=filter_length),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_length, stride=pool_stride),
            nn.Dropout(0.3)
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(computed_length, 32, 1, bidirectional=True)
            self.dropout = nn.Dropout(0.6)
        else:
            self.lstm = None
            self.dropout = None

    def forward(self, x):
        x = self.model(x)

        if self.use_lstm:
            x, _ = self.lstm(x)
            x = self.dropout(x)

        return x

    def weight_func(self, m):
        # Weight initialization not using the JASPAR Motifs.
        if isinstance(m, nn.Conv2d):
            m.weight.data.uniform_(-0.05, 0.05)
            m.bias.data.zero_()

        # Original implementation doesn't deal with LSTM weights.

    def initialize_weights(self):
        self.model.apply(self.weight_func)


class SAE(FeatureExtractor, nn.Module):
    """
    Stacked autoencoder.
    """
    def __init__(self, in_channels, intermediate_size, filter_length, pool_length, pool_stride):
        super(SAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 2 * intermediate_size, kernel_size=filter_length),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_length, stride=pool_stride),
            nn.Conv1d(2 * intermediate_size, intermediate_size, kernel_size=filter_length),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(intermediate_size, 2 * intermediate_size, kernel_size=filter_length),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(2 * intermediate_size, in_channels, kernel_size=filter_length),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded, self.decoder(encoded)

    def initialize_weights(self):
        """
        Stick with default initialization.
        """
        pass