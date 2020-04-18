#encoder-decoder architecture with skip connections as defined in the supplementary material
# of the paper Deep Image Prior
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, input_depth, num_filters, kernel_size = 3, pad_size=2):
        super(Downsample, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.cnn_1 = nn.Conv2d(input_depth, num_filters, kernel_size, stride=2, padding=pad_size)
        self.cnn_2 = nn.Conv2d(input_depth, num_filters, kernel_size, stride=1, padding=pad_size)
        self.padder_layer = nn.ReflectionPad2d(pad_size)

    def forward(self, x):
        x = self.padder_layer(x)
        x = self.cnn_1(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        x = self.padder_layer(x)
        x = self.cnn_2(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x


class Upsample(nn.Module):
    def __init__(self, input_depth, num_filters, num_filters_fixed, upsample_mode, kernel_size=3, pad_size=2):
        super(Upsample, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.batch_norm_fixed = nn.BatchNorm2d(num_filters_fixed)
        self.cnn_1 = nn.Conv2d(input_depth, num_filters, kernel_size, stride=2, padding=pad_size)
        self.cnn_2 = nn.Conv2d(input_depth, num_filters, kernel_size=1, stride=1, padding=pad_size)
        self.padder_layer = nn.ReflectionPad2d(pad_size)
        self.upsample_layer = nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        x = self.batch_norm(x)

        x = self.padder_layer(x)
        x = self.cnn_1(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        x = self.padder_layer(x)
        x = self.cnn_2(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)
        x = self.upsample_layer(x)

        return x;


class SkipConnection(nn.Module):
    def __init__(self, input_depth, num_filters, kernel_size=3, pad_size=2):
        super(SkipConnection, self).__init__()
        self.activation_function = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(num_filters)
        self.cnn = nn.Conv2d(input_depth, num_filters, kernel_size, stride=1, padding=pad_size)
        self.padder_layer = nn.ReflectionPad2d(pad_size)

    def forward(self, x):
        x = self.padder_layer(x)
        x = self.cnn(x)
        x = self.batch_norm(x)
        x = self.activation_function(x)

        return x;


class SkipArchitecture(nn.Module):
    def __init__(self, input_channels, output_channels, filters_down, filters_up, filters_skip,
                 kernel_size_down, kernel_size_up, kernel_size_skip, upsample_mode):
        super(SkipArchitecture, self).__init__()
        self.downModules = Downsample()
        self.upModules = Upsample()
        self.skip_connections = SkipConnection()
        self.sigmoid = nn.Sigmoid()
        self.number_of_channels = filters_down.length

    def forward(self, x):

        for i in range(self.number_of_channels):
            x = self.downs

        x = self.sigmoid(x)
        return x;

