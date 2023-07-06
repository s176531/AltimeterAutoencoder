from torch import nn
import torch
import functools
from typing import Type, List, Tuple

class Encoder(nn.Module):
    def __init__(
            self,
            input_channels: int,
            feature_dimension: int = 528,
            n_downsampling: int = 3,
            padding_type: nn.modules.padding._ReflectionPadNd | int = nn.ReflectionPad2d(1)
        ):
        super().__init__()
        self.input_channels = input_channels
        self.n_downsampling = n_downsampling
        self.feature_dimension = feature_dimension
        self.encoder = ResnetEncoder(
            input_n_channels = input_channels,
            out_dimension = feature_dimension,
            n_downsampling = n_downsampling,
            padding_type = padding_type
        )

    def get_out_frame_size(self, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Frame size is (Height, Width)"""
        return (
            frame_size[0]//(2**self.n_downsampling),
            frame_size[1]//(2**self.n_downsampling)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X = [batch_size, seq_len, in_channels, height, width]
        Returns:
            output = [batch_size, seq_len, feature_dimension, height/(2^n_downsampling), width/(2^n_downsampling)]
        """
        batch_size, frames, _, _, _ = x.shape

        feature_space = self.encoder(x.flatten(0, 1))
        _, channels, height, width = feature_space.shape
        feature_space = feature_space.reshape(batch_size, frames, channels, height, width)

        return feature_space

class Decoder(nn.Module):
    def __init__(
            self,
            output_channels: int,
            feature_dimension: int = 528,
            n_upsamples: int = 3,
            out_layer: nn.Module = nn.Tanh()
        ):
        super().__init__()
        self.output_channels = output_channels
        self.n_upsamples = n_upsamples
        self.feature_dimension = feature_dimension
        self.decoder = ResnetDecoder(
            output_n_channels = output_channels,
            feature_dimension = feature_dimension,
            n_upsamples = n_upsamples,
            out_layer_activation = out_layer
        )
    
    def get_out_frame_size(self, frame_size: Tuple[int, int]) -> Tuple[int, int]:
        """ Frame size is (Height, Width)"""
        return (
            int(frame_size[0] * 2**self.n_upsamples),
            int(frame_size[1] * 2**self.n_upsamples)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x = [batch_size, seq_len, feature_dimension, height, width]
        Returns:
            output = [batch_size, seq_len, feature_dimension, height*(2^n_upsamples), width*(2^n_upsamples)]
        """
        batch_size, frames, _, _, _ = x.shape

        output = self.decoder(x.flatten(0, 1))
        _, channels, height, width = output.shape

        return output.reshape(batch_size, frames, channels, height, width)

class ResnetEncoder(nn.Module):
    def __init__(
            self,
            input_n_channels: int,
            n_filters_last_conv: int = 64,
            out_dimension: int = 528,
            n_downsampling: int = 2,
            norm_layer: Type[nn.modules.batchnorm._BatchNorm] = nn.BatchNorm2d,
            use_dropout: bool = False,
            padding_type: nn.modules.padding._ReflectionPadNd | int = nn.ReflectionPad2d(1)
        ):
        """Construct a Resnet-based Encoder
        Parameters:
            input_n_channels (int)      -- the number of channels in input images
            n_filters_last_conv (int)   -- the number of filters in the last conv layer
            norm_layer                  -- normalization layer
            use_dropout (bool)          -- if use dropout layers
            padding_type (str)          -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_n_channels, n_filters_last_conv, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(n_filters_last_conv),
            nn.ReLU(True)
        ]

        # Add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                out_channels = out_dimension
            else:
                out_channels = n_filters_last_conv * mult * 2
            model.extend(
                [
                    nn.Conv2d(n_filters_last_conv * mult, out_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(out_channels),
                    nn.ReLU(True)
                ]
            )
        
        #Add 9 resnet-blocks
        for i in range(9):
            model.append(ResnetBlock(out_dimension, padding_type=padding_type, norm_layer=norm_layer, dropout=use_dropout, use_bias=use_bias))
        
        model.append(nn.ReLU())
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            X = [batch_size, input_n_channels, height, width]
        Returns:
            output = [batch_size, out_dimension, height/(2^n_downsampling), width/(2^n_downsampling)]
        """
        return self.model(x)

class ResnetDecoder(nn.Module):
    def __init__(
            self,
            output_n_channels: int,
            n_filters_last_conv: int = 64,
            feature_dimension: int = 528,
            n_upsamples: int = 2,
            norm_layer: Type[nn.modules.batchnorm._BatchNorm] = nn.BatchNorm2d,
            out_layer_activation: nn.Module = nn.Tanh()
        ):
        """Construct a Resnet-based Encoder
        Parameters:
            output_n_channels (int)     -- the number of channels in output images
            n_filters_last_conv (int)   -- the number of filters in the last conv layer
            norm_layer                  -- normalization layer
            out_layer_activation        -- activation function on output layers
        """
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #The first up-sampling layer
        mult = 2**n_upsamples
        model: List[nn.Module] = [
            nn.ConvTranspose2d(feature_dimension, int(n_filters_last_conv * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(n_filters_last_conv * mult / 2)),
            nn.ReLU(True)
        ]

        # Add upsampling layers
        for i in range(1, n_upsamples):
            mult = 2 ** (n_upsamples - i)
            model.extend(
                [
                    nn.ConvTranspose2d(
                        n_filters_last_conv * mult,
                        int(n_filters_last_conv * mult / 2),
                        kernel_size=3, 
                        stride=2,
                        padding=1, 
                        output_padding=1,
                        bias=use_bias
                    ),
                    norm_layer(int(n_filters_last_conv * mult / 2)),
                    nn.ReLU(True)
                ]
            )
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(n_filters_last_conv, output_n_channels, kernel_size=7, padding=0))
        model.append(out_layer_activation)

        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x = [batch_size, feature_dimension, height, width]
        Returns:
            output = [batch_size, output_n_channels, height*(2^n_upsamples), width*(2^n_upsamples)]
        """
        return self.model(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
            self,
            dim: int,
            padding_type: nn.modules.padding._ReflectionPadNd | int,
            norm_layer: Type[nn.modules.batchnorm._BatchNorm],
            dropout: float,
            use_bias: bool
        ):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (_ReflectionPadNd or padding size)  -- the name of padding layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        super().__init__()
        conv_block: List[nn.Module] = []
        p = 0
        if isinstance(padding_type, nn.modules.padding._ReflectionPadNd):
            conv_block.append(padding_type)
        else:
            p = padding_type

        conv_block.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True)
            ]
        )
        if dropout:
            conv_block.append(nn.Dropout(dropout))

        if isinstance(padding_type, nn.modules.padding._ReflectionPadNd):
            conv_block.append(padding_type)

        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias))
        conv_block.append(norm_layer(dim))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out