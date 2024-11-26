import json
import os
from functools import cached_property
from typing import Literal

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Block of convolutions, batch normalizations and activations.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        block_size: int = 2,
        last_block: bool = False,
        batch_normalization: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Instantiates a block of convolutions, batch normalizations and activations.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Kernel size of the convolutional layers.
        block_size
            Total number of convolutional blocks.
        last_block
            Whether this is the last block in the network.
            If True, final activation, batch normalization and dropout is omitted.
        batch_normalization
            Whether batch normalization should be used.
        dropout_rate
            Dropout rate during the training.
        """
        super().__init__()

        layers = []
        for i in range(block_size):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                )
            )
            in_channels = out_channels

            if i < block_size - 1 or not last_block:
                if batch_normalization:
                    layers.append(torch.nn.BatchNorm1d(num_features=out_channels))

                layers.append(nn.ReLU())

                if dropout_rate > 0:
                    layers.append(torch.nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EmbeddingUnet(nn.Module):
    """
    U-Net neural network for embedding of piece contour points.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        kernel_size: int,
        block_size: int = 2,
        depth: int = 3,
        batch_normalization: bool = True,
        dropout_rate: float = 0.0,
    ):

        """
        Instantiates a U-Net embedding network.

        Parameters
        ----------
        input_dim
            Number of input channels.
        embedding_dim
            Number of output channels.
        kernel_size
            Kernel size of the convolutional layers.
        block_size
            Total number of convolutional blocks.
        depth
            Depth of the network.
        batch_normalization
            Whether batch normalization should be used.
        dropout_rate
            Dropout rate during the training.
        """
        super().__init__()

        self.down_blocks = []
        output_dim = embedding_dim
        for _ in range(depth):
            self.down_blocks.append(
                ConvBlock(
                    input_dim,
                    output_dim,
                    kernel_size,
                    block_size,
                    batch_normalization=batch_normalization,
                    dropout_rate=dropout_rate,
                )
            )
            input_dim = output_dim
            output_dim = output_dim * 2

        self.down_blocks = nn.ModuleList(self.down_blocks)

        self.up_blocks = []
        self.transpositions = []
        input_dim = output_dim // 2
        output_dim = output_dim // 4
        for i in range(depth - 1):
            self.transpositions.append(
                nn.ConvTranspose1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=2,
                    stride=2,
                )
            )
            last_block = i == depth - 2
            self.up_blocks.append(
                ConvBlock(
                    input_dim,
                    output_dim,
                    kernel_size,
                    block_size,
                    last_block,
                    batch_normalization,
                    dropout_rate,
                )
            )
            input_dim = output_dim
            output_dim = output_dim // 2

        self.up_blocks = nn.ModuleList(self.up_blocks)
        self.transpositions = nn.ModuleList(self.transpositions)

        self.initialize()

    def initialize(self) -> None:
        for layer in self.transpositions:
            nn.init.xavier_uniform(layer.weight.data)
            layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_results = []
        for block in self.down_blocks[:-1]:
            x = block(x)
            down_results.append(x)
            x = nn.MaxPool1d(kernel_size=2, stride=2)(x)

        x = self.down_blocks[-1](x)

        for i, block in enumerate(self.up_blocks):
            x = self.transpositions[i](x)
            padding = 4 ** (i + 1)
            down_res = down_results.pop()
            padding = (down_res.shape[2] - x.shape[2]) // 2
            down_res = down_res[:, :, padding : padding + x.shape[2]]

            x = torch.cat([x, down_res], dim=1)
            x = block(x)
        return x


class PairNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        kernel_size: int,
        depth: int = 3,
        batch_normalization: bool = True,
        dropout_rate: float = 0,
        shared_weights: bool = True,
    ):
        """
        Instantiates a U-Net embedding network.

        Parameters
        ----------
        embedding_dim
            Number of output channels.
        kernel_size
            Kernel size of the convolutional layers.
        depth
            Depth of the network.
        batch_normalization
            Whether batch normalization should be used.
        dropout_rate
            Dropout rate during the training.
        """
        super().__init__()

        self.embedding_network1 = EmbeddingUnet(
            147,
            embedding_dim,
            kernel_size,
            depth=depth,
            batch_normalization=batch_normalization,
            dropout_rate=dropout_rate,
        )

        if shared_weights:
            self.embedding_network2 = self.embedding_network1
        else:
            self.embedding_network2 = EmbeddingUnet(
                147,
                embedding_dim,
                kernel_size,
                depth=depth,
                batch_normalization=batch_normalization,
                dropout_rate=dropout_rate,
            )

        self.shared_weights = shared_weights

    @cached_property
    def padding(self) -> int:
        dim = 256
        device = next(self.parameters()).device
        x = (torch.ones(1, 147, dim).to(device), torch.ones(1, 147, dim).to(device))

        return int((dim - self.forward(x).shape[1]) / 2)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.shared_weights:
            x1_x2 = torch.cat([x[0], x[1]], dim=0)
            x1_x2_emb = self.embedding_network1(x1_x2)
            x1_emb = x1_x2_emb[: x1_x2_emb.shape[0] // 2]
            x2_emb = x1_x2_emb[x1_x2_emb.shape[0] // 2 :]

        else:
            x1_emb = self.embedding_network1(x[0])
            x2_emb = self.embedding_network2(x[1])

        matrix: torch.Tensor = x1_emb.transpose(1, 2) @ x2_emb
        return matrix


def load_model_config(
    model_id: str,
    path: str,
) -> dict:
    """Load configuration of model with given ID.

    Parameters
    ----------
    model_id
        ID of the model.
    path
        Path to the directory containing the model.

    Returns
    -------
    config

    """
    config_path = os.path.join(path, f"{model_id}_config.json")
    config: dict = json.load(open(config_path))
    return config


def load_model(
    model_id: str, path: str, checkpoint_version: Literal["best", "latest"] = "best"
) -> PairNetwork:
    """Load model with given ID.

    Parameters
    ----------
    model_id
        ID of the model.
    path
        Path to the directory containing the model.
    checkpoint_version
        Version of the checkpoint to load. One of "best" or "latest".
        "best" loads the checkpoint with best validation results, while "latest"
        loads the last checkpoint.

    Returns
    -------
    model

    """
    config = load_model_config(model_id, path)

    checkpoint_path = os.path.join(path, f"{model_id}_{checkpoint_version}")

    model = PairNetwork(**config["model"])
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    )

    return model
