import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class CNNConfig:
    """Configuration for individual CNN in Co-DeepNet architecture.

    Supports dynamic depth for both convolutional and fully connected layers.

    Examples:
        # Default architecture (2 conv, 2 FC)
        config = CNNConfig()

        # Deeper network (4 conv, 3 FC)
        config = CNNConfig(conv_depth=4, fc_depth=3)

        # Shallow network (1 conv, 1 FC)
        config = CNNConfig(conv_depth=1, fc_depth=1)
    """
    input_features: int = 4  # Number of input features
    output_dim: int = 1  # Output dimension

    # Architecture depth
    conv_depth: int = 2  # Number of convolutional layers
    fc_depth: int = 2    # Number of fully connected layers

    # Conv layer parameters
    conv_base_channels: int = 3  # First conv layer channels (scales by 3x each layer)
    kernel_size: int = 3
    stride: int = 1

    # FC layer parameters
    fc_base_units: int = 120  # First FC layer units (scales by 0.7x each layer)

    # Activation
    activation: str = 'relu'


class CNN(nn.Module):
    """Single CNN component for Co-DeepNet.

    Dynamically builds CNN architecture based on config depth parameters:
    - 1D convolutions for tabular/sequential data
    - Max pooling with stride 1 after each conv
    - Configurable number of fully connected layers
    """

    def __init__(self, config: CNNConfig):
        super(CNN, self).__init__()
        self.config = config

        # Activation function
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.pool_enabled = []  # Track which layers should pool

        in_channels = 1
        feature_size = config.input_features

        for i in range(config.conv_depth):
            out_channels = config.conv_base_channels * (3 ** i)  # Scale by 3x each layer

            conv = nn.Conv1d(in_channels, out_channels,
                           kernel_size=config.kernel_size,
                           stride=config.stride,
                           padding=config.kernel_size // 2)

            # Only pool if we have enough features left (need at least 2 for kernel_size=2)
            can_pool = feature_size >= 2
            pool = nn.MaxPool1d(kernel_size=2, stride=1) if can_pool else nn.Identity()

            self.conv_layers.append(conv)
            self.pool_layers.append(pool)
            self.pool_enabled.append(can_pool)

            if can_pool:
                feature_size -= 1  # Pool with kernel=2, stride=1 reduces by 1

            in_channels = out_channels

        # Calculate flattened size after all conv/pool operations
        self.flattened_size = out_channels * max(1, feature_size)

        # Build fully connected layers dynamically
        self.fc_layers = nn.ModuleList()

        fc_in = self.flattened_size
        for i in range(config.fc_depth):
            fc_out = int(config.fc_base_units * (0.7 ** i))  # Scale down by 0.7x each layer
            self.fc_layers.append(nn.Linear(fc_in, fc_out))
            fc_in = fc_out

        # Special first FC layer for knowledge transmission (double input size -> same output as fc_layers[0])
        first_fc_out = int(config.fc_base_units)  # Same as fc_layers[0] output
        self.fc_with_knowledge = nn.Linear(self.flattened_size * 2, first_fc_out)

        # Output layer
        self.fc_out = nn.Linear(fc_in, config.output_dim)

    def forward(self, x, transferred_knowledge: Optional[torch.Tensor] = None):
        """Forward pass with optional knowledge transfer.

        Args:
            x: Input tensor [batch, features]
            transferred_knowledge: Feature map from inactive CNN [batch, flattened_size]

        Returns:
            output: Predictions [batch, output_dim]
            last_feature_map: For knowledge transmission [batch, flattened_size]
        """
        batch_size = x.shape[0]

        # Add channel dimension for 1D conv: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)

        # Pass through all convolutional layers
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = self.activation(conv(x))
            x = pool(x)

        # Flatten for FC layers
        last_feature_map = x.flatten(1).clone()

        # Knowledge transmission: concatenate transferred knowledge
        if transferred_knowledge is not None and transferred_knowledge.shape[0] == batch_size:
            # Detach transferred knowledge to avoid gradient issues across CNNs
            x = torch.cat([last_feature_map, transferred_knowledge.detach()], dim=1)
            # Use special FC layer that takes double input but outputs same size as fc_layers[0]
            x = self.activation(self.fc_with_knowledge(x))

            # Continue through remaining FC layers (start from index 1)
            for i in range(1, len(self.fc_layers)):
                x = self.activation(self.fc_layers[i](x))
        else:
            # No knowledge transmission - use all FC layers normally
            x = last_feature_map  # Start with the flattened feature map
            for fc in self.fc_layers:
                x = self.activation(fc(x))

        # Output layer
        output = self.fc_out(x)

        return output, last_feature_map


class CoDeepNet(nn.Module):
    """Cooperative Deep Neural Network (Co-DeepNet).

    Implements the cooperative CNN architecture from the paper:
    - Two CNNs (CNN1 and CNN2) alternate activity
    - Knowledge transmission every K iterations
    - Last feature map from inactive CNN feeds into active CNN
    """

    def __init__(self, config: Optional[CNNConfig] = None, knowledge_transmission_rate:int=20):
        super(CoDeepNet, self).__init__()

        # Use provided config or default
        if config is None:
            config = CNNConfig()

        # Initialize two identical CNNs
        self.cnn_A = CNN(config)
        self.cnn_B = CNN(config)

        # State tracking
        self.A_is_active = True  # CNN1 starts active
        self.last_feature_map_A = None
        self.last_feature_map_B = None
        self.iteration = 0
        self.knowledge_transmission_rate = knowledge_transmission_rate

    def switch_active_cnn(self):
        """Switch between CNN1 and CNN2 after knowledge transmission rate iterations."""
        self.A_is_active = not self.A_is_active

    def forward(self, x):
        """Forward pass through active CNN with knowledge from inactive CNN.

        Args:
            x: Input tensor [batch, features]

        Returns:
            output: Predictions from active CNN
        """
        if self.A_is_active:
            # CNN A is active, use knowledge from B
            output, self.last_feature_map_A = self.cnn_A(x, self.last_feature_map_B)
        else:
            # CNN B is active, use knowledge from A
            output, self.last_feature_map_B = self.cnn_B(x, self.last_feature_map_A)

        return output

    def should_switch(self):
        """Check if it's time to switch CNNs based on iteration count."""
        return (self.iteration + 1) % self.knowledge_transmission_rate == 0

    def get_active_cnn(self):
        """Get the currently active CNN for parameter updates."""
        return self.cnn_A if self.A_is_active else self.cnn_B
