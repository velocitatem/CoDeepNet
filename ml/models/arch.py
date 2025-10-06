import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class CNNConfig:
    """Configuration for individual CNN in Co-DeepNet architecture.

    Based on Table 4 from the paper:
    - Input layer: 1 feature map (for 1D data like DNA methylation or Iris features)
    - C1: Convolution layer 1 -> 3 feature maps
    - P1: Pooling layer 1 -> 3 feature maps
    - C2: Convolution layer 2 -> 9 feature maps
    - P2: Pooling layer 2 -> 9 feature maps
    - FC1: Fully connected layer 1 -> 120 neurons
    - FC2: Fully connected layer 2 -> 84 neurons
    """
    input_features: int = 4  # Iris has 4 features (or 6 CpG sites for DNA)
    conv1_out: int = 3
    conv2_out: int = 9
    fc1_out: int = 120
    fc2_out: int = 84
    output_dim: int = 1  # Regression output (age prediction) or 3 for Iris classification
    kernel_size: int = 3
    stride: int = 1
    activation: str = 'relu'


class CNN(nn.Module):
    """Single CNN component for Co-DeepNet.

    Architecture follows paper's Section 3.2 specification:
    - 1D convolutions for feature data (not images)
    - Max pooling with stride 1
    - Two fully connected layers
    """

    def __init__(self, config: CNNConfig):
        super(CNN, self).__init__()
        self.config = config

        # Convolutional layers (1D for tabular/sequential data)
        self.conv1 = nn.Conv1d(1, config.conv1_out, kernel_size=config.kernel_size,
                               stride=config.stride, padding=config.kernel_size//2)
        # Use stride=1 for pooling to maintain dimensions (as per paper)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv2 = nn.Conv1d(config.conv1_out, config.conv2_out,
                               kernel_size=config.kernel_size, stride=config.stride,
                               padding=config.kernel_size//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)

        # Activation
        self.activation = nn.ReLU() if config.activation == 'relu' else nn.Tanh()

        # Calculate flattened size after convolutions
        # After conv1 + pool1: input_features -> input_features - 1 (with stride=1 pooling)
        # After conv2 + pool2: input_features - 1 -> input_features - 2
        # For input_features=3: 3 -> 2 -> 1
        feature_size_after_pools = config.input_features - 2
        self.flattened_size = config.conv2_out * max(1, feature_size_after_pools)

        # Fully connected layers (for knowledge transmission, size may double)
        self.fc1 = nn.Linear(self.flattened_size, config.fc1_out)
        self.fc1_with_knowledge = nn.Linear(self.flattened_size * 2, config.fc1_out)
        self.fc2 = nn.Linear(config.fc1_out, config.fc2_out)
        self.fc_out = nn.Linear(config.fc2_out, config.output_dim)

    def forward(self, x, transferred_knowledge: Optional[torch.Tensor] = None):
        """Forward pass with optional knowledge transfer.

        Args:
            x: Input tensor [batch, features]
            transferred_knowledge: Feature map from inactive CNN [batch, features]

        Returns:
            output: Predictions [batch, output_dim]
            last_feature_map: For knowledge transmission [batch, conv2_out * feature_size]
        """
        batch_size = x.shape[0]

        # Add channel dimension for 1D conv: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.activation(self.conv1(x))
        x = self.pool1(x)

        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        # Flatten for FC layers
        last_feature_map = x.flatten(1).clone()  # [batch, conv2_out * feature_size]

        # Knowledge transmission: concatenate transferred knowledge
        if transferred_knowledge is not None:
            # Only use transferred knowledge if batch sizes match
            if transferred_knowledge.shape[0] == batch_size:
                # Detach transferred knowledge to avoid gradient issues across CNNs
                x = torch.cat([last_feature_map, transferred_knowledge.detach()], dim=1)
                # Use the FC layer designed for knowledge transmission
                x = self.activation(self.fc1_with_knowledge(x))
            else:
                # Batch size mismatch - skip knowledge transmission for this batch
                x = self.activation(self.fc1(last_feature_map))
        else:
            # Use the regular FC layer
            x = self.activation(self.fc1(last_feature_map))

        # Fully connected layers
        x = self.activation(self.fc2(x))
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
