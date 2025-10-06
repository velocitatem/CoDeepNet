import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from arch import CoDeepNet, CNNConfig

# create logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# print all info messages
logger.info("Initiated Co-DeepNet training loop")


class CoDeepNetTrainer:
    """Trainer implementing Algorithm 1 from the paper: Cooperation between CNNs.

    Key features:
    - Two CNNs alternate activity every K iterations
    - Knowledge transmission via last feature map
    - Active CNN trains while inactive CNN provides knowledge
    """

    def __init__(self, model, train_loader, val_loader=None, log_dir="./tensorboard",
                 knowledge_transmission_rate=20):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer only for active CNN parameters
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # For regression; use CrossEntropyLoss for classification

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        # Co-DeepNet specific
        self.knowledge_transmission_rate = knowledge_transmission_rate
        self.model.knowledge_transmission_rate = knowledge_transmission_rate

    def train_epoch(self, epoch):
        """Train for one epoch with cooperative CNN mechanism."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Forward pass through active CNN
            self.optimizer.zero_grad()
            output = self.model(data)

            # Calculate loss
            loss = self.criterion(output.squeeze(), target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log to TensorBoard
            if batch_idx % 10 == 0:
                active_cnn_name = "CNN_A" if self.model.A_is_active else "CNN_B"
                self.writer.add_scalar(f'Loss/Train_{active_cnn_name}', loss.item(),
                                       self.global_step)
                self.writer.add_scalar('KnowledgeTransmission/Network',
                                        0 if self.model.A_is_active else 1,
                                        self.global_step)
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, "
                           f"Active: {active_cnn_name}, Loss: {loss.item():.4f}")

            # Knowledge transmission: switch CNNs every K iterations
            self.model.iteration += 1
            if self.model.should_switch():
                old_active = "CNN_A" if self.model.A_is_active else "CNN_B"
                self.model.switch_active_cnn()
                new_active = "CNN_A" if self.model.A_is_active else "CNN_B"

                logger.info(f"[Knowledge Transmission] Iteration {self.model.iteration}: "
                           f"Switching from {old_active} to {new_active}")
                self.writer.add_scalar('KnowledgeTransmission/Switch',
                                       self.model.iteration, self.global_step)

            self.global_step += 1

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def validate(self, epoch):
        """Validate the model on validation set."""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches
        self.writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self, epochs):
        """Full training loop following Algorithm 1 from paper."""
        logger.info(f"Starting Co-DeepNet training for {epochs} epochs")
        logger.info(f"Knowledge transmission rate: {self.knowledge_transmission_rate} iterations")

        best_val_loss = float('inf')
        best_model = None

        for epoch in range(epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")

            # Training
            train_loss = self.train_epoch(epoch)
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)

            # Validation
            val_loss = self.validate(epoch)

            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")

            logger.info(f"Epoch {epoch + 1} Summary - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss if val_loss else 'N/A'}")

        self.writer.close()
        logger.info("Training completed")

        # Return best model
        if best_model is not None:
            self.model.load_state_dict(best_model)

        return self.model


def load_iris_data(batch_size=32, test_size=0.3, regression_target=False):
    """Load and prepare Iris dataset.

    Args:
        batch_size: Batch size for DataLoader
        test_size: Proportion of data for validation
        regression_target: If True, use sepal length as regression target.
                          If False, use species as classification target.

    Returns:
        train_loader, val_loader
    """
    logger.info("Loading Iris dataset")

    # Load Iris dataset
    iris = load_iris()
    X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width

    if regression_target:
        # Use sepal length as regression target (just for demo)
        y = X[:, 0]  # Sepal length
        X = X[:, 1:]  # Use other 3 features
        logger.info("Using Iris with regression target (sepal length)")
    else:
        # Use species as target (convert to regression for now)
        y = iris.target.astype(np.float32)
        logger.info("Using Iris with classification target (converted to regression)")

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                        random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Configuration for Iris dataset
    config = CNNConfig(
        input_features=3,  # Using 3 features (excluding target)
        conv1_out=3,
        conv2_out=9,
        fc1_out=120,
        fc2_out=84,
        output_dim=1,  # Single regression output
        kernel_size=3,
        stride=1,
        activation='relu'
    )

    # Load data
    train_loader, val_loader = load_iris_data(batch_size=16, regression_target=True)

    # Initialize Co-DeepNet model
    model = CoDeepNet(config)
    logger.info(f"Co-DeepNet initialized with config: {config}")

    # Initialize trainer
    trainer = CoDeepNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        log_dir="./tensorboard",
        knowledge_transmission_rate=20
    )

    # Train model
    trained_model = trainer.train(epochs=100)

    logger.info("Training completed successfully")
