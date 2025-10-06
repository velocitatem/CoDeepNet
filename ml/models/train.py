import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from alveslib import get_logger

logger = get_logger("ml-trainloop")
logger.info("initiated loop for training")

class Trainer:
    def __init__(self, model, train_loader, log_dir="../tensorboard"):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def train_epoch(self):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                self.writer.add_scalar('Loss/Train', loss.item(), self.step)
                self.step += 1

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
        self.writer.close()
