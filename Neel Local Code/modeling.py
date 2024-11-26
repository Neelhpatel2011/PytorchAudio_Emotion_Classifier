#Construct the model for audio classification

# Import packages!

import os
import shutil
from zipfile import ZipFile
import time
import random
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import librosa

from IPython.display import Audio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score

from sklearn.model_selection import train_test_split

#Build the model for CNN and MLP

# class MelSpec_CNN_Model(pl.LightningModule):
#     def __init__(self, input_channels=1):
#         super(MelSpec_CNN_Model, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.dropout = nn.Dropout(0.2)
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(512 * (64 // 32) * (144 // 32), 256)  # Adjust based on the output dimensions
#         self.fc1 = nn.Linear(256, 128)
#         self.bn_fc1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)

        
#     def forward(self, x):
#         # Pass through convolutional layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = self.dropout(x)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
#         x = self.dropout(x)

#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
#         x = self.dropout(x)

#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool(x)
#         x = self.dropout(x)

#         x = F.relu(self.bn5(self.conv5(x)))
#         x = self.pool(x)
#         x = self.dropout(x)

#         # Flatten the output
#         x = self.flatten(x)

#         # Dynamically set `self.fc` on the first forward pass
#         if isinstance(self.fc, nn.Identity):  # Check if it's still a placeholder
#             self.fc = nn.Linear(x.size(1), 256).to(x.device)

#         # Fully connected layers
#         x = F.relu(self.fc(x))
#         x = F.relu(self.bn_fc1(self.fc1(x)))
#         x = self.fc2(x)
#         return x


class MelSpec_CNN_Model(pl.LightningModule):
    def __init__(self, input_channels=1, dropout_rate=0.2):
        super(MelSpec_CNN_Model, self).__init__()
        self.dropout_rate = dropout_rate

        # Define 8 convolutional blocks
        self.blocks = nn.ModuleList()
        in_channels = input_channels
        out_channels_list = [64, 128, 256, 256, 512, 512, 512]  # Channels for each block

        for out_channels in out_channels_list:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 2)),  # Apply pooling at the end of every block
                    nn.Dropout(self.dropout_rate)
                )
            )
            in_channels = out_channels

        # Calculate output size after all blocks (assuming input size is [B, 1, 64, 144])
        self.flatten = nn.Flatten()
        dummy_input = torch.zeros(1, input_channels, 128, 573)  # Dummy input to calculate dimensions
        dummy_output = self._forward_blocks(dummy_input)
        flattened_size = dummy_output.size(1)

        # Fully connected layers
        self.fc = nn.Linear(flattened_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)

    def _forward_blocks(self, x):
        """Helper function to pass data through all CNN blocks."""
        for i,block in enumerate(self.blocks):
            x = block(x)
        x = self.flatten(x) #Flatten the output!
        return x

    def forward(self, x):
        x = self._forward_blocks(x)  # Pass through convolutional blocks
        x = F.relu(self.fc(x))  # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x) #output is 64
        return x

class Feature_MLP_Model(pl.LightningModule):
    def __init__(self, input_size=302):
        super(Feature_MLP_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4= nn.Linear(64,32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        return x

class CombinedModel(pl.LightningModule):
    def __init__(self, cnn, mlp, num_classes=7, learning_rate=1e-3, dropout_rate=0.2):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.mlp = mlp
        self.fc1 = nn.Linear(96, 64)  # Combining CNN (64) + MLP (32)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32,num_classes)

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Classification metrics (updated to include `task` argument)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.recall = Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", average='macro', num_classes=num_classes)

        # Save hyperparameters
        self.save_hyperparameters(ignore=["cnn", "mlp"])

    def forward(self, mel_spec, features):
        cnn_output = self.cnn(mel_spec)  # Process Mel spectrogram with CNN
        mlp_output = self.mlp(features)  # Process other features with MLP
        combined = torch.cat((cnn_output, mlp_output), dim=1)  # Concatenate outputs
        x = F.relu(self.bn1(self.fc1(combined)))
        x= F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        mel_spec = batch['waveform_data']['Mel Spectrogram']  # Shape [B, 64, 144]
        features = batch['waveform_data']['Features']  # Shape [B, 302]
        labels = batch['emotion']
        output = self(mel_spec, features)
        loss = F.cross_entropy(output, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mel_spec = batch['waveform_data']['Mel Spectrogram']
        features = batch['waveform_data']['Features']
        labels = batch['emotion']

        # Forward pass
        output = self(mel_spec, features)
        loss = F.cross_entropy(output, labels)

        # Compute predictions
        preds = torch.argmax(output, dim=1)

        # Log metrics
        self.accuracy.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.f1.update(preds, labels)

        # Log loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log metrics
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()

        self.log('val_accuracy', accuracy, prog_bar=True, logger=True)
        self.log('val_precision', precision, prog_bar=True, logger=True)
        self.log('val_recall', recall, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)

        # Reset metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        #Use a scheduler that reduces when it hits a plateau!
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
