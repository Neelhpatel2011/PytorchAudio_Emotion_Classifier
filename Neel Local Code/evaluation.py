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
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from utilities import Emotion_Classification_Waveforms
from modeling import MelSpec_CNN_Model,Feature_MLP_Model,CombinedModel
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

###########################################################################

#Use GPU acceleration if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using CUDA
SAMPLE_RATE = 24414

# Load testing data
metadata_test = np.load('Data/' + 'test_metadata_no_sur_no_aug.npy', allow_pickle=True)

test_hdf5_file = 'Data/test_data_no_sur_no_aug_normalized.hdf5'
test_metadata_df = pd.DataFrame(metadata_test)

#take out surprised category for now (TEMPORARY!)
test_metadata_df = test_metadata_df.loc[test_metadata_df['Emotion'] != 'surprised']

test_dataset = Emotion_Classification_Waveforms(
    hdf5_file_path=test_hdf5_file,
    metadata_df=test_metadata_df
)

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle = True)

#Create a new model instance with the same architecture
cnn_model = MelSpec_CNN_Model()
mlp_model = Feature_MLP_Model()
model = CombinedModel(cnn=cnn_model, mlp=mlp_model)

# Load the final model from checkpoint
model = CombinedModel.load_from_checkpoint("final_model.ckpt",cnn=cnn_model,mlp=mlp_model)
model.eval() 
model.to(device)

# #Load the saved weights (Optionally!)
# model.load_state_dict(torch.load("final_model_weights.pth"))
# model.eval() 
# model.to(device)

num_classes = 6

# Metrics initialization
accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
precision_metric = Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
recall_metric = Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)
f1_metric = F1Score(task="multiclass", average='macro', num_classes=num_classes).to(device)
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

# Initialize accumulators for loss and metrics
criterion = torch.nn.CrossEntropyLoss()  # Define loss function
total_loss = 0
total_batches = 0

# Perform testing
with torch.inference_mode():  # Disable gradient computation
    for batch in test_dataloader:
        # Get the batch data
        mel_spec = batch['waveform_data']['Mel Spectrogram'].to(device)  # Move to device
        features = batch['waveform_data']['Features'].to(device)
        labels = batch['emotion'].to(device)

        # Forward pass through the model
        outputs = model(mel_spec, features)

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs, dim=1)

        # Update metrics
        accuracy_metric.update(preds, labels)
        precision_metric.update(preds, labels)
        recall_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        confusion_matrix.update(preds, labels)
        conf_matrix = confusion_matrix.compute()

        total_batches += 1

# Compute final metrics
average_loss = total_loss / total_batches
accuracy = accuracy_metric.compute()
precision = precision_metric.compute()
recall = recall_metric.compute()
f1 = f1_metric.compute()

# Print metrics
print(f"Test Loss: {average_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print(test_metadata_df['Emotion'].unique())

#######################VISUALIZATIONS!#########################




# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Reset metrics
accuracy_metric.reset()
precision_metric.reset()
recall_metric.reset()
f1_metric.reset()
confusion_matrix.reset()