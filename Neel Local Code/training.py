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
import torch.multiprocessing as mp

import h5py

###########################################################################

#Use GPU acceleration if possible
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using CUDA
SAMPLE_RATE = 24414

#Load the training data

mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training_combined.npy')
features_train = np.load('Data/' + 'training_features_combined.npy')
metadata_train = np.load('Data/' + 'training_metadata_combined.npy',allow_pickle=True)
#Load up test and train metadata.csv


# Load testing data
mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
features_test = np.load('Data/' + 'test_features.npy')
metadata_test = np.load('Data/' + 'test_metadata.npy',allow_pickle=True)

#Load HDF5 format files 
train_hdf5_file = 'Data/training_data.hdf5'
test_hdf5_file = 'Data/test_data.hdf5'

#Train and test metadata dataframes
train_metadata_df = pd.DataFrame(metadata_train.tolist())
test_metadata_df = pd.DataFrame(metadata_test.tolist())

# Create datasets
full_train_dataset = Emotion_Classification_Waveforms(
    hdf5_file_path=train_hdf5_file,
    metadata_df=train_metadata_df
)

test_dataset = Emotion_Classification_Waveforms(
    hdf5_file_path=test_hdf5_file,
    metadata_df=test_metadata_df
)

# #Train and test waveform dictionaries

# train_waveforms_dict = {"Mel Spectrogram":mel_specs_train,
#                         "Features":features_train}

# test_waveforms_dict = {"Mel Spectrogram":mel_specs_test,
#                             "Features":features_test}

# full_train_dataset = Emotion_Classification_Waveforms(
#         waveforms_dict=train_waveforms_dict,  # preloaded waveforms dictionary
#         metadata_df=train_metadata_df       
#     )

# test_dataset = Emotion_Classification_Waveforms(waveforms_dict=test_waveforms_dict,
#                                                     metadata_df=test_metadata_df)


# Split the training dataset into training and validation sets
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
val_size = len(full_train_dataset) - train_size  # Remaining 20% for validation
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders for training and validation and testing
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4,persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4,persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle = False)

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)

    # Instantiate the models
    cnn_model = MelSpec_CNN_Model()
    mlp_model = Feature_MLP_Model()

    # Instantiate the combined model
    model = CombinedModel(cnn=cnn_model, mlp=mlp_model)

    # Use PyTorch Lightning's Trainer to fit the model
    #Early stopping if val loss doesn't change
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=10,          # Number of epochs to wait without improvement
        verbose=True,
        mode='min'           # Minimize 'val_loss'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="emotion-classification-{epoch:02d}-{val_loss:.2f}",  # File naming format
        save_top_k=3,  # Save the top 3 models
        mode="min",  # Minimize the monitored metric (e.g., val_loss)
        save_last=True  # Always save the latest checkpoint
    )

    logger = TensorBoardLogger("logs/", name="emotion_classification")

    trainer = Trainer(max_epochs=50, 
                    callbacks=[early_stopping,checkpoint_callback],
                    logger=logger,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1 if torch.cuda.is_available() else None  # Use 1 GPU or CPU
    )


    trainer.fit(model, train_dataloader, val_dataloader)

    # Save the final trained model (complete Lightning module)
    trainer.save_checkpoint("final_model.ckpt")
    print("Model saved to final_model.ckpt")

    # Optionally, save only the model weights
    torch.save(model.state_dict(), "final_model_weights.pth")
    print("Model weights saved to final_model_weights.pth")