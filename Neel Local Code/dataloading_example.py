
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

import pickle
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

if __name__ == "__main__":

    #Load the training data

    mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training_combined.npy')
    features_train = np.load('Data/' + 'training_features_combined.npy')
    metadata_train = np.load('Data/' + 'training_metadata_combined.npy',allow_pickle=True)

    # Load testing data
    mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
    features_test = np.load('Data/' + 'test_features.npy')
    metadata_test = np.load('Data/' + 'test_metadata.npy',allow_pickle=True)

    #Train and test waveform dictionaries

    train_waveforms_dict = {"Mel Spectrogram":mel_specs_train,
                            "Features":features_train}

    test_waveforms_dict = {"Mel Spectrogram":mel_specs_test,
                                "Features":features_test}

    #Train and test metadata dataframes
    train_metadata_df = pd.DataFrame(metadata_train.tolist())
    test_metadata_df = pd.DataFrame(metadata_test)

    # full_train_dataset = Emotion_Classification_Waveforms(
    #         waveforms_dict=train_waveforms_dict,  # preloaded waveforms dictionary
    #         metadata_df=train_metadata_df,        # metadata DataFrame
    #         device=device            
    #     )

    # test_dataset = Emotion_Classification_Waveforms(waveforms_dict=test_waveforms_dict,
    #                                                     metadata_df=test_metdata_df,
    #                                                     device = device)


    # # Split the training dataset into training and validation sets
    # train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    # val_size = len(full_train_dataset) - train_size  # Remaining 20% for validation
    # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # # Create DataLoaders for training and validation and testing
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4,persistent_workers=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle = False)
  
    print(train_metadata_df['Emotion'].value_counts())
    print(test_metadata_df['Emotion'].value_counts())

    # for batch in train_dataloader:
    #     print("Waveform Data Type:", type(batch['waveform_data']['Mel Spectrogram']), "Shape:", batch['waveform_data']['Mel Spectrogram'].shape)
    #     print("Emotion Data Type:", type(batch['emotion']))
    #     print("Gender Data Type:", type(batch['gender']))

    #     print(len(train_dataloader))
    #     break  # Stop after the first batch to avoid excessive output
    
    # # print("Dataset length:", len(full_train_dataset))
    # # print("Sample item:", full_train_dataset[0])

    # for i in range(len(train_dataset)):
    #     sample = train_dataset[i]
    #     try:
    #         pickle.dumps(sample)
    #     except Exception as e:
    #         print(f"Sample {i} is not picklable: {e}")


    




    

