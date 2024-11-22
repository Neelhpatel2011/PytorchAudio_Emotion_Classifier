
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

    # # Load training data

    # # mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training.npy')
    # # features_train = np.load('Data/' + 'training_features.npy')
    # # metadata_train = np.load('Data/' + 'training_metadata.npy')

    # mel_specs_combined = np.load('Data/mel_spectrograms_training_combined.npy')
    # features_combined = np.load('Data/training_features_combined.npy')
    # metadata_combined = np.load('Data/training_metadata_combined.npy')

    # # Load testing data
    # mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
    # features_test = np.load('Data/' + 'test_features.npy')
    # metadata_test = np.load('Data/' + 'test_metadata.npy')

    # train_waveforms_dict = {"Mel Spectrogram":mel_specs_combined,
    #                         "Features":features_combined}
    
    # test_waveforms_dict = {"Mel Spectrogram":mel_specs_test,
    #                         "Features":features_test}
    
    # train_metadata_df = pd.DataFrame(metadata_combined)
    # test_metdata_df = pd.DataFrame(metadata_test)


    # print(train_metadata_df['Emotion'].unique())

    # start = time.time()
    # # train_dataloader = load_dataset(train_metadata_df,
    # #                                 waveforms_preloaded = True,
    # #                                 waveforms_dict=train_waveforms_dict,
    # #                                 same_length_all=True,
    # #                                 sample_rate=SAMPLE_RATE,
    # #                                 seconds_of_audio=3)


    # # Create your custom dataset instance
    # full_train_dataset = Emotion_Classification_Waveforms(
    #     waveforms_dict=train_waveforms_dict,  # Your preloaded waveforms dictionary
    #     metadata_df=train_metadata_df,        # Your metadata DataFrame
    #     device=device                   # Device (e.g., 'cuda' or 'cpu')
    # )

    # test_dataset = Emotion_Classification_Waveforms(waveforms_dict=test_waveforms_dict,
    #                                                 metadata_df=test_metdata_df,
    #                                                 device = device)

    # # Split the dataset into training and validation sets
    # train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    # val_size = len(full_train_dataset) - train_size  # Remaining 20% for validation
    # train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # # Create DataLoaders for training and validation
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # test_dataloader = DataLoader(test_dataset, batch_size=16,shuffle = False)


    # # Iterate through DataLoader
    # for batch in tqdm(train_dataloader):
    #     waveform_features = batch['waveform_data']
    #     emotions = batch['emotion']
    #     genders = batch['gender']
    #     print(waveform_features['Mel Spectrogram'].shape,
    #           waveform_features['Features'].shape, 
    #           emotions, 
    #           genders)  
    #     #print(emotions,genders)
        

    # print(f"DataLoader initialization took: {time.time() - start:.2f} seconds")



    #Testing the model out!

    model = MelSpec_CNN_Model(input_channels=1)
    dummy_input = torch.zeros(4, 1, 64, 144)
    output = model(dummy_input)
    print("Output shape:", output.shape)






    

