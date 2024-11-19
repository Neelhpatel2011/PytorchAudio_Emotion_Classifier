
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
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

from utilities import extract_hnr,extract_mel_spectrogram,extract_mfcc,extract_rms,extract_zero_crossing_rate
from utilities import load_dataset, Emotion_Classification_Waveforms

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

transformations = {
    'Mel Spectrogram': extract_mel_spectrogram,
    'MFCC': extract_mfcc,
    'Zero Crossing Rate': extract_zero_crossing_rate,
    'HNR': extract_hnr,
    'RMS': extract_rms
}

if __name__ == "__main__":

    # Load training data

    mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training.npy')
    features_train = np.load('Data/' + 'training_features.npy')
    metadata_train = np.load('Data/' + 'training_metadata.npy')

    # Load testing data
    mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
    features_test = np.load('Data/' + 'test_features.npy')
    metadata_test = np.load('Data/' + 'test_metadata.npy')

    train_waveforms_dict = {"Mel Spectrogram":mel_specs_train,
                            "Features":features_train}
    
    train_metadata_df = pd.DataFrame(metadata_train)

    start = time.time()
    train_dataloader = load_dataset(train_metadata_df,
                                    waveforms_preloaded = True,
                                    waveforms_dict=train_waveforms_dict,
                                    same_length_all=True,
                                    sample_rate=SAMPLE_RATE,
                                    seconds_of_audio=3)

    # Iterate through DataLoader
    for batch in tqdm(train_dataloader):
        waveform_features = batch['waveform_data']
        emotions = batch['emotion']
        genders = batch['gender']
        # print(waveform_features['Mel Spectrogram'].shape,
        #       waveform_features['Features'].shape, 
        #       emotions, 
        #       genders)  
        print(emotions,genders)
        

    print(f"DataLoader initialization took: {time.time() - start:.2f} seconds")
