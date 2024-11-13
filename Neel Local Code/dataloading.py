
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
from utilities import load_dataset



#Use GPU acceleration if possible
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
   
#print(f'Using {device}') 

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

    data_file_path = os.path.join(os.getcwd(), 'Data','metadata-and-augmentations')
    os.chdir(data_file_path)

    print(data_file_path)


    # Load data
    augmented_train_data_df = pd.read_csv(os.path.join(data_file_path, 'augmented_training_df.csv'))

    # Initialize DataLoader
    start = time.time()
    train_dataloader = load_dataset(augmented_train_data_df,
                                    same_length_all=True,
                                    sample_rate=SAMPLE_RATE,
                                    seconds_of_audio=3,
                                    transformations=transformations)
    print(f"DataLoader initialization took: {time.time() - start:.2f} seconds")

    # Iterate through DataLoader
    for batch in tqdm(train_dataloader):
        waveforms = batch['waveform_features']
        emotions = batch['emotion']
        genders = batch['gender']
        sample_rate = batch['sample rate']
        print(len(waveforms), emotions, genders, sample_rate)
        break


