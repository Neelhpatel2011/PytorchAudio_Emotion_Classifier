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

from utilities import extract_hnr,extract_mel_spectrogram,extract_mfcc,extract_rms,extract_zero_crossing_rate
from utilities import load_dataset

import h5py

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using CUDA

print(device)

def add_white_noise(waveform, 
                    noise_level=np.random.uniform(low =  0.0001,high = 0.001)):
    noise = torch.randn_like(waveform, device = device) * noise_level
    print(noise.device)
    return waveform + noise

def time_stretch(waveform, sample_rate, rate=None):
    if rate is None:
        rate = np.random.uniform(0.95, 1.05)
    waveform_np = waveform.squeeze().cpu().numpy()  # Convert to NumPy
    stretched = librosa.effects.time_stretch(waveform_np, rate=rate)
    return torch.tensor(stretched, device = device).unsqueeze(0)  # Convert back to tensor with channel dimension

def pitch_scale(waveform, sample_rate, 
                n_steps=None):
    if n_steps is None:
        n_steps = np.random.uniform(low = -1, high = 1)
    waveform_np = waveform.squeeze().cpu().numpy()  # Convert to NumPy
    pitched = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)
    return torch.tensor(pitched, device = device).unsqueeze(0)  # Convert back to tensor with channel dimension

def polarity_inversion(waveform):
    return -waveform

def apply_gain(waveform, gain_factor = np.random.uniform(low = 0, high = 5)):
    gain = torchaudio.transforms.Vol(gain = gain_factor, gain_type = 'amplitude').to(device)
    return gain(waveform)

def apply_augmentations(waveform, sample_rate):
    # List of possible augmentations
    augmentations = [
        lambda x: add_white_noise(x),
        lambda x: time_stretch(x, sample_rate),
        lambda x: pitch_scale(x, sample_rate),
        lambda x: polarity_inversion(x),
        lambda x: apply_gain(x)
    ]
    
    # Randomly choose one or more augmentations to apply
    num_augmentations = 3
    selected_augmentations = random.sample(augmentations, num_augmentations)
    
    for augment in selected_augmentations:
        waveform = augment(waveform)
    return waveform

#Change the directory to the data directory
os.chdir('Data/')
combined_metadata_df = pd.read_csv('augmentations_NEW/combined_metadata_df.csv')

#Get rid of surprised category
combined_metadata_df = combined_metadata_df.loc[combined_metadata_df['Emotion']!='surprised']
augmented_dir = 'augmentations_NEW'

def create_training_augmentations(metadata_df, new_sample_rate):
    augmented_training_samples = []
    # First round of augmentation training samples (no surprised) (AugmentedV1)
    
    for i, row in tqdm(metadata_df.iterrows(),total = len(metadata_df),desc=f"Augmenting Version 1"):
        file_path = row['Filepath']
        file_name = row['Filename'].split('.')[0]

        augmented_file_name = f"{file_name}_augmented_training_V1_{i}.wav"
        gender = row['Gender']
        emotional_intensity = row['Emotional Intensity']
        emotion = row['Emotion']


        waveform, original_sr = torchaudio.load(file_path)
        waveform = waveform.to(device)
        #print(file_name)

        # Ensure waveform is 2D [1, num_samples] if it's mono
        if waveform.ndim > 1:
            waveform = waveform.mean(dim = 0).unsqueeze(0)
            
         # Resample if necessary
        if original_sr != new_sample_rate and new_sample_rate is not None:
            resampler = torchaudio.transforms.Resample(original_sr, 
                                                       new_sample_rate,
                                                       lowpass_filter_width=16, 
                                                       resampling_method='sinc_interp_hann').to(device)
            waveform = resampler(waveform)
            
        # Apply augmentations to create the first round of augmented samples
        augmented_waveform = apply_augmentations(waveform, original_sr)
        augmented_waveform = augmented_waveform.cpu()

        # Save augmented sample to a new file
        augmented_file_path = os.path.join(augmented_dir, augmented_file_name)
        torchaudio.save(augmented_file_path, augmented_waveform, original_sr, channels_first = True)

        # Append augmented sample details to the list
        augmented_training_samples.append({
            'Filename': augmented_file_name,
            'Filepath': augmented_file_path,
            'Gender': gender,
            'Emotion': emotion,  # Keep the same emotion label
            'Emotional Intensity': emotional_intensity,  
            'Augmentation_Type': f'AugmentedV'  # Tag the augmentation type
        })

    # Convert to DataFrame for compatibility
    augmented_training_df = pd.DataFrame(augmented_training_samples)
    return augmented_training_df

#augmented_training_df = create_training_augmentations(combined_metadata_df,new_sample_rate = 24414) 

#Test augmentations for 2 files

import torchaudio
# Load the .wav file 
waveform1, sr = torchaudio.load(r"C:\Users\Neel Patel\Downloads\03-01-07-02-01-02-02_resampled.wav")
waveform1 = waveform1.to(device)
# Verify the waveform and sampling rate
print(f"Waveform shape: {waveform1.shape}")
print(f"Sampling rate: {sr}")

# Apply augmentations
augmented_waveform = apply_augmentations(waveform1, sample_rate=sr)


# Save the augmented waveform to a new .wav file
torchaudio.save('augmentations_NEW/testaugmentations.wav', augmented_waveform.cpu(), sr, channels_first=True)

print("Augmented audio saved to testaugmentations.wav")