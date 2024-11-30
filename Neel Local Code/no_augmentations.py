# Import packages!
#%%
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

def pad_or_trim_waveform(waveform, target_length):
        """Pad or trim waveform to a fixed target length in samples."""
        num_samples = waveform.shape[1]  # waveform shape is (channels, num_samples)

        if num_samples < target_length:
            # Pad if the waveform is shorter than target length
            padding = target_length - num_samples
            waveform = F.pad(waveform, (0, padding))#.to(device) #pad the left with 0 0s and pad the right with padding amount of 0s
        elif num_samples > target_length:
            # Trim if the waveform is longer than target length
            waveform = waveform[:, :target_length]

        return waveform

def extract_features(waveform):
    device = waveform.device
    mfcc_result = extract_mfcc(waveform).to(device)
    zcr_result = extract_zero_crossing_rate(waveform).to(device)
    hnr_result = extract_hnr(waveform).to(device)
    rms_result = extract_rms(waveform).to(device)
    result = torch.cat((mfcc_result.squeeze(0), zcr_result, hnr_result, rms_result), dim=0)
    return result

def process_audio_files(data_df, base_path):
    mel_spectrograms_list = []
    features_list = []
    metadata_list = []

    for index, file in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing audio files"):
        file_path = os.path.join(base_path, file['Filepath'])
        waveform, sample_rate = torchaudio.load(file_path)

        # Handle multi-channel audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        #First get the first 3 seconds of the audio
        waveform = pad_or_trim_waveform(waveform,target_length=24414*3)
        waveform = waveform.to(device)

        mel_spec = extract_mel_spectrogram(waveform).cpu().numpy()
        features = extract_features(waveform).cpu().numpy()  # Convert to numpy array

        # Collect features and metadata
        mel_spectrograms_list.append(mel_spec)
        features_list.append(features)
        metadata_list.append({
            'Filename': file['Filename'],
            'Filepath': file['Filepath'],
            'Gender': file['Gender'],
            'Emotion': file['Emotion']
        })
    return np.array(mel_spectrograms_list), np.array(features_list), metadata_list

#First get the train, val, and test indices
data_path = 'Data/'
unaugmented_df= pd.read_csv(data_path+'augmentations_NEW/combined_metadata_df.csv')

#Let's load the resampled audio directory and use the resampled audio files
resampled_dir = os.path.join(os.getcwd(),'Data','Audio_Resampled','Audio_Resampled')

unaugmented_df['Filepath'] = unaugmented_df['Filepath'].apply(
    lambda x: os.path.join(resampled_dir, os.path.basename(x))
)

#Get rid of surprised
unaugmented_df = unaugmented_df[unaugmented_df['Emotion'] != 'surprised']
# Reset index after filtering
unaugmented_df = unaugmented_df.reset_index(drop=True)

# Split the data into training, validation, and test sets with stratification
train_df, temp_df = train_test_split(
    unaugmented_df, test_size=0.3, random_state=SEED, stratify=unaugmented_df['Emotion']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['Emotion']
)

# Print the distribution of emotions in each set
print("Training set emotion distribution:")
print(train_df['Emotion'].value_counts())
print("\nValidation set emotion distribution:")
print(val_df['Emotion'].value_counts())
print("\nTest set emotion distribution:")
print(test_df['Emotion'].value_counts())

base_path = r'C:\Users\Neel Patel\Documents\Github Repositories\PytorchAudio_Emotion_Classifier'
# Process training data
mel_specs_train, features_train, metadata_train = process_audio_files(train_df, base_path)

metadata_train_array = np.array([
    (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
    for meta in metadata_train
], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

np.save('Data/mel_spectrograms_training_no_sur_no_aug.npy', mel_specs_train)
np.save('Data/training_features_no_sur_no_aug.npy', features_train)
np.save('Data/training_metadata_no_sur_no_aug.npy', metadata_train_array)

# Process validation data
mel_specs_val, features_val, metadata_val = process_audio_files(val_df, base_path)

metadata_val_array = np.array([
    (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
    for meta in metadata_val
], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

np.save('Data/mel_spectrograms_validation_no_sur_no_aug.npy', mel_specs_val)
np.save('Data/validation_features_no_sur_no_aug.npy', features_val)
np.save('Data/validation_metadata_no_sur_no_aug.npy', metadata_val_array)

# Process test data
mel_specs_test, features_test, metadata_test = process_audio_files(test_df, base_path)

metadata_test_array = np.array([
    (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
    for meta in metadata_test
], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

np.save('Data/mel_spectrograms_test_no_sur_no_aug.npy', mel_specs_test)
np.save('Data/test_features_no_sur_no_aug.npy', features_test)
np.save('Data/test_metadata_no_sur_no_aug.npy', metadata_test_array)

# Delete previous HDF5 files if they exist
train_hdf5_path = 'Data/training_data_no_sur_no_aug.hdf5'
val_hdf5_path = 'Data/validation_data_no_sur_no_aug.hdf5'
test_hdf5_path = 'Data/test_data_no_sur_no_aug.hdf5'

#delete previous hdf5 files
for file_path in [train_hdf5_path, val_hdf5_path, test_hdf5_path]:
    if os.path.exists(file_path):
        os.remove(file_path)

# Save training data to HDF5
with h5py.File(train_hdf5_path, 'w') as hdf:
    hdf.create_dataset('mel_spectrograms', data=mel_specs_train)
    hdf.create_dataset('features', data=features_train)

# Save validation data to HDF5
with h5py.File(val_hdf5_path, 'w') as hdf:
    hdf.create_dataset('mel_spectrograms', data=mel_specs_val)
    hdf.create_dataset('features', data=features_val)

# Save test data to HDF5
with h5py.File(test_hdf5_path, 'w') as hdf:
    hdf.create_dataset('mel_spectrograms', data=mel_specs_test)
    hdf.create_dataset('features', data=features_test)

print("Data processing complete. HDF5 files saved.")


########## NORMALIZATION ###################################

# Paths to your HDF5 files
train_hdf5_path = 'Data/training_data_no_sur_no_aug.hdf5'
val_hdf5_path = 'Data/validation_data_no_sur_no_aug.hdf5'
test_hdf5_path = 'Data/test_data_no_sur_no_aug.hdf5'

# Paths to save the normalized HDF5 files
train_hdf5_normalized_path = 'Data/training_data_no_sur_no_aug_normalized.hdf5'
val_hdf5_normalized_path = 'Data/validation_data_no_sur_no_aug_normalized.hdf5'
test_hdf5_normalized_path = 'Data/test_data_no_sur_no_aug_normalized.hdf5'

# Load the training data
with h5py.File(train_hdf5_path, 'r') as hdf:
    mel_specs_train = hdf['mel_spectrograms'][:]
    features_train = hdf['features'][:]

# Compute mean and std for Mel spectrograms across all training data
mel_spec_mean = np.mean(mel_specs_train)
mel_spec_std = np.std(mel_specs_train)

# Compute mean and std for each feature dimension in features
feature_mean = np.mean(features_train, axis=0)  # Shape: (302,)
feature_std = np.std(features_train, axis=0)    # Shape: (302,)

# To avoid division by zero, replace zeros in feature_std with a small epsilon
epsilon = 1e-8
feature_std_adj = np.where(feature_std == 0, epsilon, feature_std)

# Normalize the training data
normalized_mel_specs_train = (mel_specs_train - mel_spec_mean) / mel_spec_std
normalized_features_train = (features_train - feature_mean) / feature_std_adj

# Save the normalized training data to a new HDF5 file
with h5py.File(train_hdf5_normalized_path, 'w') as hdf:
    hdf.create_dataset('mel_spectrograms', data=normalized_mel_specs_train)
    hdf.create_dataset('features', data=normalized_features_train)

print("Normalized training data saved to:", train_hdf5_normalized_path)

# Save the computed mean and std for later use
np.save('Data/mel_spec_mean_no_aug.npy', mel_spec_mean)
np.save('Data/mel_spec_std_no_aug.npy', mel_spec_std)
np.save('Data/feature_mean_no_aug.npy', feature_mean)
np.save('Data/feature_std_no_aug.npy', feature_std_adj)

print("Normalization statistics saved to 'Data/' directory.")

# Function to normalize and save validation or test data
def normalize_and_save_data(input_hdf5_path, output_hdf5_path, mel_spec_mean, mel_spec_std, feature_mean, feature_std_adj):
    with h5py.File(input_hdf5_path, 'r') as hdf:
        mel_specs = hdf['mel_spectrograms'][:]
        features = hdf['features'][:]

    # Normalize the data
    normalized_mel_specs = (mel_specs - mel_spec_mean) / mel_spec_std
    normalized_features = (features - feature_mean) / feature_std_adj

    # Save the normalized data to a new HDF5 file
    with h5py.File(output_hdf5_path, 'w') as hdf:
        hdf.create_dataset('mel_spectrograms', data=normalized_mel_specs)
        hdf.create_dataset('features', data=normalized_features)

    print(f"Normalized data saved to: {output_hdf5_path}")

# Load the saved normalization statistics
mel_spec_mean = np.load('Data/mel_spec_mean_no_aug.npy')
mel_spec_std = np.load('Data/mel_spec_std_no_aug.npy')
feature_mean = np.load('Data/feature_mean_no_aug.npy')
feature_std_adj = np.load('Data/feature_std_no_aug.npy')

# Normalize and save the validation data
normalize_and_save_data(
    input_hdf5_path=val_hdf5_path,
    output_hdf5_path=val_hdf5_normalized_path,
    mel_spec_mean=mel_spec_mean,
    mel_spec_std=mel_spec_std,
    feature_mean=feature_mean,
    feature_std_adj=feature_std_adj
)

# Normalize and save the test data
normalize_and_save_data(
    input_hdf5_path=test_hdf5_path,
    output_hdf5_path=test_hdf5_normalized_path,
    mel_spec_mean=mel_spec_mean,
    mel_spec_std=mel_spec_std,
    feature_mean=feature_mean,
    feature_std_adj=feature_std_adj
)

print("Normalization complete.")
