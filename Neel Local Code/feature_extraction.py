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
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using CUDA

###

# 1. Extract features offline

# 2. Mel Spectogram file of all audio files with emotion and gender and filepath (numpy file)
# 3. Other features will be extracted where each row is 1 audio file's features concatenated and then we store emotion gender and filepath

#Model inputs

#include dropout of regularization (dropout of 0.3)

#MLP - 13 + 144 + 1 + 144 = 302 neurons in input layer
# activation fxn will be relu()
# 3 hidden layers will be layer 1 = 128, layer 2 = 64 (gets fed into CNN dense layer) feed these into layer 3  = 32 
# final output layer will be with softmax of #number of emotions (7) (take argmax and find emotion predicted)

#CNN - 

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

def process_audio_files_train(data_df, base_path):
    mel_spectrograms_list = []
    features_list = []
    metadata_list = []

    for index, file in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing audio files"):
        file_path = os.path.join(base_path, file['Filepath'])
        waveform, sample_rate = torchaudio.load(file_path)

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

def process_audio_files_test(data_df, base_path):
    mel_spectrograms_list = []
    features_list = []
    metadata_list = []

    for index, file in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing audio files"):
        #file_path = os.path.join(base_path, file['Filepath'])

        filepath = file['Filepath']
        path_parts = filepath.split("/")
        start_index = path_parts.index('speech-emotion-recognition-en')  # Finding the starting index of your desired folder
        # Add 'Data' and join the remaining parts
        desired_path = os.path.join('Data', *path_parts[start_index:])

        waveform, sample_rate = torchaudio.load(desired_path)

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
            'Filepath': desired_path,
            'Gender': file['Gender'],
            'Emotion': file['Emotion']
        })
    return np.array(mel_spectrograms_list), np.array(features_list), metadata_list


# #Get the augmented_surprised_samples!

# data_path = 'Data/metadata-and-augmentations/'
# augmented_surprised_df = pd.read_csv(data_path+'augmented_surprised_df.csv')

# mel_specs_surprised, features_surprised, metadata_surprised= process_audio_files_train(augmented_surprised_df, data_path)

# metadata_surprised_array = np.array([
#     (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
#     for meta in metadata_surprised
# ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

# np.save(r'Data/mel_spectrograms_surprised.npy', mel_specs_surprised)
# np.save(r'Data/surprised_features.npy', features_surprised)
# np.save(r'Data/surprised_metadata.npy', metadata_surprised_array)


# #Get the training features
# data_path = 'Data/metadata-and-augmentations/'
# augmented_training_df = pd.read_csv(data_path+'augmented_training_df.csv')

# mel_specs_train, features_train, metadata_train = process_audio_files_train(augmented_training_df, data_path)

# metadata_train_array = np.array([
#     (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
#     for meta in metadata_train
# ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

# np.save(r'Data/mel_spectrograms_training.npy', mel_specs_train)
# np.save(r'Data/training_features.npy', features_train)
# np.save(r'Data/training_metadata.npy', metadata_train_array)



# #Get the testing features

# testing_df = pd.read_csv(data_path+'testing_df.csv')
# mel_specs_test, features_test, metadata_test= process_audio_files_test(testing_df, data_path)

# metadata_test_array = np.array([
#     (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
#     for meta in metadata_test
# ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

# np.save(r'Data/mel_spectrograms_test.npy', mel_specs_test)
# np.save(r'Data/test_features.npy', features_test)
# np.save(r'Data/test_metadata.npy', metadata_test_array)


# # #Load the data and test it out

# # mel_specs_surprised = np.load('Data/'+ 'mel_spectrograms_surprised.npy')
# # features_surprised = np.load('Data/' + 'surprised_features.npy')
# # metadata_surprised = np.load('Data/' + 'surprised_metadata.npy')

# # # Load training data
# # mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training.npy')
# # features_train = np.load('Data/' + 'training_features.npy')
# # metadata_train = np.load('Data/' + 'training_metadata.npy')

# #Combined surprised and existing training and save again!

# # Concatenate Mel spectrograms
# mel_specs_train = np.concatenate((mel_specs_train, mel_specs_surprised), axis=0)

# # Concatenate features
# features_train = np.concatenate((features_train, features_surprised), axis=0)

# # Concatenate metadata (structured array)
# metadata_train = np.concatenate((metadata_train, metadata_surprised))

# # Save the updated combined training data
# np.save('Data/mel_spectrograms_training_combined.npy', mel_specs_train)
# np.save('Data/training_features_combined.npy', features_train)
# np.save('Data/training_metadata_combined.npy', metadata_train)

# # # Load testing data
# # mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
# # features_test = np.load('Data/' + 'test_features.npy')
# # metadata_test = np.load('Data/' + 'test_metadata.npy')


# Load combined data to test
mel_specs_combined = np.load('Data/mel_spectrograms_training_combined.npy',allow_pickle=True)
features_combined = np.load('Data/training_features_combined.npy',allow_pickle=True)
metadata_combined = np.load('Data/training_metadata_combined.npy',allow_pickle=True)

# Print their shapes to verify
print("Shape of combined Mel spectrograms:", mel_specs_combined.shape)
print("Shape of combined features:", features_combined.shape)
print("Number of metadata entries:", len(metadata_combined))


# print("Testing Mel Spectrograms Shape:", mel_specs_test.shape)
# print("Testing Features Shape:", features_test.shape)
# print("Testing Metadata Shape:", metadata_test.shape)
