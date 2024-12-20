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



if __name__ == "__main__":
    #Get th augmented_surprised_samples!

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


    #Get the training features

    data_path = 'Data/metadata-and-augmentations/'
    augmented_training_df = pd.read_csv(data_path+'augmented_training_df.csv')

    #THESE ARE AUGMENTED BUT DO NOT HAVE SURPRISED!
    mel_specs_train, features_train, metadata_train = process_audio_files_train(augmented_training_df, data_path)

    metadata_train_array = np.array([
        (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
        for meta in metadata_train
    ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

    np.save(r'Data/mel_spectrograms_training_no_sur.npy', mel_specs_train)
    np.save(r'Data/training_features_no_sur.npy', features_train)
    np.save(r'Data/training_metadata_no_sur.npy', metadata_train_array)

    os.chdir(r'C:\Users\Neel Patel\Documents\Github Repositories\PytorchAudio_Emotion_Classifier')
    data_path = 'Data/metadata-and-augmentations/'
    #Get the validation features (unaugmented training 20%)
    val_training_df = pd.read_csv(data_path+'training_df.csv')

    #GET RID OF THE SURPRISED CATEGORY!
    val_training_df = val_training_df.loc[val_training_df['Emotion'] != 'surprised']

    #Get random 20% of this data with equal balance of emotions!
    #Function to sample 20% of data with equal balance of emotions
    def sample_balanced(df, fraction=0.2):
        # Group by Emotion
        grouped = df.groupby('Emotion')

        # Sample fraction of each group and concatenate
        sampled_df = grouped.apply(lambda x: x.sample(frac=fraction, random_state=SEED)).reset_index(drop=True)
        return sampled_df

    # Get the validation dataset
    val_training_df = sample_balanced(val_training_df, fraction=0.2)

    mel_specs_val, features_val, metadata_val = process_audio_files_test(val_training_df, data_path)

    metadata_val = np.array([
        (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
        for meta in metadata_val
    ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

    np.save(r'Data/mel_spectrograms_val_no_sur.npy', mel_specs_val)
    np.save(r'Data/val_features_no_sur.npy', features_val)
    np.save(r'Data/val_metadata_no_sur.npy', metadata_val)

    #Get the testing features

    testing_df = pd.read_csv(data_path+'testing_df.csv')
    mel_specs_test, features_test, metadata_test= process_audio_files_test(testing_df, data_path)

    metadata_test_array = np.array([
        (meta['Filename'], meta['Filepath'], meta['Gender'], meta['Emotion'])
        for meta in metadata_test
    ], dtype=[('Filename', 'U256'), ('Filepath', 'U256'), ('Gender', 'U10'), ('Emotion', 'U10')])

    np.save(r'Data/mel_spectrograms_test.npy', mel_specs_test)
    np.save(r'Data/test_features.npy', features_test)
    np.save(r'Data/test_metadata.npy', metadata_test_array)

    # #Load the data and test it out

    # mel_specs_surprised = np.load('Data/'+ 'mel_spectrograms_surprised.npy')
    # features_surprised = np.load('Data/' + 'surprised_features.npy')
    # metadata_surprised = np.load('Data/' + 'surprised_metadata.npy')

    # # Load training data
    # mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training.npy')
    # features_train = np.load('Data/' + 'training_features.npy')
    # metadata_train = np.load('Data/' + 'training_metadata.npy')

    #Combined surprised and existing training and save again!

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

    # #Load the training data

    # mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training_combined.npy')
    # features_train = np.load('Data/' + 'training_features_combined.npy')
    # metadata_train = np.load('Data/' + 'training_metadata_combined.npy',allow_pickle=True)
    #Load up test and train metadata.csv


    # # Load testing data
    # mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
    # features_test = np.load('Data/' + 'test_features.npy')
    # metadata_test = np.load('Data/' + 'test_metadata.npy',allow_pickle=True)

    #delete previous hdf5 files

    file_path = 'Data/training_data_no_sur.hdf5'
    if os.path.exists(file_path):
        os.remove(file_path)

    file_path = 'Data/validation_data_no_sur.hdf5'
    if os.path.exists(file_path):
        os.remove(file_path)

    file_path = 'Data/test_data_no_sur.hdf5'
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save training data
    with h5py.File('Data/training_data_no_sur.hdf5', 'w') as hdf:
        hdf.create_dataset('mel_spectrograms', data=mel_specs_train)
        hdf.create_dataset('features', data=features_train)

    with h5py.File('Data/validation_data_no_sur.hdf5', 'w') as hdf:
        hdf.create_dataset('mel_spectrograms', data=mel_specs_val)
        hdf.create_dataset('features', data=features_val)

    # Save test data
    with h5py.File('Data/test_data_no_sur.hdf5', 'w') as hdf:
        hdf.create_dataset('mel_spectrograms', data=mel_specs_test)
        hdf.create_dataset('features', data=features_test)


