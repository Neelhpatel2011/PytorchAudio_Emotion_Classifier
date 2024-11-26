

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
from torchaudio.functional import amplitude_to_DB

import h5py

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

#Helper functions:
def get_max_audio_length(metadata_df):
    """Calculate the maximum length of audio samples in the dataset."""
    start = time.time()
    max_length = 0
    for file_path in metadata_df['Filepath']:
        waveform, sample_rate = torchaudio.load(file_path)
        num_samples = waveform.shape[1]  # Number of samples in the waveform
        if num_samples > max_length:
            max_length = num_samples
    end = time.time()
    print (f'Max_length_found: {max_length}. Took {start - end} seconds')
    return max_length

def same_length_batch(batch):
    # Collate function to handle variable-length sequences

    # Extract waveforms, emotions, and genders from the batch
    waveforms = [item['waveform'].squeeze(0) for item in batch]  # Remove channel dimension if present (it's all mono anyways!)
    emotions = torch.stack([item['emotion'] for item in batch]) #example [0,1,2]
    genders = torch.stack([item['gender'] for item in batch]) #same as above
    intensity = torch.stack([item['intensity'] for item in batch])
    sample_rate = torch.stack([item['sample rate'] for item in batch])

    
    # Pad all waveforms to the same length (of the longest in the batch) ****THIS MEANS WE WILL HAVE TO MAKE THE NN VARIABLE LENGTH DEPENDING ON THE BATCH!
    waveforms_padded = pad_sequence(waveforms, batch_first=True) #put the batch dimension first!
    
    # Return padded waveforms and corresponding labels
    return {'waveform': waveforms_padded, 
            'emotion': emotions, 
            'gender': genders,
            'intensity':intensity,
            'sample rate':sample_rate}

#Create the custom pytorch dataset
class Emotion_Classification_Dataset(Dataset):
    def __init__(self, metadata_df, 
                 transformations=None, 
                 same_length_all = True, 
                 target_length = None,
                 target_sr = None,
                 augment_surprised=False,
                 device = device):
        """
        Args:
            metadata_df (DataFrame): DataFrame containing file paths and labels.
            target_length (int): Target length for all audio samples in number of samples.
            transformations (callable, optional): Optional list of transformations to be applied on a sample (e.g., audio augmentation).
            same_length_all (Boolean): If True, enforce same length for all audio samples.
        """
        self.metadata_df = metadata_df
        self.transformations = transformations
        self.same_length_all = same_length_all #Boolean
        self.target_length = target_length
        self.augment_surprised = augment_surprised  # Boolean for augmenting "surprised" category
        self.target_sr = target_sr
        
    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):

        start_time = time.time()

    
        # Get file path and labels from the DataFrame
        file_path = self.metadata_df.iloc[idx]['Filepath']
        emotion_label = self.metadata_df.iloc[idx]['Emotion']
        gender_label = self.metadata_df.iloc[idx]['Gender']
        intensity = self.metadata_df.iloc[idx]['Emotional Intensity']

        # step1_time = time.time()
        # print(f"Step 1 (Fetching metadata): {step1_time - start_time:.4f} seconds")
        
        # Load audio file (torchaudio returns waveform and sample rate)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # step2_time = time.time()
        # print(f"Step 2 (Loading audio): {step2_time - step1_time:.4f} seconds")

        
        waveform = waveform.to(device)
        
        #Resample the data
        #waveform = self.resample_if_necessary(waveform, sample_rate) 
        
        # step3_time = time.time()
        # print(f"Step 3 (Resampling): {step3_time - step2_time:.4f} seconds")


        #Pad the data to make the same length
        if self.same_length_all and self.target_length is not None:
            waveform = self.pad_or_trim_waveform(waveform, self.target_length)
            
        # step4_time = time.time()
        # print(f"Step 4 (Padding/Trimming): {step4_time - step3_time:.4f} seconds")

        #waveform.cpu()
        waveform_feature_dict = {'original waveform':waveform}
        
        # # Extract features based on transformations
        # if self.transformations:
        #     #waveform = waveform.to(device)
        #     for transformation in self.transformations:
        #         waveform_feature = transformation(waveform)
        #         waveform_feature_dict[transformation.__class__.__name__] = waveform_feature#.cpu()
        
        if self.transformations:
            for feature_name, transformation in self.transformations.items():
                feature = transformation(waveform)
                # Convert feature to tensor if needed and move to the appropriate device
                if isinstance(feature, np.ndarray):
                    feature = torch.tensor(feature, device=device)

                waveform_feature_dict[feature_name] = feature



        # step5_time = time.time()
        # print(f"Step 5 (Transformations): {step5_time - step4_time:.4f} seconds")

        # Convert labels to tensors or numerical values
        emotion_mapping = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "fear": 4, "disgust": 5, "surprised": 6
        }
        gender_mapping = {"male": 0, "female": 1}
        

        # step6_time = time.time()
        # print(f"Step 6 (Label conversion): {step6_time - step5_time:.4f} seconds")


        emotion_tensor = torch.tensor(emotion_mapping[emotion_label])
        gender_tensor = torch.tensor(gender_mapping[gender_label])

        #intensity_tensor = torch.tensor(intensity_mapping[intensity])

        # You can return the labels as part of a dictionary
        sample = {'waveform_features': waveform_feature_dict, 
                  'emotion': emotion_tensor, 
                  'gender': gender_tensor,
                  'sample rate':torch.tensor(sample_rate)}

        # total_time = time.time()
        # print(f"Total time for __getitem__: {total_time - start_time:.4f} seconds")
        
        return sample
    
    def pad_or_trim_waveform(self, waveform, target_length):
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
    
class Emotion_Classification_Waveforms(Dataset):
    def __init__(self, hdf5_file_path,
                 metadata_df):
        
        self.metadata_df = metadata_df
        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = None  # Will be opened in __getitem__

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):

        if self.hdf5_file is None:
            # Each worker process must open its own file handle
            self.hdf5_file = h5py.File(self.hdf5_file_path, 'r')

        emotion_mapping = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "fear": 4, "disgust": 5, "surprised": 6
        }

        gender_mapping = {"male": 0, "female": 1}

        emotion_label = self.metadata_df.iloc[idx]['Emotion']
        gender_label = self.metadata_df.iloc[idx]['Gender']
        file_name = self.metadata_df.iloc[idx]['Filename']
        emotion_tensor = torch.tensor(emotion_mapping[emotion_label])
        gender_tensor = torch.tensor(gender_mapping[gender_label])

       # Access data from the HDF5 file
        mel_spec = self.hdf5_file['mel_spectrograms'][idx]
        features = self.hdf5_file['features'][idx]

        # Process mel_spec
        mel_spec_db = mel_spec_to_db(mel_spec_array=mel_spec)
        mel_spec_db = torch.tensor(mel_spec_db).unsqueeze(0).float()
        features = torch.tensor(features).float()

        waveform_data = {
            'Mel Spectrogram': mel_spec_db,
            'Features': features
        }

        return {
            'waveform_data': waveform_data,
            'emotion': emotion_tensor,
            'gender': gender_tensor,
            'filename': file_name
        }
    
    def __getstate__(self):
        # Ensure the HDF5 file handle is not pickled
        state = self.__dict__.copy()
        state['hdf5_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.hdf5_file = None  # Reinitialize in the worker process


def load_dataset(metadata_df, 
                 waveforms_preloaded = True,
                 waveforms_dict = None,
                 same_length_all = True,
                 sample_rate = 24414,
                 seconds_of_audio = 3,
                transformations = None,
                device = device):
    
    if waveforms_preloaded:
        waveforms_dataset = Emotion_Classification_Waveforms(metadata_df=metadata_df,
                                                             waveforms_dict=waveforms_dict,
                                                             device = device)
        
        dataloader = DataLoader(waveforms_dataset,
                                batch_size=16,
                                shuffle=True,
                                num_workers=0)
    else:
        if same_length_all:
            max_length = sample_rate * seconds_of_audio

            combined_dataset = Emotion_Classification_Dataset(metadata_df=metadata_df,
                                                    transformations = transformations,
                                                    same_length_all = True,
                                                    target_length = max_length,
                                                    target_sr = 24414)
            dataloader = DataLoader(combined_dataset, 
                                    batch_size=16, 
                                    shuffle=True,  
                                    num_workers = 1,
                                    persistent_workers=True)
        else:
            combined_dataset = Emotion_Classification_Dataset(metadata_df=metadata_df,
                                                    transformations = transformations,
                                                    same_length_all = False,
                                                    target_sr = 24414)
            dataloader = DataLoader(combined_dataset, 
                                    batch_size=16, 
                                    shuffle=True, 
                                    collate_fn=same_length_batch, 
                                    num_workers = 1,
                                    persistent_workers=True)
    return dataloader

# Function to compute Mel Spectrogram without any data manipulation (for CNN use)
def extract_mel_spectrogram(waveform, sample_rate = 24414, n_fft = 1024, hop_length = 128, n_mels = 128):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    ).to(waveform.device)  # Move transform to device (CPU or GPU)
    mel_spec = mel_transform(waveform).float()
    return mel_spec

def mel_spec_to_db(mel_spec_array, multiplier=10.0, amin=1e-6, db_multiplier=0.0):
    specgram = torch.tensor(mel_spec_array, dtype=torch.float32).squeeze(0)
    spec_db = amplitude_to_DB(specgram,multiplier,amin,db_multiplier)
    return spec_db

# Function to compute MFCC features
def extract_mfcc(waveform, sample_rate = 24414, n_mfcc=13, melkwargs = {"n_fft": 1024, "hop_length": 512, "n_mels": 64}):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_ratAe=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs).to(waveform.device)  # Move transform to device (CPU or GPU)
    mfccs = mfcc_transform(waveform).float()
    mfccs = torch.mean(mfccs, dim = 2)
    return mfccs

# Function to compute Zero-Crossing Rate (ZCR)
def extract_zero_crossing_rate(waveform):
    # zcr = ((waveform[:, 1:] * waveform[:, :-1] < 0).sum(dim=1).float())/waveform.shape[1]
    # return zcr
    waveform = waveform.detach().cpu().numpy()
    
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)
    zcr = zcr.flatten()
    return torch.tensor(zcr,dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

# Function to compute Harmonic-to-Noise Ratio (HNR) using torchaudio (approximation)
def extract_hnr(waveform):
    waveform = waveform.detach().cpu().numpy()

    harmonic = librosa.effects.harmonic(y=waveform)
    percussive = librosa.effects.percussive(y=waveform)
    hnr_mean = torch.tensor(np.mean(harmonic / (percussive + 1e-6)),dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu').unsqueeze(0)
    return hnr_mean

def extract_rms(waveform):
    waveform = waveform.detach().cpu().numpy()

    rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=512)
    rms = rms.flatten()
    rms_tensor = torch.tensor(rms, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    return rms_tensor
