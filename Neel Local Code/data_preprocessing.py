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

#Use GPU acceleration if possible
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
   
print(f'Using {device}') 

# Set seeds for reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # If using CUDA

def RAVDESS_extractor(audio_dir):
    data_list = []
    columns = ['Filename','Filepath','Gender','Emotion','Emotional Intensity']
    RAV_metadata_df = pd.DataFrame(columns = columns)
    
    # Map identifiers to their corresponding values
    emotion_dict = {
      "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
      "05": "angry", "06": "fear", "07": "disgust", "08": "surprised"
    }
    
    intensity_dict = {"01": "medium", "02": "high"}
    statement_dict = {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"}
    
    
    data_list = []
    for actor_folder in os.listdir(audio_dir):
      actor_path = os.path.join(audio_dir, actor_folder)
    
      if os.path.isdir(actor_path):  # Check if it's a folder
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    parts = file.split(".")[0].split("-") #first split the .wav extension then the '-'
    
                    # Extract metadata from the filename
                    modality = parts[0]  # Not used, as it’s audio-only for now
                    vocal_channel = "speech" if parts[1] == "01" else "song"
                    emotion = emotion_dict[parts[2]]
                    emotional_intensity = intensity_dict[parts[3]]
                    statement = statement_dict[parts[4]]
                    actor_id = int(parts[6])
                    gender = "male" if actor_id % 2 != 0 else "female"
                    file_path = os.path.join(actor_path, file)  # Full path to the file
                    
                    # Append to datalist (ignoring the repetition)
                    data_list.append({
                        'Filename': file,
                        'Filepath':file_path,
                        'Gender': gender,
                        'Emotion': emotion,
                        'Emotional Intensity': emotional_intensity
                    })
    
    df_addon = pd.DataFrame(data_list)
    RAV_metadata_df = pd.concat([RAV_metadata_df, df_addon], ignore_index=True)

    return RAV_metadata_df

def CREMA_extractor(audio_dir,crema_metadata_df):
    data_list = []
    emotion_map_dict = {'SAD':'sad',
                       'ANG':'angry',
                       'DIS':'disgust',
                       'FEA':'fear',
                       'HAP':'happy',
                       'NEU':'neutral'}
    intensity_dict = {'LO':'low',
                     'MD':'medium',
                     'HI':'high',
                     'XX':'unknown',
                     'X':'unknown'}

    columns = ['Filename','Filepath','Gender','Emotion','Emotional Intensity']
    crema_organized_df = pd.DataFrame(columns = columns)
    
    for file in os.listdir(audio_dir):
        parts = file.split('.')[0].split('_')

        file_name = file
        file_path = os.path.join(audio_dir,file)
        actor_id = int(parts[0])

        gender = crema_metadata_df.loc[crema_metadata_df['ActorID'] == actor_id]['Sex'].values[0].lower()
        emotion = emotion_map_dict[parts[2]]

        #debugging
        #print(file_name)
        intensity = intensity_dict[parts[3]]

        data_list.append({'Filename': file_name,
                         'Filepath':file_path,
                         'Gender':gender,
                         'Emotion':emotion,
                         'Emotional Intensity':intensity})

    df_addon = pd.DataFrame(data_list)
    crema_organized_df = pd.concat([crema_organized_df,df_addon],ignore_index=True)
    return crema_organized_df

def SAVEE_extractor(audio_dir):
    columns = ['Filename','Filepath','Gender','Emotion','Emotional Intensity']
    savee_metadata_df = pd.DataFrame(columns = columns)

    data_list = []
    
    emotion_map_dict = {'sa':'sad',
                       'a':'angry',
                       'd':'disgust',
                       'f':'fear',
                       'h':'happy',
                       'n':'neutral',
                        'su':'surprised'}

    for file in os.listdir(audio_dir):
        parts = file.split('.')[0].split('_')

        file_name = file
        file_path = os.path.join(audio_dir,file)
        gender = 'male'
        
        emotion_code = "".join([s for s in parts[1] if s.isalpha()])
        emotion = emotion_map_dict[emotion_code]
        intensity = 'unknown'

        data_list.append({'Filename': file_name,
                         'Filepath':file_path,
                         'Gender':gender,
                         'Emotion':emotion,
                         'Emotional Intensity':intensity})

    df_addon = pd.DataFrame(data_list)
    savee_metadata_df = pd.concat([savee_metadata_df,df_addon],ignore_index=True)
    return savee_metadata_df

def TESS_extractor(audio_dir):
    columns = ['Filename','Filepath','Gender','Emotion','Emotional Intensity']
    tess_metadata_df = pd.DataFrame(columns = columns)
    
    emotion_map_dict = {'sad':'sad',
                       'angry':'angry',
                       'disgust':'disgust',
                       'fear':'fear',
                       'happy':'happy',
                       'neutral':'neutral',
                       'ps':'surprised'}
    data_list = []
    
    for folder in os.listdir(audio_dir):
      folder_path = os.path.join(audio_dir, folder)
    
      if os.path.isdir(folder_path):  # Check if it's a folder
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    file_name = file
                    file_path = os.path.join(folder_path, file)
                    
                    parts = file.split('.')[0].split('_')
                    emotion = emotion_map_dict[parts[2].lower()]
                    intensity = 'unknown'
                    gender = 'female'
                    
                    data_list.append({'Filename': file_name,
                         'Filepath':file_path,
                         'Gender':gender,
                         'Emotion':emotion,
                         'Emotional Intensity':intensity})

    df_addon = pd.DataFrame(data_list)
    tess_metadata_df = pd.concat([tess_metadata_df,df_addon],ignore_index=True)
    return tess_metadata_df


#Import file_path

#Both sex
RAVDESS_path ='Data/speech-emotion-recognition-en/Ravdess/audio_speech_actors_01-24/'

#Both Sex
Crema_path = 'Data/speech-emotion-recognition-en/Crema/'
crema_metadata_df = pd.read_csv('Data/crema-metadata-extra-information/VideoDemographics.csv')

#Only male
SAVEE_path = 'Data/speech-emotion-recognition-en/Savee/' 

#Only female
TESS_path = 'Data/speech-emotion-recognition-en/Tess/'
    
ravdess_metadata_df = RAVDESS_extractor(RAVDESS_path)
crema_organized_df = CREMA_extractor(Crema_path,crema_metadata_df)
savee_metadata_df = SAVEE_extractor(SAVEE_path)
tess_metadata_df = TESS_extractor(TESS_path)

combined_metadata_df = pd.concat([ravdess_metadata_df,
                                  crema_organized_df,
                                  savee_metadata_df,
                                  tess_metadata_df])

combined_metadata_df.to_csv('Data/augmentations_NEW/combined_metadata_df.csv')