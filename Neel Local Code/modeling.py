#Construct the model for audio classification

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
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchaudio.transforms as T
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

#Build the model for CNN and MLP

class MelSpec_CNN_Model(pl.LightningModule):
    def __init__():
        super().__init__()
    

class Feature_MLP_Model(pl.LightningModule):
    def __init__():
        super().__init__()

