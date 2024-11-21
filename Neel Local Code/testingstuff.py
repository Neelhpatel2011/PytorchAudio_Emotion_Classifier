# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# import torchaudio
# import torch
# import pandas as pd

# from torchaudio.functional import amplitude_to_DB

# # Load training data
# mel_specs_train = np.load('Data/'+ 'mel_spectrograms_training.npy')
# features_train = np.load('Data/' + 'training_features.npy')
# metadata_train = np.load('Data/' + 'training_metadata.npy')

# # Load testing data
# mel_specs_test = np.load('Data/' + 'mel_spectrograms_test.npy')
# features_test = np.load('Data/' + 'test_features.npy')
# metadata_test = np.load('Data/' + 'test_metadata.npy')

# # Print sizes
# print("Training Mel Spectrograms Shape:", mel_specs_train.shape)
# print("Training Features Shape:", features_train.shape)
# print("Training Metadata Shape:", metadata_train.shape)

# print("Testing Mel Spectrograms Shape:", mel_specs_test.shape)
# print("Testing Features Shape:", features_test.shape)
# print("Testing Metadata Shape:", metadata_test.shape)


# # print(np.squeeze(mel_specs_train[15],0).shape)
# # #specgram = np.squeeze(mel_specs_train[0],0)
# # specgram = mel_specs_train[0]
# # specgram = torch.tensor(specgram, dtype=torch.float32).squeeze(0)

# # spec_db = amplitude_to_DB(specgram,10,1e-5,0)
# # librosa.display.specshow(spec_db.numpy())
# # #librosa.display.specshow(librosa.power_to_db(specgram),sr=24414, x_axis='time', y_axis='mel')

# # plt.show()

# #print(pd.DataFrame(metadata_train).columns)
# idx = 42

# waveforms_dict = {"Mel Spectrogram":mel_specs_train,
#                   "Waveform Features": features_train}

# waveform_data = {}

# for keys,values in waveforms_dict.items():
#     if keys == 'Mel Spectrogram':
#         mel_spec = waveforms_dict['Mel Spectrogram'][idx]
#         #fucntion to convert mel spec to mel_spec_db
#         waveform_data[keys] = mel_spec_db
#     else:
#         waveforms_dict[keys][idx]
