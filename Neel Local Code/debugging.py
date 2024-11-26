#OLD DATASET CODE

class Emotion_Classification_Waveforms(Dataset):
    def __init__(self, waveforms_dict,
                 metadata_df):
        
        self.metadata_df = metadata_df
        self.waveforms_dict = waveforms_dict
        

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        emotion_mapping = {
            "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
            "fear": 4, "disgust": 5, "surprised": 6
        }

        gender_mapping = {"male": 0, "female": 1}

        emotion_label = self.metadata_df.iloc[idx]['Emotion']
        gender_label = self.metadata_df.iloc[idx]['Gender']
        file_name = self.metadata_df.iloc[idx]['Filename']

        # print(f'Emotion label is {emotion_label}')
        # # print(f'Gender label is {gender_label}')
        # print(f'Emotion mapping is {emotion_mapping[emotion_label]}')

        emotion_tensor = torch.tensor(emotion_mapping[emotion_label])
        gender_tensor = torch.tensor(gender_mapping[gender_label])

        waveform_data = {}
        for keys,values in self.waveforms_dict.items():
            if keys == 'Mel Spectrogram':
                mel_spec = self.waveforms_dict['Mel Spectrogram'][idx]
                mel_spec_db = mel_spec_to_db(mel_spec_array=mel_spec)
                waveform_data[keys] = mel_spec_db.unsqueeze(0)
            else:
                features = self.waveforms_dict[keys][idx]
                waveform_data[keys] = features
        
        # print(f'Emotion tensor is {emotion_tensor}')

        return {
            'waveform_data': waveform_data,
            'emotion': emotion_tensor,
            'gender': gender_tensor,
            'filename': file_name
        }
    
# #Train and test waveform dictionaries

# train_waveforms_dict = {"Mel Spectrogram":mel_specs_train,
#                         "Features":features_train}

# test_waveforms_dict = {"Mel Spectrogram":mel_specs_test,
#                             "Features":features_test}

# full_train_dataset = Emotion_Classification_Waveforms(
#         waveforms_dict=train_waveforms_dict,  # preloaded waveforms dictionary
#         metadata_df=train_metadata_df       
#     )

# test_dataset = Emotion_Classification_Waveforms(waveforms_dict=test_waveforms_dict,
#                                                     metadata_df=test_metadata_df)
