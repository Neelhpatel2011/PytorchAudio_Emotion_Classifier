import os
import h5py
import numpy as np

# Set the seed for reproducibility if not already set
SEED = 42
np.random.seed(SEED)

# Paths to your HDF5 files
train_hdf5_path = 'Data/training_data_no_sur.hdf5'
val_hdf5_path = 'Data/validation_data_no_sur.hdf5'
test_hdf5_path = 'Data/test_data_no_sur.hdf5'

# Paths to save the normalized HDF5 files
train_hdf5_normalized_path = 'Data/training_data_no_sur_normalized.hdf5'
val_hdf5_normalized_path = 'Data/validation_data_no_sur_normalized.hdf5'
test_hdf5_normalized_path = 'Data/test_data_no_sur_normalized.hdf5'

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
#feature_std_adj = np.where(feature_std == 0, epsilon, feature_std)
feature_std_adj = (feature_std + epsilon)

# Normalize the training data
normalized_mel_specs_train = (mel_specs_train - mel_spec_mean) / mel_spec_std
normalized_features_train = (features_train - feature_mean) / feature_std_adj

# Save the normalized training data to a new HDF5 file
with h5py.File(train_hdf5_normalized_path, 'w') as hdf:
    hdf.create_dataset('mel_spectrograms', data=normalized_mel_specs_train)
    hdf.create_dataset('features', data=normalized_features_train)

print("Normalized training data saved to:", train_hdf5_normalized_path)

# Save the computed mean and std for later use
np.save('Data/mel_spec_mean.npy', mel_spec_mean)
np.save('Data/mel_spec_std.npy', mel_spec_std)
np.save('Data/feature_mean.npy', feature_mean)
np.save('Data/feature_std.npy', feature_std_adj)

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
mel_spec_mean = np.load('Data/mel_spec_mean.npy')
mel_spec_std = np.load('Data/mel_spec_std.npy')
feature_mean = np.load('Data/feature_mean.npy')
feature_std_adj = np.load('Data/feature_std.npy')

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
