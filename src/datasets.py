import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from scipy.signal import resample, butter, filtfilt

def preprocess_eeg(eeg_data, new_sampling_rate=128, lowcut=0.5, highcut=40.0, fs=250.0):
    eeg_data = resample(eeg_data, int(new_sampling_rate * eeg_data.shape[1] / fs), axis=1)
    nyquist = 0.5 * new_sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    eeg_data = filtfilt(b, a, eeg_data, axis=1)
    eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)
    baseline = np.mean(eeg_data[:, :new_sampling_rate], axis=1, keepdims=True)
    eeg_data = eeg_data - baseline
    return eeg_data

def augment_eeg(eeg_data):
    noise = np.random.normal(0, 0.1, eeg_data.shape)
    eeg_data += noise
    shift = np.random.randint(-10, 10)
    eeg_data = np.roll(eeg_data, shift, axis=1)
    return eeg_data

class MEGDataset(Dataset):
    def __init__(self, data_dir, split='train', augment=False, debug=False):
        self.data_dir = os.path.abspath(data_dir)
        self.split = split
        self.augment = augment
        self.debug = debug

        data_dir_path = os.path.join(self.data_dir, f'{split}_X')
        subjects_dir_path = os.path.join(self.data_dir, f'{split}_subject_idxs')

        labels_dir_path = None if split == 'test' else os.path.join(self.data_dir, f'{split}_y')

        if self.debug:
            print(f"Checking directory existence:")
            print(f"Directory {data_dir_path} exists: {os.path.isdir(data_dir_path)}")
            if labels_dir_path:
                print(f"Directory {labels_dir_path} exists: {os.path.isdir(labels_dir_path)}")
            print(f"Directory {subjects_dir_path} exists: {os.path.isdir(subjects_dir_path)}")

            if os.path.isdir(data_dir_path):
                print(f"Files in {data_dir_path}: {os.listdir(data_dir_path)}")
            if labels_dir_path and os.path.isdir(labels_dir_path):
                print(f"Files in {labels_dir_path}: {os.listdir(labels_dir_path)}")
            if os.path.isdir(subjects_dir_path):
                print(f"Files in {subjects_dir_path}: {os.listdir(subjects_dir_path)}")

        self.data_paths = sorted(glob(os.path.join(data_dir_path, '*.npy')))
        self.labels_paths = sorted(glob(os.path.join(labels_dir_path, '*.npy'))) if labels_dir_path else []
        self.subject_paths = sorted(glob(os.path.join(subjects_dir_path, '*.npy')))

        if self.debug:
            print(f'Initializing dataset from {self.data_dir} with split {split}')
            print(f'Looking for data files in {data_dir_path}')
            if labels_dir_path:
                print(f'Looking for label files in {labels_dir_path}')
            print(f'Looking for subject files in {subjects_dir_path}')
            print(f'Found {len(self.data_paths)} data files, {len(self.labels_paths)} label files, and {len(self.subject_paths)} subject files.')

        if len(self.data_paths) == 0 or len(self.subject_paths) == 0 or (split != 'test' and len(self.labels_paths) == 0):
            raise ValueError(f"No data found for split '{split}' in directory '{self.data_dir}'")
        
        self.num_samples = len(self.data_paths)
        self.num_classes = 1854

        sample_data = np.load(self.data_paths[0])
        if self.debug:
            print(f"Sample data shape: {sample_data.shape}")
        self.num_channels = sample_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        subject_path = self.subject_paths[index]

        label_path = None if self.split == 'test' else self.labels_paths[index]

        if self.debug:
            print(f'Loading data from {data_path}')
            if label_path:
                print(f'Loading label from {label_path}')
            print(f'Loading subject info from {subject_path}')

        X = np.load(data_path)
        subject_idx = np.load(subject_path)

        X = preprocess_eeg(X)
        if self.augment:
            X = augment_eeg(X)

        X = torch.tensor(X, dtype=torch.float32)
        subject_idx = torch.tensor(subject_idx, dtype=torch.long)

        if label_path:
            y = np.load(label_path)
            y = torch.tensor(y, dtype=torch.long)
            return X, y, subject_idx
        else:
            return X, subject_idx

def get_dataloaders(data_dir, batch_size, num_workers, augment=False, debug=False):
    train_dataset = MEGDataset(data_dir, 'train', augment, debug)
    val_dataset = MEGDataset(data_dir, 'val', False, debug)
    test_dataset = MEGDataset(data_dir, 'test', False, debug)

    print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
