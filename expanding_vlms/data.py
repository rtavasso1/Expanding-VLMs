
# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import interpolate
from scipy.signal import butter, lfilter
import os
import random
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def downsample(data, length):
    # Your original timestamps (assuming it starts at 0 and ends at 1)
    original_timestamps = np.linspace(0, 1, data.shape[0])

    # Your new timestamps
    new_timestamps = np.linspace(0, 1, length) # interp is [data_length, shortest_length_file_length]

    # A placeholder for your downsampled data
    downsampled_data = np.zeros((length, 3))

    # Loop through each feature
    for i in range(3):
        # Create a cubic spline interpolation for the i-th feature
        filtered_data = butter_lowpass_filter(data[:,i], cutoff=25, fs=100, order=5)
        cs = interpolate.CubicSpline(original_timestamps, filtered_data)

        # Interpolate the downsampled data points for the i-th feature
        downsampled_data[:, i] = cs(new_timestamps)
    return downsampled_data

def load_random_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Find the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the middle quarters of the video
    lower_bound = total_frames // 4
    upper_bound = 3 * total_frames // 4
    
    # Randomly select a frame index within the middle quarters
    random_frame_idx = random.randint(lower_bound, upper_bound)
    
    # Navigate to the selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, (224, 224))
    else:
        cap.release()
        return None

    cap.release()
    return torch.tensor(frame, dtype=torch.float32)

def load_activity(activity_paths):
    dfs = []
    min_len = float('inf')
    to_downsample = []
    
    for path in activity_paths:
        if os.path.getsize(path) == 0:
            print(f"Skipping empty file: {path}")
            return None
        df = pd.read_csv(path, header=None)
        min_len = min(min_len, len(df))
        dfs.append(df.iloc[:, 1:])
        if 'acc_' in path: # downsample accelerometer data
            to_downsample.append(True)
        else:
            to_downsample.append(False)
    
    downsampled_arrays = [downsample(df.to_numpy(),min_len) if to_downsample[i] else df.to_numpy() for i, df in enumerate(dfs)]
    trimmed_arrays = [arr[:min_len] for arr in downsampled_arrays]
    combined_arrays = np.concatenate(trimmed_arrays, axis=1)
    tensor_data = torch.tensor(combined_arrays, dtype=torch.float32)
    
    return tensor_data

def load_all_data(root_dirs, video_root_dir):
    """
    Load precomputed embeddings, otherwise, go thru files and build dict
    of video filepaths and corresponding IMU data
    
    Args:
        root_dirs: directories containing IMU sensor signals
        video_root_dir: directory for video paths
    Returns:
        dict with key of scenario, value of IMU data and video embeddings/file paths
    """
    if os.path.exists('./saved_weights/dataset_perceiver_embeddings_fixed.pkl'):
        with open('./saved_weights/dataset_perceiver_embeddings_fixed.pkl', 'rb') as file:
            return pickle.load(file)
        
    if os.path.exists('./saved_weights/dataset_paths.pkl'):
        with open('./saved_weights/dataset_paths.pkl', 'rb') as file:
            return pickle.load(file)
    
    dataset = {}
    
    for subj in range(1, 21):
        for scene in range(1, 5):
            for sess in range(1, 6):
                
                init_path = os.path.join(root_dirs[0], f'subject{subj}', f'scene{scene}', f'session{sess}')
                if not os.path.exists(init_path):
                    print(f"Path does not exist: {init_path}")
                    continue
                init_files = set([f for f in os.listdir(init_path) if f.endswith('.csv')])
                
                for fname in init_files:
                    activity_paths = []
                    video_paths = []
                    
                    for root_dir in root_dirs:
                        path = os.path.join(root_dir, f'subject{subj}', f'scene{scene}', f'session{sess}')
                        full_path = os.path.join(path, fname)
                        
                        if os.path.exists(full_path):
                            activity_paths.append(full_path)
                        else:
                            print(f"File {fname} not found in {path}")
                            break
                    
                    for cam in range(1,5):
                        video_fname = os.path.splitext(fname)[0] + '.mp4'
                        video_path = os.path.join(video_root_dir, f'subject{subj}', f'cam{cam}', f'scene{scene}', f'session{sess}', video_fname)
                        
                        if os.path.exists(video_path):
                            video_paths.append(video_path)

                    if len(activity_paths) == len(root_dirs) and len(video_paths) >= 1: # video paths can have 0-4 elements
                        activity_name = os.path.splitext(fname)[0]
                        activity_data = load_activity(activity_paths)
                        
                        if activity_data is not None and video_paths is not None:
                            key = f'subject{subj}_scene{scene}_session{sess}_{activity_name}'
                            dataset[key] = (activity_data, video_paths)

    return dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
"""
Below are the definitions for 3 distinct datasets, collate fns, and dataloaders.
Each is designed to handle a different video input: video paths, video embeddings, 
and video perceiver embeddings. To save training costs, I precompute these static 
embeddings and save them to a file, loading them on the fly. The most efficient
training is done using the precomputed perceiver embeddings. 
"""

class Video_Path_Dataset(Dataset):
    def __init__(self, imu, video_paths, label):
        self.imu = imu
        self.video_paths = video_paths
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        video_path = random.choice(self.video_paths[idx])
        return self.imu[idx], video_path, self.label[idx]
    
class Video_Embeddings_Dataset(Dataset):
    def __init__(self, imu, video, label):
        self.imu = imu
        self.video = video
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # Randomly permute the indices of the rows
        permuted_indices = torch.randperm(self.video[idx].size(0))

        # Get the index of the first randomly permuted row
        random_row_index = permuted_indices[0]

        # Select the random row using the index
        random_cam = self.video[idx][random_row_index]
        
        return self.imu[idx], random_cam, self.label[idx]
    
class Video_Perceiver_Dataset(Dataset):
    def __init__(self, imu, video_class, video_perceiver, label):
        self.imu = imu
        self.video_class = video_class
        self.video_perceiver = video_perceiver
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # Randomly permute the indices of the rows
        permuted_indices = torch.randperm(self.video_class[idx].size(0))

        # Get the index of the first randomly permuted row
        random_row_index = permuted_indices[0]

        # Select the random row using the index
        random_video_class = self.video_class[idx][random_row_index]
        random_video_perceiver = self.video_perceiver[idx][random_row_index]
        
        return self.imu[idx], random_video_class, random_video_perceiver, self.label[idx]

def sample_imu_window(imu_data, length=256):
    sampled_imu_Xs = []
    for X in imu_data:
        T = X.shape[0]
        if T >= length:
            start_idx = random.randint(0, T - length)
            sampled_X = X[start_idx:start_idx + length, :]
        else:
            padding = torch.zeros(length - T, X.shape[1])
            sampled_X = torch.cat([X, padding], dim=0)
        sampled_imu_Xs.append(sampled_X)
    
def collate_fn(batch):
    imu_Xs, video_paths, ys = zip(*batch)
    sampled_video_Xs = []
    
    # Handle IMU data
    sampled_imu_Xs = sample_imu_window(imu_Xs, length=256)

    # Handle video data
    for video_path in video_paths:
        video_tensor = load_random_middle_frame(video_path)
        if video_tensor is None:
            print(f"Skipping empty video: {video_path}")
            video_tensor = torch.zeros((1, 224, 224, 3))
        sampled_video_Xs.append(video_tensor)
    
    return torch.stack(sampled_imu_Xs), torch.stack(sampled_video_Xs).squeeze(), torch.tensor(ys, dtype=torch.long)

def collate_fn_precomputed_embeddings(batch):
    imu_Xs, videos, ys = zip(*batch)
    
    # Handle IMU data
    sampled_imu_Xs = sample_imu_window(imu_Xs, length=256)
    
    return torch.stack(sampled_imu_Xs), torch.stack(videos).squeeze(), torch.tensor(ys, dtype=torch.long)

def collate_fn_precomputed_perceiver_embeddings(batch):
    imu_Xs, videos_class, videos_perceiver, ys = zip(*batch)
    
    # Handle IMU data
    sampled_imu_Xs = sample_imu_window(imu_Xs, length=256)
    
    return torch.stack(sampled_imu_Xs), torch.stack(videos_class).squeeze(), torch.stack(videos_perceiver).squeeze(), torch.tensor(ys, dtype=torch.long)
    
def prepare_dataloader(dataset, test_size=0.2, batch_size=128, shuffle=True, num_workers=4, worker_init_fn=seed_worker):
    imu_x_data = []
    video_x_data = []
    y_data = []
    label_mapping = {}
    label_count = 0

    for key, (imu_tensor_data, video_paths_data) in dataset.items():
        imu_x_data.append(imu_tensor_data)
        video_x_data.append(video_paths_data)
        
        activity_name = key.split('_')[-1]
        if activity_name not in label_mapping:
            label_mapping[activity_name] = label_count
            label_count += 1

        y_data.append(label_mapping[activity_name])

    # Split the data
    imu_train, imu_test, video_train, video_test, y_train, y_test = train_test_split(
        imu_x_data, video_x_data, y_data, test_size=test_size, random_state=42)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = Video_Path_Dataset(imu_train, video_train, y_train)
    test_dataset = Video_Path_Dataset(imu_test, video_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, worker_init_fn=seed_worker, drop_last=True, pin_memory=True, prefetch_factor=batch_size//num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=batch_size//num_workers)

    return train_loader, test_loader, label_mapping
    
def prepare_dataloader_precomputed_embeddings(dataset, test_size=0.2, batch_size=128, shuffle=True, num_workers=4, worker_init_fn=seed_worker):
    imu_x_data = []
    video_x_data = []
    y_data = []
    label_mapping = {}
    label_count = 0

    for key, (imu_tensor_data, video_data) in dataset.items():
        imu_x_data.append(imu_tensor_data)
        video_x_data.append(video_data)
        
        activity_name = key.split('_')[-1]
        if activity_name not in label_mapping:
            label_mapping[activity_name] = label_count
            label_count += 1

        y_data.append(label_mapping[activity_name])

    # Split the data
    imu_train, imu_test, video_train, video_test, y_train, y_test = train_test_split(
        imu_x_data, video_x_data, y_data, test_size=test_size, random_state=42)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = Video_Embeddings_Dataset(imu_train, video_train, y_train)
    test_dataset = Video_Embeddings_Dataset(imu_test, video_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_precomputed_embeddings, num_workers=0, worker_init_fn=seed_worker, drop_last=True, pin_memory=True, prefetch_factor=batch_size//num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_precomputed_embeddings, num_workers=0, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=batch_size//num_workers)

    return train_loader, test_loader, label_mapping

def prepare_dataloader_precomputed_perceiver_embeddings(dataset, test_size=0.2, batch_size=128, shuffle=True, num_workers=4, worker_init_fn=seed_worker):
    imu_x_data = []
    video_class_data = []
    video_perceiver_data = []
    y_data = []
    label_mapping = {}
    label_count = 0

    for key, (imu_tensor_data, video_class, video_perceiver) in dataset.items():
        imu_x_data.append(imu_tensor_data)
        video_class_data.append(video_class)
        video_perceiver_data.append(video_perceiver)
        
        activity_name = key.split('_')[-1]
        if activity_name not in label_mapping:
            label_mapping[activity_name] = label_count
            label_count += 1

        y_data.append(label_mapping[activity_name])

    # Split the data
    imu_train, imu_test, video_class_train, video_class_test, video_perceiver_train, video_perceiver_test, y_train, y_test = train_test_split(
        imu_x_data, video_class_data, video_perceiver_data, y_data, test_size=test_size, random_state=42)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = Video_Perceiver_Dataset(imu_train, video_class_train, video_perceiver_train, y_train)
    test_dataset = Video_Perceiver_Dataset(imu_test, video_class_test, video_perceiver_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_precomputed_perceiver_embeddings, num_workers=num_workers, worker_init_fn=seed_worker, drop_last=True, pin_memory=True, prefetch_factor=batch_size//num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_precomputed_perceiver_embeddings, num_workers=num_workers, worker_init_fn=seed_worker, pin_memory=True, prefetch_factor=batch_size//num_workers)

    return train_loader, test_loader, label_mapping