
# Import necessary libraries
import torch
from torch.utils.data import Dataset
import numpy as np

class filepathLoader(Dataset):
    def __init__(self, video_files, imu_files, interp_size): # as dict handles functionality for when the multiple cameras are combined in dictionary
        self.video_files = video_files
        self.imu_files = imu_files
        self.interp_size = torch.tensor(interp_size)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        return self.video_files[idx], self.imu_files[idx], self.interp_size[idx]
    
def collate_fn(batch):
    return zip(*batch)

class dictionaryFilepathLoader(Dataset):
    def __init__(self, listOfDicts):
        self.listOfDicts = listOfDicts
        
    def __len__(self):
        return len(self.listOfDicts)
    
    def __getitem__(self, idx):
        imu = self.listOfDicts[idx]['imu']
        interp = self.listOfDicts[idx]['interp']
        num_videos = len(self.listOfDicts[idx]['video'])
        video_num = np.random.randint(num_videos)
        video = self.listOfDicts[idx]['vid_embeds'][video_num]
        
        return video, imu, torch.tensor(interp)