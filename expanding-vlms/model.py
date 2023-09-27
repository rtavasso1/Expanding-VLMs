
# Import necessary libraries
import torch
import torch.nn as nn
import cv2
from collections import OrderedDict
import numpy as np
from torchvision.io import read_video
from transformers import CLIPVisionModel, CLIPProcessor
from scipy import interpolate
from scipy.signal import butter, lfilter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from .modeling_flamingo import FlamingoPerceiverResampler
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class BTCNorm(nn.BatchNorm1d):
    def forward(self, x: torch.Tensor):
        x = x.permute(1,0,2)
        x = x.permute(0,2,1)
        x = super().forward(x)
        x = x.permute(0,2,1)
        x = x.permute(1,0,2)
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = BTCNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = BTCNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_channels, patch_size, width, sequence_length, num_heads, num_layers,dropout=0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        #self.linear = nn.Linear(sequence_length, width)

        self.scale = width ** -0.5
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.pos_embedding = nn.Parameter(self.scale * torch.randn((sequence_length // patch_size)+ 1, width))
        self.ln_pre = LayerNorm(width)
        self.prenorm = nn.BatchNorm1d(input_channels)

        self.transformer = Transformer(width, num_layers, num_heads)

        self.ln_post = LayerNorm(width)
        
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
        
    def downsample(self, data, interp):
        # Your original timestamps (assuming it starts at 0 and ends at 1)
        original_timestamps = np.linspace(0, 1, data.shape[0])

        # Your new timestamps
        new_timestamps = np.linspace(0, 1, interp[-1].item()) # interp is [data_length, shortest_length_file_length]

        # A placeholder for your downsampled data
        downsampled_data = np.zeros((interp[-1].item(), 3))

        # Loop through each feature
        for i in range(3):
            # Create a cubic spline interpolation for the i-th feature
            filtered_data = self.butter_lowpass_filter(data[:,i], cutoff=25, fs=100, order=5)
            cs = interpolate.CubicSpline(original_timestamps, filtered_data)

            # Interpolate the downsampled data points for the i-th feature
            downsampled_data[:, i] = cs(new_timestamps)
        return downsampled_data

    def extract_imu_windows(self, imu_path, num_frames, interp_size=None):
        if isinstance(imu_path, str):
            # check if file exists
            if not os.path.exists(imu_path):
                raise FileNotFoundError(f"No such file: '{imu_path}'")
                
            imu = pd.read_csv(imu_path, header=None)
            imu = imu.to_numpy()[:,1:] # filter timestamp
            
            if interp_size is not None:
                imu = imu[:interp_size[0]]
                if interp_size[0] > interp_size[1]:
                    imu = self.downsample(imu, interp_size)
            
            if num_frames == 1:
                frame_step = 0
                window_size = 256
            else: 
                frame_step = 50  # sampling rate for IMU (different from video sampling rate)
                window_size = 64  # window is centered around the frame
            imu_windows = np.zeros((num_frames, window_size, imu.shape[1]))
            for i in range(num_frames):
                start = max(0, (i+1)*frame_step-window_size)
                end = min(imu.shape[0], (i+1)*frame_step+window_size)
                window = imu[start:end]
                if window.shape[0] < window_size:  # if window is smaller than expected
                    #print(window.shape)
                    pad = np.zeros((window_size - window.shape[0], window.shape[1]))
                    window = np.concatenate((window, pad), axis=0)  # pad zeros at the end
                imu_windows[i] = window

            return imu_windows
        
        elif isinstance(imu_path, list):
            assert isinstance(num_frames, int) and len(imu_path) == 4, "'num_frames' must be a list of same length as 'imu_path'"
            assert (interp_size is None) or len(imu_path) == len(interp_size), "'interp_size' must be a list of same length as 'imu_path' or None"
            
            imu_windows = []
            for j, path in enumerate(imu_path):
                imu_window = self.extract_imu_windows(path, num_frames, [interp_size[j],interp_size[-1]])
                imu_windows.append(imu_window)
            imu_windows = np.concatenate(imu_windows, axis=2)
            return imu_windows
        
        elif all(isinstance(item, tuple) for item in imu_path): # list of tuples
            #print(len(interp_size))
            assert isinstance(num_frames, list) and len(imu_path) == len(num_frames), "'num_frames' must be a list of same length as 'imu_path'"
            assert (interp_size is None) or len(imu_path) == len(interp_size), "'interp_size' must be a tuple of same length as 'imu_path' or None"

            combined_sensor_windows = []
            for k, tupleOfPaths in enumerate(imu_path):
                listOfPaths = list(tupleOfPaths)
                imu_windows = self.extract_imu_windows(listOfPaths,num_frames[k],interp_size[k])
                #imu_windows = np.expand_dims(imu_windows, 0)
                combined_sensor_windows.append(imu_windows)
                #print(imu_windows.shape)
            combined_sensor_windows = np.concatenate(combined_sensor_windows,axis=0)
            #combined_sensor_windows = torch.tensor(combined_sensor_windows).float().to(device)
            #print(combined_sensor_windows.shape)
            # num_frames can be different for each video so the number of windows extracted can mismatch and will make ragged array so we keep as list
            return combined_sensor_windows

    def forward(self, x: torch.Tensor, num_frames, interp_size):
        x = self.extract_imu_windows(x, num_frames, interp_size)
        x = torch.from_numpy(x).float().to(device)
        x = x.permute(0,2,1)
        x = self.prenorm(x)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.permute(0,2,1)

        B, T, C = x.shape


        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        pos_embed = self.pos_embedding.expand(B, -1, -1)
        x = x + pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        return x
    
class videoFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_frames_torchvision(self,video_path, frame_step):
        if isinstance(video_path, str):
            video, audio, info = read_video(video_path, pts_unit='sec')

            # assuming video frames are of shape (T, H, W, C)
            num_frames = video.shape[0]

            # Generate a list of frame indices that we are interested in
            if frame_step is not None:
                frame_indices = list(range(frame_step, num_frames, frame_step))
            else:
                try:
                    frame_indices = [np.random.randint(30,num_frames-30)] # sample random frame not in first or last second
                except:
                    frame_indices = [np.random.randint(0,num_frames)]

            # Index the video tensor with the list of frame indices
            selected_frames = video[frame_indices].float().permute(0, 3, 1, 2)

            return selected_frames.to(device), len(frame_indices)

        elif isinstance(video_path, tuple):
            video, num_frames_list = [], []
            for path in video_path:
                #print(path)
                selected_frames, num_frames = self.extract_frames_torchvision(path, frame_step)
                #print(selected_frames.shape)
                video.append(selected_frames)
                num_frames_list.append(num_frames)
            video = torch.cat(video, dim=0)
            return video, num_frames_list

    def extract_frames_cv2(self, video_path, frame_step):
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)

            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_step is not None:
                frame_indices = list(range(frame_step, num_frames, frame_step))
            else:
                try:
                    frame_indices = [np.random.randint(30, num_frames-30)] # sample random frame not in first or last second
                except:
                    frame_indices = [np.random.randint(0, num_frames)]

            selected_frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converting from BGR to RGB
                selected_frames.append(torch.from_numpy(frame).float().permute(2, 0, 1))

            cap.release()

            selected_frames = torch.stack(selected_frames).to(device)

            return selected_frames, len(frame_indices)

        elif isinstance(video_path, tuple):
            video, num_frames_list = [], []
            for path in video_path:
                selected_frames, num_frames = self.extract_frames_cv2(path, frame_step)
                video.append(selected_frames)
                num_frames_list.append(num_frames)
            video = torch.cat(video, dim=0)
            return video, num_frames_list
    
    def extract_frames_cv2_with_multithread(self, video_paths, frame_step):
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for video_path in video_paths:
                futures.append(executor.submit(self.extract_frames_cv2, video_path, frame_step))

            results = [future.result() for future in futures]

            videos = [result[0] for result in results]
            num_frames_list = [result[1] for result in results]

            videos = torch.cat(videos, dim=0)
            return videos, num_frames_list
    
    def forward(self, video_paths, pooled):
        # get frames
        frames, num_frames = self.extract_frames_cv2_with_multithread(video_paths, None)

        inputs = self.processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            if pooled:
                frame_features = self.model(**inputs).pooler_output
            else:
                frame_features = self.model(**inputs).last_hidden_state

        return frame_features, num_frames
    
class contrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=128, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0, stride=1, padding='same').to(device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, video_path, imu_path, interp_size, return_features=False):
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size)
        
        video_features = torch.cat([vid[0].unsqueeze(0) for vid in video_path], dim=0).to(device) # (B, 1024)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        imu_features = imu_features / imu_features.norm(dim=1, keepdim=True)
        
        if return_features==True:
            return imu_features, video_features

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_imu = logit_scale * imu_features @ video_features.t()
        logits_per_video = logits_per_imu.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_imu, logits_per_video
    
class contrastiveSequenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=1024//8, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0, stride=1, padding='same').to(device)
        self.IMUEncoder.pooled = False # return sequence only during inference
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.upCast = nn.Linear(1024//8,1024)
        self.head = nn.Linear(1024,35)

    def forward(self, video_path, imu_path, interp_size, return_features=False):
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size) # (B, 1024)
        video_features = torch.cat([vid[0].unsqueeze(0) for vid in video_path], dim=0).to(device) # (B, 1024)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        imu_features = imu_features / imu_features.norm(dim=1, keepdim=True)
        
        if return_features==True:
            return imu_features, video_features

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_imu = logit_scale * imu_features @ video_features.transpose(-2,-1) # (B, 256, 256)
        logits_per_video = logits_per_imu.t()
        pred = self.head(imu_features)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_imu, logits_per_video, pred
    
class supervisedArithmetic(nn.Module):
    def __init__(self):
        super().__init__()
        #self.videoEncoder = videoFrameEncoder().to(device)
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=1024, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0).to(device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.head = nn.Linear(1024,35)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, video_path, imu_path, interp_size, return_features=False):
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size)
        video_features = torch.cat([vid[0].unsqueeze(0) for vid in video_path], dim=0).to(device) # (B, 256, 1024)
        
        features = video_features + imu_features
        features = features / features.norm(dim=1, keepdim=True)
        
        if return_features==True:
            return imu_features, video_features

        out = self.head(features)
        return out
    
class perceivingContrastive(nn.Module):
    def __init__(self):
        super().__init__()
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=128, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0, stride=1, padding='same').to(device)
        self.IMUEncoder.pooled = True # return sequence
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.perceiver = FlamingoPerceiverResampler(dim=1024)
        self.perceiver.load_state_dict(torch.load('./otter/perceiverModule.pt'))
        self.perceiver.eval()
        for param in self.perceiver.parameters():
            param.requires_grad = False
        
        self.head = nn.Linear(1024,35)
    
    def forward(self, video_path, imu_path, interp_size, return_features=False):
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size) # (B, 256, 128)       
        video_features = torch.cat([vid[1:].unsqueeze(0) for vid in video_path], dim=0).to(device) # (B, 256, 1024)
        
        imu_features = self.perceiver(imu_features.unsqueeze(1).unsqueeze(1)).squeeze().mean(1) # (B, 1024)
        video_features = self.perceiver(video_features.unsqueeze(1).unsqueeze(1)).squeeze().mean(1) # (B, 1024)
        
        pred = self.head(imu_features)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        imu_features = imu_features / imu_features.norm(dim=1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_imu = logit_scale * imu_features @ video_features.t()
        logits_per_video = logits_per_imu.t()
        
        return logits_per_imu, logits_per_video, pred
    
class supervisedIMUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=128, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0, stride=1, padding='same').to(device)
        self.IMUEncoder.pooled = True # return sequence
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.perceiver = FlamingoPerceiverResampler(dim=1024)
        self.perceiver.load_state_dict(torch.load('./otter/perceiverModule.pt'))
        self.perceiver.eval()
        for param in self.perceiver.parameters():
            param.requires_grad = False
        
        self.head = nn.Linear(1024,35)
    
    def forward(self, video_path, imu_path, interp_size, return_features=False):
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size).unsqueeze(1).unsqueeze(1) # (B, 1, 1, 256, 128)
        imu_features = self.perceiver(imu_features).squeeze()
        imu_features = imu_features.mean(1)
        out = self.head(imu_features)
        return out

class mseContrastiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.videoEncoder = videoFrameEncoder().to(device)
        self.IMUEncoder = VisionTransformer(input_channels=12, patch_size=16, width=1024, sequence_length=256, num_heads=8, num_layers=8, dropout=0.0).to(device)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.mse = nn.MSELoss(reduction='none')

    def pairwise_mse(self, X, Y):
        X_exp = X.unsqueeze(1)  # X_exp becomes of size (B, 1, C)
        Y_exp = Y.unsqueeze(0)  # Y_exp becomes of size (1, B, C)

        # Compute pairwise squared differences
        diff = X_exp - Y_exp  # diff is of size (B, B, C) due to broadcasting

        # Compute mean squared error along the third dimension
        mse = (diff ** 2).mean(dim=2)  # mse is of size (B, B)
        epsilon = 1e-8  # small constant
        reciprocal_mse = torch.reciprocal(mse + epsilon)
        return reciprocal_mse

    def forward(self, video_path, imu_path, interp_size, return_features=False):
        #video_features, num_frames = self.videoEncoder(video_path)
        imu_features = self.IMUEncoder(imu_path, [1 for _ in range(len(imu_path))], interp_size)
        
        video_features = torch.from_numpy(torch.concatenate(video_path, axis=0)).to(device)
        #imu_features = torch.from_numpy(np.concatenate(imu_path, axis=0)).to(device)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        imu_features = imu_features / imu_features.norm(dim=1, keepdim=True)
        
        if return_features==True:
            return imu_features, video_features

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_imu = logit_scale * self.pairwise_mse(imu_features,video_features)
        logits_per_video = logits_per_imu.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_imu, logits_per_video
