import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from transformers import CLIPVisionModel, CLIPProcessor
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoPerceiverResampler
from einops import rearrange
from typing import Tuple, Optional, Union
import math
from utils import load_pretrained_perceiver_from_file
import torch.utils.checkpoint as checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class BTCNorm(nn.BatchNorm1d):
    def __init__(self, dim: int, input_shape: str = 'BTC'):
        super().__init__(dim)
        assert input_shape in ['BTC', 'TBC']
        self.input_shape = input_shape

    def forward(self, x: torch.Tensor):
        if self.input_shape == 'TBC':
            x = rearrange(x, "T B C -> B C T")
            x = super().forward(x)
            x = rearrange(x, "B C T -> T B C")
        elif self.input_shape == 'BTC':
            x = rearrange(x, "B T C -> B C T")
            x = super().forward(x)
            x = rearrange(x, "B C T -> B T C")
        return x

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout)  # Dropout after attention
        self.ln_1 = BTCNorm(d_model, input_shape='TBC')

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.dropout2 = nn.Dropout(dropout)  # Dropout after feed-forward
        self.ln_2 = BTCNorm(d_model, input_shape='TBC')

        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.dropout1(self.attention(self.ln_1(x)))  # Add dropout after attention
        x = x + self.dropout2(self.mlp(self.ln_2(x)))  # Add dropout after feed-forward
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float = 0.0, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, dropout, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

def sinusoidal_pos_embedding(sequence_length, d_model):
    # Create sinusoidal frequencies
    frequencies = torch.zeros(sequence_length, d_model)
    for pos in range(sequence_length):
        for i in range(0, d_model, 2):
            div_term = math.pow(10000, 2 * i / d_model)
            frequencies[pos, i] = math.sin(pos / div_term)
            if i + 1 < d_model:
                frequencies[pos, i + 1] = math.cos(pos / div_term)
    
    return frequencies

def freeze_params(module: nn.Module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

class VisionTransformer(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 patch_size: int, 
                 width: int, 
                 sequence_length: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float = 0.0, 
                 projection: int = 1024, 
                 num_classes: int = 35, 
                 padding: Union[str, int] = 'same',
                 stride: int = 1):
        super().__init__()

        if padding == 'same':
            output_length = sequence_length + 1
            stride = 1
        else:
            output_length = (sequence_length // patch_size) + 1
            stride = patch_size

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False, padding=padding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))

        # Sinusoidal Positional Embedding
        self.pos_embedding = nn.Parameter(sinusoidal_pos_embedding(output_length, width), requires_grad=False)

        self.prenorm = BTCNorm(input_channels, input_shape='BTC')
        self.transformer = Transformer(width, num_layers, num_heads, dropout)
        self.ln_post = BTCNorm(width, input_shape='BTC')

        if projection is not None:
            self.projection = nn.Linear(width, projection)
        
        if num_classes is not None:
            self.head = nn.Linear(projection if projection is not None else width, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.prenorm(x)
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        B, T, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos_embed = self.pos_embedding.expand(B, -1, -1)
        x = x + pos_embed

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)

        x = self.projection(x) if hasattr(self, "projection") else x

        self.cls = x[:, 0, :]
        self.last_hidden_state = x[:, 1:, 0]
        
        return x
        
    def class_logits(self, imu_features: torch.Tensor, imu_features_perceiver: Optional[torch.Tensor], supervised_on_perceiver: bool) -> torch.Tensor:
        if supervised_on_perceiver and imu_features_perceiver is not None:
            return self.head(imu_features_perceiver)
        return self.head(imu_features)

    
class videoFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def forward(self, frames: torch.Tensor):
        with torch.no_grad():
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device) # Not sure why processor moves tensor to CPU but it does

            frame_features = self.model(**inputs).last_hidden_state

        return frame_features   

def print_memory(label):
    print(f"{label}: Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB, Cached: {torch.cuda.memory_cached() / 1024 ** 2} MB")
    
class perceivingContrastive(nn.Module):
    def __init__(self,
                 # config: dict = None,
                 input_channels: int = 12,
                 patch_size: int = 16,
                 width: int = 128,
                 sequence_length: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 stride: int = 1,
                 padding: str = 'same',
                 projection: int = 1024,
                 num_classes: int = 35
                 ):
        super().__init__()
        self.imu_encoder = VisionTransformer(input_channels=input_channels, patch_size=patch_size, width=width, sequence_length=sequence_length, num_heads=num_heads, num_layers=num_layers, dropout=dropout, stride=stride, padding=padding, projection=projection, num_classes=num_classes)

        self.perceiver = load_pretrained_perceiver_from_file('./saved_weights/mpt7b_perceiver.pt', dim=1024)
        freeze_params(self.perceiver)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, imu: torch.Tensor, video_class: torch.Tensor, video_perceiver: torch.Tensor, use_perceiver: bool = False, supervised_on_perceiver: bool = False, use_perceiver_on_video_only: bool = False) -> Tuple:
        imu_features, video_features = self.encode_features(imu, video_class, video_already_encoded=True)
        imu_features_perceiver = self.perceiver_pass(imu_features[:,1:,:])
        video_features_perceiver = video_perceiver
 
        pred = self.imu_encoder.class_logits(imu_features[:,0,:], imu_features_perceiver, supervised_on_perceiver)
        
        logits_per_imu, logits_per_video = self.compute_logits(imu_features[:,0,:], video_features, imu_features_perceiver, video_features_perceiver, use_perceiver, use_perceiver_on_video_only)
        
        return logits_per_imu, logits_per_video, pred, (imu_features[:,0,:], imu_features_perceiver), (video_features, video_features_perceiver)
    
    def encode_features(self, imu: torch.Tensor, video: torch.Tensor, video_already_encoded: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        imu_features = self.imu_encoder(imu)
        if not video_already_encoded:
            video_features = self.video_encoder(video)
        else:
            video_features = video
        return imu_features, video_features
    
    def perceiver_pass(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            perceived = self.perceiver(features.unsqueeze(2).unsqueeze(2)).squeeze().mean(1).mean(1)
        return perceived
    
    def cosine_similarity(self, imu, video):
        logit_scale = self.logit_scale.exp()
        
        imu_norm = torch.norm(imu, p=2, dim=1, keepdim=True)
        imu_normalized = imu / imu_norm
        
        video_norm = torch.norm(video, p=2, dim=1, keepdim=True)
        video_normalized = video / video_norm
        
        logits_per_imu = logit_scale * imu_normalized @ video_normalized.t()
        logits_per_video = logits_per_imu.t()
        return logits_per_imu, logits_per_video
    
    def compute_logits(self, imu_features: torch.Tensor, video_features: torch.Tensor, imu_features_perceiver: Optional[torch.Tensor], video_features_perceiver: Optional[torch.Tensor], use_perceiver: bool, use_perceiver_on_video_only: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if use_perceiver_on_video_only:
            logits_per_imu, logits_per_video = self.cosine_similarity(imu_features, video_features_perceiver)
            return logits_per_imu, logits_per_video
        
        if not use_perceiver:
            logits_per_imu, logits_per_video = self.cosine_similarity(imu_features, video_features)
            return logits_per_imu, logits_per_video

        logits_per_imu, logits_per_video = self.cosine_similarity(imu_features_perceiver, video_features_perceiver)
        return logits_per_imu, logits_per_video