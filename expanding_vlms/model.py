import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from transformers import CLIPVisionModel, CLIPProcessor, CLIPVisionModelWithProjection
from Otter.src.otter_ai.models.flamingo.modeling_flamingo import FlamingoPerceiverResampler
from einops import rearrange
from typing import Tuple, Optional, Union
import math
from utils import load_pretrained_perceiver_from_file, independent_encoders_paths
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
        
class OldVisionTransformer(nn.Module):
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
                 stride: int = 1,
                 num_class_tokens: int = 1,
                 loss_remapping: bool = False,
                 **kwargs):
        super().__init__()

        if padding == 'same':
            output_length = sequence_length + num_class_tokens
            stride = 1
        else:
            output_length = (sequence_length // patch_size) + num_class_tokens
            stride = patch_size

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False, padding=padding)
        self.cls_token = nn.Parameter(torch.randn(1, num_class_tokens, width))

        # Sinusoidal Positional Embedding
        self.pos_embedding = nn.Parameter(sinusoidal_pos_embedding(output_length, width), requires_grad=False)

        self.prenorm = BTCNorm(input_channels, input_shape='BTC')
        self.transformer = Transformer(width, num_layers, num_heads, dropout)
        self.ln_post = BTCNorm(width, input_shape='BTC')

        if projection is not None:
            self.projection = nn.Linear(width, projection)
        
        if num_classes is not None:
            self.head = nn.Linear(projection if projection is not None else width, num_classes)
            
        if loss_remapping:
            #self.loss_mapper = nn.Conv2d(num_class_tokens,1,1)
            self.loss_mapper = nn.Parameter(torch.randn(num_class_tokens))
            self.temp = nn.Parameter(torch.ones(1))
            pass

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
                 stride: int = 1,
                 num_class_tokens: int = 1,
                 loss_remapping: bool = False,
                 **kwargs):
        super().__init__()

        if padding == 'same':
            output_length = sequence_length + num_class_tokens
            stride = 1
        else:
            output_length = (sequence_length // patch_size) + num_class_tokens
            stride = patch_size

        self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False, padding=padding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.reg_token = nn.Parameter(torch.randn(1, num_class_tokens-1, width))

        # Sinusoidal Positional Embedding
        self.pos_embedding = nn.Parameter(sinusoidal_pos_embedding(output_length, width), requires_grad=False)

        self.prenorm = BTCNorm(input_channels, input_shape='BTC')
        self.transformer = Transformer(width, num_layers, num_heads, dropout)
        self.ln_post = BTCNorm(width, input_shape='BTC')

        if projection is not None:
            self.projection = nn.Linear(width, projection)
            self.video_projection = nn.Linear(projection, width)
        
        if num_classes is not None:
            self.head = nn.Linear(projection if projection is not None else width, num_classes)
            
        if loss_remapping:
            #self.loss_mapper = nn.Conv2d(num_class_tokens,1,1)
            self.loss_mapper = nn.Parameter(torch.randn(num_class_tokens))
            self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, video: torch.Tensor = None):
        x = self.prenorm(x)
        x = x.permute(0,2,1)
        x = self.conv1d(x)
        x = x.permute(0,2,1)
        B, T, C = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.reg_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, reg_tokens, x], dim=1)
        
        if video is not None:
            #print(x.shape,video.shape)
            video = self.video_projection(video)
            x = torch.cat([x, video], dim=1)

        pos_embed = self.pos_embedding.expand(B, -1, -1)
        if video is None:
            #pos_embed = pos_embed[:,:-64,:]
            pass
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
    
class CombinedVisionTransformers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.imu_encoders = torch.nn.ModuleList([VisionTransformer(input_channels=12, patch_size=16, width=256, sequence_length=256, num_heads=8, num_layers=8, dropout=0.3, stride=16, padding=0, projection=1024, num_classes=35, num_class_tokens=2, loss_remapping=True) for _ in range(64)])

        self.blacklist = [2, 5, 7, 9, 17, 21, 26, 32, 33, 35, 39, 41, 49, 53, 55]
        # Correctly load state dictionaries for each VisionTransformer
        for i, encoder in enumerate(self.imu_encoders):
            encoder_state = torch.load(independent_encoders_paths[i])['model_state_dict']
            
            # Adjust state_dict keys if necessary
            adjusted_state_dict = {key.partition('.')[2]: value for key, value in encoder_state.items()}
            encoder.load_state_dict(adjusted_state_dict)
    
    def forward(self, imu):
        tokens = []
        for encoder in self.imu_encoders:
            token = encoder(imu)[:,1,:].unsqueeze(1) # get the reg token
            tokens.append(token)
        tokens = torch.cat(tokens, dim=1)
        #tokens[self.blacklist] = tokens[self.blacklist] * 0
        
        return tokens      


class Discriminator(nn.Module):
    def __init__(self, output_length=129, width=1024, num_layers=8, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))

        # Sinusoidal Positional Embedding
        self.pos_embedding = nn.Parameter(sinusoidal_pos_embedding(output_length, width), requires_grad=False)

        #self.prenorm = BTCNorm(width, input_shape='BTC')
        self.transformer = Transformer(width, num_layers, num_heads, dropout)
        self.ln_post = BTCNorm(width, input_shape='BTC')
        
        self.projection = nn.Linear(width, 1)
        self.sigmoid = nn.Sigmoid()
        
    def shuffle_half_of_tensors(self, imu_data):
        """
        Randomly shuffle the first half of the IMU such that it no longer is overlapping with paired image

        Parameters:
        imu_data (torch.Tensor): The tensor containing IMU data.
        image_data (torch.Tensor): The tensor containing image data.

        Returns:
        torch.Tensor, torch.Tensor: Shuffled IMU data, image data
        """
        # Calculate the midpoint of the batch
        mid_point = len(imu_data) // 2

        # Select the first half of each tensor
        imu_first_half = imu_data[:mid_point]

        # Shuffle the first half
        perm = torch.randperm(mid_point)
        imu_shuffled = imu_first_half[perm]

        # Concatenate the shuffled half with the unshuffled second half
        imu_data_shuffled = torch.cat((imu_shuffled, imu_data[mid_point:]), dim=0)

        return imu_data_shuffled
        
    def forward(self, imu, image):
        B, T, C = imu.shape
        shuffled_imu = self.shuffle_half_of_tensors(imu) # first half of imu is permuted, such that labels are [0, 0, 0, ..., 1, 1, 1]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, shuffled_imu, image),dim=1) # B, 64, 1024 => B, 129, 1024
        
        pos_embed = self.pos_embedding.expand(B, -1, -1)
        x = x + pos_embed
        
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        
        x = self.projection(x[:,0,:]) # 1d tensor of size batch_size
        #x = self.sigmoid(x)
        
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder_fc1 = nn.Linear(64*1024, 32000)
        self.encoder_fc2 = nn.Linear(32000, 16000)
        self.encoder_fc3 = nn.Linear(16000, 1024)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(1024, 16000)
        self.decoder_fc2 = nn.Linear(16000, 32000)
        self.decoder_fc3 = nn.Linear(32000, 64*1024)

    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = self.encoder_fc3(x)
        return x

    def decode(self, x):
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x)
        return x

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded.view(x.size(0), 64, 1024)  # Reshape back to original dimensions
    
class videoFrameEncoder(nn.Module):
    def __init__(self, projection=False):
        super().__init__()
        self.projection = projection
        if not projection:
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        else:
            self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    def forward(self, frames: torch.Tensor):
        with torch.no_grad():
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(device) # Not sure why processor moves tensor to CPU but it does

            frame_features = self.model(**inputs).last_hidden_state
            if self.projection:
                frame_features = self.model.visual_projection(frame_features)

        return frame_features   

def print_memory(label):
    print(f"{label}: Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB, Cached: {torch.cuda.memory_cached() / 1024 ** 2} MB")
    
class perceivingContrastive(nn.Module):
    def __init__(self,
                 input_channels: int = 12,
                 patch_size: int = 16,
                 width: int = 256,
                 sequence_length: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.3,
                 stride: int = 1,
                 padding: str = 0,
                 projection: int = 1024,
                 num_classes: int = 35,
                 num_class_tokens: int = 2,
                 loss_remapping: bool = True,
                 num_encoders: int = 1,
                 config = None
                 ):
        super().__init__()
        if config is None or config.combined_encoders == False:
            self.imu_encoders = torch.nn.ModuleList([VisionTransformer(input_channels=input_channels, patch_size=patch_size, width=width, sequence_length=sequence_length, num_heads=num_heads, num_layers=num_layers, dropout=dropout, stride=stride, padding=padding, projection=projection, num_classes=num_classes, num_class_tokens=num_class_tokens, loss_remapping=loss_remapping) for _ in range(num_encoders)])
        else:
            self.imu_encoders = CombinedVisionTransformers()

        if False:
            self.perceiver = load_pretrained_perceiver_from_file('./saved_weights/mpt7b_perceiver.pt', dim=1024)
            freeze_params(self.perceiver)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.num_class_tokens = num_class_tokens
        self.padding = padding
        
        self.config = config
        
        #self.discriminator = Discriminator()
    
    def forward(self, imu: torch.Tensor, video_class: torch.Tensor, video_perceiver: torch.Tensor, use_perceiver: bool = False, supervised_on_perceiver: bool = False, use_perceiver_on_video_only: bool = False, metrics_on_perceiver: bool = False, virtual_batch_size = 256, sequence_wise = False) -> Tuple:
        imu_features, video_features = self.encode_features(imu, video_class, append_video_to_imu=False)
        
        if use_perceiver or supervised_on_perceiver or metrics_on_perceiver:
            imu_features_perceiver = self.perceiver_pass(imu_features[:,:self.num_class_tokens,:])#.mean(1)
        else:
            imu_features_perceiver = None
        video_features_perceiver = video_perceiver#.mean(1)
        
        supervised_pred = []
        if self.config is None or self.config.combined_encoders == False:
            for i, imu_encoder in enumerate(self.imu_encoders):
                supervised_pred.append(imu_encoder.class_logits(imu_features[i][:,0,:], imu_features_perceiver, supervised_on_perceiver))
            supervised_pred = torch.vstack(supervised_pred)
        else:
            supervised_pred = None
        
        
        #supervised_pred = self.imu_encoder.class_logits(imu_features[:,0,:], imu_features_perceiver, supervised_on_perceiver)
        discriminator_pred = None #self.discriminator(imu_features[:,1:self.num_class_tokens,:],video_features_perceiver)
        
        if False:
            processed_imu_features = []
            for feature in imu_features:
                for _ in range(64):
                    processed_imu_features.append(feature[:,1,:].unsqueeze(1))
                #B = feature.shape[0]
            if len(processed_imu_features) == 1:
                # zeros = torch.zeros((B,64-len(processed_imu_features),1024)).half().to('cuda')
                # processed_imu_features.append(zeros)
                pass
            processed_imu_features = torch.cat(processed_imu_features, dim=1)
            #processed_imu_features[:, self.config.token_num, :] = processed_imu_features[:, 0, :] # make sure we are computing the logits correctly
        else:
            processed_imu_features = imu_features[0][:,1:65,:]
            video_features = video_features[:,1:65,:]
            
        
        logits_per_imu, logits_per_video = self.compute_logits(processed_imu_features, video_features, imu_features_perceiver, video_features_perceiver, use_perceiver, use_perceiver_on_video_only, virtual_batch_size=virtual_batch_size, sequence_wise=sequence_wise) # imu_features[:,1:self.num_class_tokens,:].squeeze()
        
        if sequence_wise:
            if len(video_features.shape) == 3:
                video_sequence = video_features[:,1:257,:]
                video_features = video_features[:,0,:] # return only class token, have to do bc im using multiple file configs that treat this differently
            else:
                video_sequence = None
        else:
            video_sequence = None
         
        return logits_per_imu, logits_per_video, (supervised_pred, discriminator_pred), (imu_features[0][:,0,:], imu_features_perceiver, processed_imu_features), (video_features, video_features_perceiver, video_sequence) # imu_features[:,1:self.num_class_tokens,:]
    
    def encode_features(self, imu: torch.Tensor, video_features: torch.Tensor, append_video_to_imu: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if append_video_to_imu:
            video_to_append = video_features
        else:
            video_to_append = None
        imu_features = []
        if self.config is None or self.config.combined_encoders == False:
            for imu_encoder in self.imu_encoders:
                imu_features.append(imu_encoder(imu, video_to_append))
        else:
            imu_features.append(self.imu_encoders(imu))

        return imu_features, video_features
    
    def perceiver_pass(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            perceived = self.perceiver(features.unsqueeze(1).unsqueeze(1)).squeeze()#.mean(1).mean(1)
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
    
    def cosine_similarity_sequence(self, imu, video):
        logit_scale = self.logit_scale.exp()
        
        imu_norm = torch.norm(imu, p=2, dim=2, keepdim=True) # (B, T, C)
        imu_normalized = imu / imu_norm
        
        video_norm = torch.norm(video, p=2, dim=2, keepdim=True) # (B, T, C)
        video_normalized = video / video_norm
        
        imu_normalized = imu_normalized.permute(1,0,2)
        video_normalized = video_normalized.permute(1,0,2) # (T, B, C)
        
        logits_per_imu = logit_scale * imu_normalized @ video_normalized.permute(0,2,1) # (T, B, B)
        logits_per_video = logits_per_imu.permute(0,2,1)
        return logits_per_imu, logits_per_video
    
    def cosine_similarity_flatten_sequence(self, imu, video):
        B, T, C = imu.shape
        
        imu = imu.reshape(B,T*C)
        video = video.reshape(B,T*C)
        
        logits_per_imu, logits_per_video = self.cosine_similarity(imu,video)
        
        return logits_per_imu.unsqueeze(0), logits_per_video.unsqueeze(0) # unsqueeze first dim to work with sequence logic when sequence_wise=True
    
    def compute_logits(self, imu_features: torch.Tensor, video_features: torch.Tensor, 
                   imu_features_perceiver: Optional[torch.Tensor], 
                   video_features_perceiver: Optional[torch.Tensor], 
                   use_perceiver: bool, use_perceiver_on_video_only: bool, 
                   virtual_batch_size: int, sequence_wise: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if virtual_batch_size == -1: virtual_batch_size = imu_features.size(0)
        
        def shuffle_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            indices = torch.randperm(tensor1.size(0))
            return tensor1[indices], tensor2[indices]

        def compute_virtual_batch_logits(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            logits_per_imu, logits_per_video = [], []
            #for _ in range(0, tensor1.size(0)//virtual_batch_size):
            #tensor1, tensor2 = shuffle_tensors(tensor1, tensor2)
            
            for i in range(0, tensor1.size(0), virtual_batch_size):
                batch_imu = tensor1[i:i + virtual_batch_size]
                batch_video = tensor2[i:i + virtual_batch_size]
                if sequence_wise:
                    logits_imu1, logits_video1 = self.cosine_similarity_sequence(batch_imu, batch_video)
                    logits_imu2, logits_video2 = self.cosine_similarity_flatten_sequence(batch_imu, batch_video)
                    logits_imu = torch.vstack((logits_imu1,logits_imu2))
                    logits_video = torch.vstack((logits_video1,logits_video2))
                else:
                    logits_imu, logits_video = self.cosine_similarity(batch_imu, batch_video)
                logits_per_imu.append(logits_imu)
                logits_per_video.append(logits_video)
            return logits_per_imu, logits_per_video

        # Selecting the appropriate feature tensors
        if use_perceiver_on_video_only:
            imu, video = imu_features, video_features_perceiver
        elif not use_perceiver:
            imu, video = imu_features, video_features
        else:
            imu, video = imu_features_perceiver, video_features_perceiver

        logits_per_imu, logits_per_video = compute_virtual_batch_logits(imu, video)

        return logits_per_imu, logits_per_video