import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the relative path to Otter/src from the current directory
sys.path.append(os.path.join(current_dir, "Otter", "src"))

from utils import runs_of_interest
from utils import load_pretrained_perceiver_from_file
from model import perceivingContrastive, sinusoidal_pos_embedding, CombinedVisionTransformers, OldVisionTransformer, VisionTransformer
from data import load_all_data, prepare_dataloader_precomputed_perceiver_embeddings
from train import compute_loss, compute_metrics
import torch
import torch.nn as nn
import argparse
from dataclasses import dataclass, field
from typing import List, Optional

run_names = {0: {'name': 'vital-sweep-15', 'path': '../Checkpoints/9sf4pyuw/vital-sweep-15/epoch160_avg_train_loss_0.774.pt'},
             1: {'name': 'exalted-sweep-30', 'path': 'firstTest/Expanding-VLMs/Checkpoints/9sf4pyuw/exalted-sweep-30/epoch200_avg_train_loss_0.916.pt'},
             2: {'name': 'youthful-sweep-77', 'path': 'firstTest/Expanding-VLMs/Checkpoints/rjujwu12/youthful-sweep-77/epoch300_avg_train_loss_2.045.pt'},
             3: {'name': 'lemon-sweep-1', 'path': 'firstTest/Expanding-VLMs/Checkpoints/mf5mnb1j/lemon-sweep-1/epoch490_avg_train_loss_4.026.pt'},
             4: {'name': 'classic-sweep-36', 'path': '../Checkpoints/km7s9903/classic-sweep-36/epoch250_avg_train_loss_4.051.pt'},
             5: {'name': 'good-sweep-1', 'path': '../Checkpoints/764jxsre/good-sweep-1/epoch610_avg_train_loss_2.982.pt'},
             6: {'name': 'peachy-sweep-1', 'path': '../Checkpoints/ybhzyc1f/peachy-sweep-1/epoch90_avg_train_loss_5.570.pt'}}



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metrics(model, dataloader, lossfcn, device, config, sequence_wise):
    model = model.to(device)
    model.eval()
    epoch_test_loss = torch.tensor(0.0).to(device)
    epoch_test_contrastive_loss = torch.tensor(0.0).to(device)
    epoch_test_supervised_loss = torch.tensor(0.0).to(device)

    # Initialize lists to store batch-wise metrics for averaging later
    cosine_similarity_tests = []
    recall_tests = {f"Test Recall at {k_val}": [] for k_val in [1,5,10]}
    acc_tests = []

    if config.use_perceiver:
        perceiver = load_pretrained_perceiver_from_file('./saved_weights/mpt7b_perceiver.pt', dim=1024).to(device)
    
    with torch.no_grad():
        for i, (imu, video_class, video_perceiver, y) in enumerate(dataloader):
            imu, video_class, video_perceiver, y = imu.to(device), video_class.to(device), video_perceiver.to(device), y.to(device)
            #video_class = video_class[:,1:,:]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                imu_features = model(imu, None)
                if not config.combined_encoders:
                    imu_features = imu_features[:,1:config.num_class_tokens,:]
                    
                if config.use_perceiver:
                    imu_features = perceiver(imu_features.reshape(-1, 1, 1, 256, 1024)).reshape(-1,64,1024)
                    #video_perceiver = perceiver(video_class.reshape(-1, 1, 1, 256, 1024)).reshape(-1,64,1024)
                #print(imu_features.shape)

                # Calculate and log additional metrics for validation
                cosine_similarity_test, recall_test, acc_test = compute_metrics(None, 
                                                                                  None, 
                                                                                  (None, None, imu_features), 
                                                                                  (video_class, video_perceiver, None),
                                                                                  config.metrics_on_perceiver,
                                                                                  temperature=1,
                                                                                  k=config.k,
                                                                                  text_embeds=None,
                                                                                  y=y,
                                                                                  use_perceiver_on_video_only=config.use_perceiver_on_video_only, 
                                                                                  current_epoch=None,
                                                                                  config=config,
                                                                                  evaluate=True)

            for k_val, recall_value in recall_test.items():
                recall_tests[f"Test Recall at {k_val}"].append(recall_value)


        recall_test_averaged = {key: sum(value) / len(value) for key, value in recall_tests.items()}

    return recall_test_averaged

@dataclass
class Config:
    token_num: List[int] = field(default_factory=list)
    combined_encoders: bool = True
    virtual_batch_size: int = -1
    supervised_on_perceiver: bool = False
    metrics_on_perceiver: bool = False
    use_perceiver: bool = False
    use_perceiver_on_video_only: bool = True
    contrast_on_sequence: bool = True
    use_contrastive_loss: bool = True
    use_supervised_loss: bool = True
    k: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.token_num:
            self.token_num = [i for i in range(64)]
        if not self.k:
            self.k = [1, 5, 10]

def oldMain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_number', type=int, default=0)
    args = parser.parse_args()

    config = runs_of_interest[run_names[args.run_number]['name']]

    lossfcn = torch.nn.CrossEntropyLoss()

    for sequence_wise, num_class_tokens, path in zip([True, False],[257,257],[path, None]):
        dataset = load_all_data(root_dirs, video_root_dir, path=path)
        train_dataloader, test_dataloader, label_mapping = prepare_dataloader_precomputed_perceiver_embeddings(dataset, batch_size=256, num_workers=4)

        model = perceivingContrastive(patch_size=config['patch_size']['value'],
                                         width=config['width']['value'],
                                         dropout=config['dropout']['value'],
                                         padding=config['padding']['value'],
                                         num_class_tokens=num_class_tokens).to(device)

        checkpoint = torch.load(run_names[args.run_number]['path'])
        state_dict = checkpoint['model_state_dict']

        #if sequence_wise:
        #    cls_token = state_dict['cls_token']
        #    repeated_cls_token = cls_token.repeat(1, 257, 1)  # Repeat along dim1
        #    state_dict['cls_token'] = repeated_cls_token

        #    adjusted_pos_embedding = torch.cat([pos_embedding, torch.zeros(256, 128)], dim=0) 
        #    state_dict['pos_embedding'] = adjusted_pos_embedding

        model.imu_encoder.load_state_dict(state_dict)


        metrics = get_metrics(model, test_dataloader, lossfcn, device, config, sequence_wise)
        print(metrics)

@dataclass
class VTConfig:
    input_channels: int = 12
    patch_size: int = 16
    width: int = 256
    sequence_length: int = 256
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.3
    projection: int = 1024
    num_classes: int = 35
    padding: int = 0
    stride: int = 1
    num_class_tokens: int = 65
    loss_remapping: bool = True
    path: Optional[str] = None  # Use Optional for nullable types
    token_num: List[int] = field(default_factory=lambda: [i for i in range(64)])
    k: List[int] = field(default_factory=lambda: [1, 5, 10])
    combined_encoders: bool = False
    metrics_on_perceiver: bool = False
    use_perceiver_on_video_only: bool = True
    use_perceiver: bool = False

    def __post_init__(self):
        if not self.token_num:
            self.token_num = [i for i in range(64)]
        if not self.k:
            self.k = [1, 5, 10]
        
# idx: (model_class, supervised config, unsupervised config)
runs = {0: (CombinedVisionTransformers, VTConfig(combined_encoders=True), None),
        1: (OldVisionTransformer, VTConfig(path='../Checkpoints/upxew53l/fancy-sweep-8/epoch340_avg_train_loss_1.391.pt'), VTConfig(path='../Checkpoints/upxew53l/toasty-sweep-10/epoch290_avg_train_loss_1.536.pt')), # temperature 1
        2: (OldVisionTransformer, VTConfig(path='../Checkpoints/t64xgprz/dashing-sweep-1/epoch240_avg_train_loss_2.633.pt'), VTConfig(path='../Checkpoints/t64xgprz/breezy-sweep-2/epoch260_avg_train_loss_2.466.pt')), # temperature 5
        3: (OldVisionTransformer, VTConfig(path='../Checkpoints/xzfuxtdd/pretty-sweep-1/epoch280_avg_train_loss_2.772.pt'), VTConfig(path='../Checkpoints/xzfuxtdd/comic-sweep-2/epoch270_avg_train_loss_2.683.pt')), # temperature 10
        4: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/jolly-sweep-1/epoch500_avg_train_loss_3.759.pt')), # dataset ablation 1
        5: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/trim-sweep-2/epoch380_avg_train_loss_3.406.pt')), # dataset ablation 2
        6: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/sweepy-sweep-3/epoch360_avg_train_loss_3.133.pt')), # dataset ablation 3
        7: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/twilight-sweep-4/epoch350_avg_train_loss_2.846.pt')), # dataset ablation 4
        8: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/brisk-sweep-5/epoch330_avg_train_loss_2.776.pt')), # dataset ablation 5
        9: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/swift-sweep-6/epoch280_avg_train_loss_2.663.pt')), # dataset ablation 6
        10: (VisionTransformer, None, VTConfig(path='../Checkpoints/zxnrq5bw/olive-sweep-7/epoch240_avg_train_loss_2.862.pt')), # dataset ablation 7
        11: (VisionTransformer, None, VTConfig(path='../Checkpoints/ser3p8ns/magic-sweep-1/epoch60_avg_train_loss_2.283.pt')), # video in encoder
        12: (VisionTransformer, VTConfig(path='../Checkpoints/bqnl4hih/upbeat-sweep-1/epoch260_avg_train_loss_1.747.pt'), VTConfig(path='../Checkpoints/bqnl4hih/dainty-sweep-2/epoch250_avg_train_loss_1.795.pt')), # Temp Learnable, 65
        13: (VisionTransformer, VTConfig(path='../Checkpoints/3444gigx/major-sweep-1/epoch250_avg_train_loss_1.435.pt'), VTConfig(path='../Checkpoints/3444gigx/sandy-sweep-2/epoch270_avg_train_loss_1.075.pt')), # Temp 1, 65
        14: (VisionTransformer, VTConfig(path='../Checkpoints/m0npleep/daily-sweep-1/epoch260_avg_train_loss_2.408.pt'), VTConfig(path='../Checkpoints/m0npleep/young-sweep-2/epoch260_avg_train_loss_2.329.pt')), # Temp 5, 65
        15: (VisionTransformer, VTConfig(path='../Checkpoints/jird5gbi/super-sweep-1/epoch270_avg_train_loss_2.778.pt'), VTConfig(path='../Checkpoints/jird5gbi/fresh-sweep-2/epoch240_avg_train_loss_2.753.pt')), # Temp 10, 65
        16: (VisionTransformer, VTConfig(path='../Checkpoints/hg21jjvu/tough-sweep-1/epoch280_avg_train_loss_1.298.pt', num_class_tokens=257, use_perceiver=True), VTConfig(path='../Checkpoints/hg21jjvu/desert-sweep-2/epoch110_avg_train_loss_5.499.pt', num_class_tokens=257, use_perceiver=True)), # 257, Temp Learnable
        17: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/breezy-sweep-1/epoch270_avg_train_loss_2.656.pt', width=256), None), # batch 256
        18: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/generous-sweep-2/epoch50_avg_train_loss_5.551.pt', width=512), None), # batch 256
        19: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/icy-sweep-3/epoch80_avg_train_loss_5.578.pt', width=1024), None), # batch 256
        20: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/lemon-sweep-4/epoch450_avg_train_loss_4.101.pt', width=256), None), # batch 512
        21: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/fine-sweep-5/epoch90_avg_train_loss_6.114.pt', width=512), None), # batch 512
        22: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/tough-sweep-6/epoch50_avg_train_loss_6.231.pt', width=1024), None), # batch 512
        23: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/divine-sweep-7/epoch120_avg_train_loss_6.988.pt', width=256), None), # batch 1024
        24: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/generous-sweep-8/epoch90_avg_train_loss_6.988.pt', width=512), None), # batch 1024
        25: (VisionTransformer, VTConfig(path='../Checkpoints/uaq2a9ie/sweepy-sweep-9/epoch180_avg_train_loss_7.084.pt', width=1024), None), # batch 1024
       }
        
# [, , , , , , , , ]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_number', type=int, default=0)
    args = parser.parse_args()
    model_class, *configs = runs[args.run_number]
    
    for i, config in enumerate(configs):
        #if i==0: continue
        if config is None: continue
        model = model_class(**config.__dict__)
        
        root_dirs = ['../data/acc_watch_clip', '../data/acc_phone_clip', '../data/gyro_clip', '../data/orientation_clip']
        video_root_dir = '../data/video'
        path = './saved_weights/dataset_embeddings_with_perceiver_pooled.pkl' if config.use_perceiver else './saved_weights/dataset_perceiver_embeddings_no_pooling.pkl'

        dataset = load_all_data(root_dirs, video_root_dir, path=path)
        train_dataloader, test_dataloader, label_mapping = prepare_dataloader_precomputed_perceiver_embeddings(dataset, batch_size=1000, num_workers=4, drop_last=True)

        if not config.combined_encoders:
            checkpoint = torch.load(config.path)
            state_dict = checkpoint['model_state_dict']
            if 'loss_mapper' in state_dict: # resolve size conflicts
                del state_dict['loss_mapper']
            if args.run_number != 1: # modified how model was structure for later runs
                state_dict = {key[2:]: value for key, value in state_dict.items()} # remove 0. from key name
            model.load_state_dict(state_dict, strict=False)

        metrics = get_metrics(model, test_dataloader, torch.nn.CrossEntropyLoss(), device, config, sequence_wise=True)
        torch.cuda.empty_cache()
        print(metrics)