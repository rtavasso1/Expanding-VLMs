import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the relative path to Otter/src from the current directory
sys.path.append(os.path.join(current_dir, "Otter", "src"))

import torch
import wandb
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import CCA
from model import perceivingContrastive
from data import load_all_data, prepare_dataloader, prepare_dataloader_precomputed_perceiver_embeddings
import argparse
import yaml

# Load the config from the file into a dictionary
def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def merge_configs(file_config, args_config):
    merged_config = file_config.copy()
    for key, value in vars(args_config).items():
        if value is not None:  # Only override values provided as command line args
            merged_config[key] = value
    return merged_config


def compute_loss(use_supervised_loss, use_contrastive_loss, logits_per_imu, logits_per_video, pred, y, lossfcn, contrastive_labels):
    if use_supervised_loss:
        supervised_loss = lossfcn(pred, y)
    else:
        supervised_loss = 0

    if use_contrastive_loss:
        contrastive_loss = (lossfcn(logits_per_imu, contrastive_labels) + lossfcn(logits_per_video, contrastive_labels)) / 2
    else:
        contrastive_loss = 0

    return supervised_loss + contrastive_loss

def compute_CCA(imu_features: torch.Tensor, video_features: torch.Tensor, n_components) -> float:
    # assert features are of shape (B, C)
    assert imu_features.shape == video_features.shape
    assert len(imu_features.shape) == 2
    
    cca = CCA(n_components=n_components) # TODO: What is the best number of components?
    cca.fit(imu_features, video_features)
    
    # Average canonical correlations
    avg_corr = np.mean(cca.score(imu_features, video_features))
    
    return avg_corr

def compute_avg_cosine_similarity(imu_logits, video_logits, temperature):
    # Get the diagonal elements
    imu_diag = torch.diagonal(imu_logits)
    
    # Compute the average cosine similarity
    avg_cosine_similarity = torch.mean(imu_diag)/temperature.exp()
    
    return avg_cosine_similarity

def compute_mutual_info(imu_features, video_features):
    B, C = imu_features.shape
    mutual_info_values = []

    for i in range(C):
        mi = mutual_info_regression(imu_features[:, i].reshape(-1, 1), video_features[:, i])
        mutual_info_values.append(mi[0])

    avg_mutual_info = sum(mutual_info_values) / len(mutual_info_values)
    return avg_mutual_info

def compute_metrics(logits_per_imu, logits_per_video, imu_features, video_features, metrics_on_perceiver, n_components, temperature):
    # Which features to use for metrics
    index = 1 if metrics_on_perceiver else 0
    imu_features = imu_features[index]
    video_features = video_features[index]

    avg_cosine_similarity = compute_avg_cosine_similarity(logits_per_imu, logits_per_video, temperature)
    avg_mutual_info = compute_mutual_info(imu_features.detach().cpu().numpy(), video_features.detach().cpu().numpy())
    avg_cca = compute_CCA(imu_features.detach().cpu().numpy(), video_features.detach().cpu().numpy(), n_components)
    return avg_cosine_similarity, avg_mutual_info, avg_cca

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, lossfcn, device, config):
    model.train()
    epoch_train_loss = torch.tensor(0.0).to(device)
    for i, (imu, video_class, video_perceiver, y) in enumerate(dataloader):
        #with torch.profiler.profile(profile_memory=True) as prof:
        imu, video_class, video_perceiver, y = imu.to(device), video_class.to(device), video_perceiver.to(device), y.to(device)
        optimizer.zero_grad()
        contrastive_labels = torch.arange(len(imu)).to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits_per_imu, logits_per_video, pred, imu_features, video_features = model(imu=imu, 
                                                                                            video_class=video_class,
                                                                                            video_perceiver=video_perceiver,
                                                                                            use_perceiver=config.use_perceiver, 
                                                                                            supervised_on_perceiver=config.supervised_on_perceiver)
            #print_memory("After Forward Pass")
            loss = compute_loss(config.use_supervised_loss, 
                                config.use_contrastive_loss, 
                                logits_per_imu, 
                                logits_per_video, 
                                pred, 
                                y, 
                                lossfcn, 
                                contrastive_labels)
            #print_memory("After Loss Pass")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Calculate and log additional metrics
        avg_cosine_similarity_train, avg_mutual_info_train, avg_cca_train = compute_metrics(logits_per_imu, 
                                                                                            logits_per_video, 
                                                                                            imu_features, 
                                                                                            video_features,
                                                                                            config.metrics_on_perceiver,
                                                                                            config.n_components,
                                                                                            temperature=model.logit_scale)

        wandb.log({
            "Train Loss": loss.cpu().item(),
            "Avg Cosine Similarity Train": avg_cosine_similarity_train.cpu().item(),
            "Avg Mutual Info Train": avg_mutual_info_train,
            "Avg CCA Train": avg_cca_train
        })

        #print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))

        epoch_train_loss += loss.detach()
        
    return (epoch_train_loss / len(dataloader)).cpu().item()

def validate(model, dataloader, lossfcn, device, config):
    model.eval()
    epoch_test_loss = torch.tensor(0.0).to(device)

    # Initialize lists to store batch-wise metrics for averaging later
    avg_cosine_similarity_vals = []
    avg_mutual_info_vals = []
    avg_cca_vals = []

    with torch.no_grad():
        for i, (imu, video_class, video_perceiver, y) in enumerate(dataloader):
            imu, video_class, video_perceiver, y = imu.to(device), video_class.to(device), video_perceiver.to(device), y.to(device)
            contrastive_labels = torch.arange(len(imu)).to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits_per_imu, logits_per_video, pred, imu_features, video_features = model(imu=imu, 
                                                                                             video_class=video_class,
                                                                                             video_perceiver=video_perceiver,
                                                                                             use_perceiver=config.use_perceiver, 
                                                                                             supervised_on_perceiver=config.supervised_on_perceiver)
                loss = compute_loss(config.use_supervised_loss, 
                                    config.use_contrastive_loss, 
                                    logits_per_imu, 
                                    logits_per_video, 
                                    pred, 
                                    y, 
                                    lossfcn, 
                                    contrastive_labels)

            # Calculate and log additional metrics for validation
            avg_cosine_similarity_val, avg_mutual_info_val, avg_cca_val = compute_metrics(logits_per_imu, 
                                                                                          logits_per_video, 
                                                                                          imu_features, 
                                                                                          video_features,
                                                                                          config.metrics_on_perceiver,
                                                                                          config.n_components,
                                                                                          temperature=model.logit_scale)
            
            avg_cosine_similarity_vals.append(avg_cosine_similarity_val.cpu().item())
            avg_mutual_info_vals.append(avg_mutual_info_val)
            avg_cca_vals.append(avg_cca_val)

            epoch_test_loss += loss.detach()
            
        wandb.log({
            "Avg Val Loss": (epoch_test_loss / len(dataloader)).cpu().item(),
            "Avg Cosine Similarity Val": sum(avg_cosine_similarity_vals) / len(avg_cosine_similarity_vals),
            "Avg Mutual Info Val": sum(avg_mutual_info_vals) / len(avg_mutual_info_vals),
            "Avg CCA Val": sum(avg_cca_vals) / len(avg_cca_vals),
        })

    return (epoch_test_loss / len(dataloader)).cpu().item()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_loss
            self.counter = 0
        return False

def str_or_int(value):
    try:
        return int(value)
    except ValueError:
        return value
    
def print_memory(label):
    print(f"{label}: Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB, Cached: {torch.cuda.memory_cached() / 1024 ** 2} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_perceiver', type=bool, default=True)
    parser.add_argument('--T_max', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_dir', type=str, default='../Checkpoints')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--early_stopping_delta', type=float, default=0.001)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--metrics_on_perceiver', type=bool, default=True)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_training_steps', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--root_dirs', type=str, default='["../data/acc_watch_clip", "../data/acc_phone_clip", "../data/gyro_clip", "../data/orientation_clip"]')
    parser.add_argument('--supervised_on_perceiver', type=bool, default=False)
    parser.add_argument('--use_contrastive_loss', type=bool, default=True)
    parser.add_argument('--use_supervised_loss', type=bool, default=True)
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument('--video_root_dir', type=str, default='../data/video')
    parser.add_argument('--padding', type=str_or_int, default='same')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--config', type=str, required=False, help='Path to a specific config file')
    args = parser.parse_args()

    file_config = load_config(args.config) if args.config else {}
    # wandb_config = {k: v for k, v in wandb.config.as_dict().items()} if wandb.run else {}

    config = merge_configs(file_config, args)
    
    #if config['pooled'] and config['metrics_on_perceiver']:
    #    print("Invalid configuration: pooled and metrics_on_perceiver can't both be True.")
    #    sys.exit(1)  # Exit the script


    if config.get('use_wandb', True):
        wandb.init(project="Contrastive Learning Ablations", config=config)
    else:
        wandb.init(project="Contrastive Learning Ablations", config=config, mode="disabled")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dirs = ['../data/acc_watch_clip', '../data/acc_phone_clip', '../data/gyro_clip', '../data/orientation_clip']
    video_root_dir = '../data/video'
    dataset = load_all_data(root_dirs, video_root_dir)
    torch.cuda.empty_cache()
    
    train_dataloader, test_dataloader, label_mapping = prepare_dataloader_precomputed_perceiver_embeddings(dataset, batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers)

    model = perceivingContrastive(patch_size=wandb.config.patch_size, padding=wandb.config.padding, dropout=wandb.config.dropout).to(device)
    print('Number of Model Parameters', sum([param.nelement() for param in model.parameters()]), '\n', 'Trainable params: ', sum([param.nelement() for param in model.parameters() if param.requires_grad]))
    print_memory("Before Forward Pass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    lossfcn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config.num_training_steps, eta_min=1e-5)

    early_stopping = EarlyStopping(patience=wandb.config.early_stopping_patience, delta=wandb.config.early_stopping_delta)

    for epoch in range(wandb.config.num_epochs):
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, lr_scheduler, lossfcn, device, wandb.config)
        wandb.log({"Epoch": epoch, "Avg Train Loss": avg_train_loss})

        checkpoint_filename = f"{wandb.config.checkpoint_dir}/epoch{epoch}_avg_train_loss_{avg_train_loss:.3f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.imu_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_filename)

        avg_test_loss = validate(model, test_dataloader, lossfcn, device, wandb.config)
        #wandb.log({"Epoch": epoch, "Avg Test Loss": avg_test_loss})

        if early_stopping(avg_test_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()