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
from data import load_all_data, prepare_dataloader, prepare_dataloader_precomputed_perceiver_embeddings, prepare_dataloader_precomputed_projected_embeddings
from utils import compute_label_embeddings
import argparse
import yaml
import random
import math
#from torchinfo import summary

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


def compute_loss(use_supervised_loss, use_contrastive_loss, logits_per_imu, logits_per_video, pred, y, lossfcn, sequence_wise, epoch, imu_features, video_features, model, config):
    if use_supervised_loss:
        supervised_loss = lossfcn(pred[0], y.repeat(1))
    else:
        supervised_loss = torch.tensor(0)

    window_size = 2
    epoch_interval = 1
    if epoch is not None:
        window_idx = max(((epoch%(257*epoch_interval))//epoch_interval-window_size+1),0)
        
    if use_contrastive_loss == 'infonce' or use_contrastive_loss == True:
        contrastive_losses = []
        
        for imu_virtual_batch, video_virtual_batch in zip(logits_per_imu,logits_per_video):
            if sequence_wise:
                for i, (imu_sequence_logits, video_sequence_logits) in enumerate(zip(imu_virtual_batch, video_virtual_batch)):
                #    #if random.random() < 0.03: # execute 3% of the time
                    if epoch==None or True:
                        contrastive_labels = torch.arange(len(imu_sequence_logits)).to(imu_sequence_logits.device) # virtual batch size can sometimes be < expected due to end of batch splits
                        contrastive_loss = (lossfcn(imu_sequence_logits, contrastive_labels) + lossfcn(video_sequence_logits, contrastive_labels)) / 2
                        contrastive_losses.append(contrastive_loss)    
            else:
                contrastive_labels = torch.arange(len(imu_virtual_batch)).to(imu_virtual_batch.device) # virtual batch size can sometimes be < expected due to end of batch splits
                contrastive_loss = (lossfcn(imu_virtual_batch, contrastive_labels) + lossfcn(video_virtual_batch, contrastive_labels)) / 2
                contrastive_losses.append(contrastive_loss)
        
        contrastive_losses = torch.hstack(contrastive_losses)
        loss_remapper = torch.nn.functional.softmax(model.imu_encoders[0].loss_mapper/model.imu_encoders[0].temp, dim=0) # model.imu_encoders[0].temp
        contrastive_losses = [contrastive_losses @ loss_remapper]
            
    elif use_contrastive_loss == 'mse':
        imu_features = imu_features[2]
        video_features = video_features[2] # (B, 257, C)
        
        contrastive_losses = []
        
        if sequence_wise:
            if epoch is not None:
                rows = list(range(window_idx, window_idx+window_size))
                rows = (rows + [0]) if 0 not in rows else rows
            else:
                rows = list(range(0,257))
                
            contrastive_loss = lossfcn(imu_features, video_features)
            contrastive_loss = torch.mean(contrastive_loss, dim=(0,2))[rows]
            for loss in contrastive_loss:
                contrastive_losses.append(loss)               
        else:
            contrastive_loss = lossfcn(imu_features, video_features)
            contrastive_loss = torch.mean(contrastive_loss)
            contrastive_losses.append(contrastive_loss)
        
        for i in range(len(contrastive_losses)):
            contrastive_losses[i] = contrastive_losses[i] / len(contrastive_losses) # we backpropogate on each loss term, so we must normalize by the total number of loss terms so we dont backprop a ton of huge gradients
    else:
        contrastive_losses = [torch.tensor(0)]
        
    loss = sum(contrastive_losses) + supervised_loss

    return loss, contrastive_losses, supervised_loss

def compute_CCA(imu_features: torch.Tensor, video_features: torch.Tensor, n_components) -> float:
    # assert features are of shape (B, C)
    assert imu_features.shape == video_features.shape
    assert len(imu_features.shape) == 2
    
    cca = CCA(n_components=n_components) # TODO: What is the best number of components?
    cca.fit(imu_features, video_features)
    
    # Average canonical correlations
    avg_corr = np.mean(cca.score(imu_features, video_features))
    
    return avg_corr

def compute_avg_cosine_similarity(imu_logits, video_logits, temperature, epoch):
    window_size = 2
    epoch_interval = 2
    if False: #epoch is not None:
        window_idx = max(((epoch%(257*epoch_interval))//epoch_interval-window_size+1),0)
        rows = list(range(window_idx, window_idx+window_size))
        rows = (rows + [0]) if 0 not in rows else rows
    else:
        rows = list(range(0,imu_logits.shape[0]))
    # Get the diagonal elements
    if len(imu_logits.shape) == 3:
        imu_diag = torch.diagonal(imu_logits, dim1=1, dim2=2)
        imu_diag = imu_diag[rows]
    else:
        imu_diag = torch.diagonal(imu_logits, dim1=0, dim2=1)

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

def compute_retrieval_metrics(query_embeddings, reference_embeddings, k):
    """
    Computes retrieval metrics such as Precision@k, Recall@k, and Mean Reciprocal Rank (MRR).
    Args:
        query_embeddings (torch.Tensor): Embeddings for the query items.
        reference_embeddings (torch.Tensor): Embeddings for the reference items.
        k (int): The number of top results to consider for computing the metrics.
    Returns:
        tuple: Recall@k
    """
    if len(query_embeddings.shape) == 3:
        query_embeddings = query_embeddings.permute(1,0,2) # (257, B, C)
        reference_embeddings = reference_embeddings.permute(1,2,0) # (257, C, B)
        dim = 2
        # Assuming the diagonal entries in the similarity matrix correspond to correct matches
        correct_matches = torch.arange(query_embeddings.size(1)).unsqueeze(0).expand(query_embeddings.size(0), -1).unsqueeze(2).to(query_embeddings.device)
    else:
        reference_embeddings = reference_embeddings.T
        dim = 1
        # Assuming the diagonal entries in the similarity matrix correspond to correct matches
        correct_matches = torch.arange(query_embeddings.size(0)).unsqueeze(1).to(query_embeddings.device)
        
    reference_embeddings = reference_embeddings.to(query_embeddings.dtype)
    
    # Calculate similarity matrix
    #print(query_embeddings.shape,reference_embeddings.shape)
    similarity = torch.matmul(query_embeddings, reference_embeddings)
    
    # Sort the results by similarity for each query
    top_k_values, top_k_indices = torch.topk(similarity, k, dim=dim)
    
    # Calculate Recall@k: Proportion of relevant items found in top k queries
    recall_at_k = (top_k_indices == correct_matches).any(dim=dim).float().mean().item()
    
    return recall_at_k

def compute_zero_shot_classification(query_embeddings, class_embeddings, y):
    """
    Computes zero-shot classification accuracy.
    Args:
        query_embeddings (torch.Tensor): Embeddings for the query items.
        class_embeddings (torch.Tensor): Embeddings for the class text prompts.
        y (torch.Tensor): integer label, assumes label mapping from class -> int is same as order of class_embeddings, which it should be
    Returns:
        float: Zero-shot classification accuracy.
    """
    # Calculate similarity matrix
    class_embeddings = class_embeddings.to(y.device)
    similarity = torch.matmul(query_embeddings, class_embeddings[y].T)
    
    # Get the predicted class indices
    predicted_classes = torch.argmax(similarity, dim=1)
    
    # Calculate accuracy
    accuracy = (predicted_classes == y).float().mean().item()
    
    return accuracy       

def compute_metrics(logits_per_imu, logits_per_video, imu_features, video_features, metrics_on_perceiver, temperature, k, text_embeds, y, use_perceiver_on_video_only=False, current_epoch=None, config=None, evaluate=False):
    # Which features to use for metrics
    #print(imu_features[0].shape,imu_features[2].shape)
    index = 1 if use_perceiver_on_video_only or use_perceiver else 2
    imu_features = imu_features[2][:,config.token_num,:] # get not perceiver embeds
    video_features = video_features[index][:,config.token_num,:]
    
    if not evaluate: 
        cosine_similarity = []
        for imu_virtual_batch, video_virtual_batch in zip(logits_per_imu, logits_per_video):
            cosine_similarity.append(compute_avg_cosine_similarity(imu_virtual_batch[config.token_num,:,:], video_virtual_batch[config.token_num,:,:], temperature, current_epoch))
        avg_cosine_similarity = sum(cosine_similarity)/len(cosine_similarity)
    else:
        avg_cosine_similarity = None

    if isinstance(k,int):
        k = [k]
    recall = {}
    for k_val in k:
        recall[k_val] = compute_retrieval_metrics(imu_features,video_features,k_val)
        
    acc = 0 #compute_zero_shot_classification(imu_features, text_embeds, y)
    
    return avg_cosine_similarity, recall, acc

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, lossfcn, device, config, text_embeds, current_epoch):
    model.train()
    epoch_train_loss = torch.tensor(0.0).to(device)
    epoch_train_contrastive_loss = torch.tensor(0.0).to(device)
    epoch_train_supervised_loss = torch.tensor(0.0).to(device)

    # Initialize lists to store batch-wise metrics for averaging later
    cosine_similarity_trains = []
    recall_trains = {f"Train Recall at {k_val}": [] for k_val in config.k}
    acc_trains = []

    for i, (imu, video_class, video_perceiver, y) in enumerate(dataloader):
        imu, video_class, video_perceiver, y = imu.to(device), video_class.to(device), video_perceiver.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits_per_imu, logits_per_video, pred, imu_features, video_features = model(imu=imu, 
                                                                                            video_class=video_class,
                                                                                            video_perceiver=video_perceiver,
                                                                                            use_perceiver=config.use_perceiver, 
                                                                                            supervised_on_perceiver=config.supervised_on_perceiver,
                                                                                            use_perceiver_on_video_only=config.use_perceiver_on_video_only,
                                                                                            metrics_on_perceiver=config.metrics_on_perceiver,
                                                                                            virtual_batch_size=config.virtual_batch_size,
                                                                                            sequence_wise=config.contrast_on_sequence)
            loss, contrastive_loss, supervised_loss = compute_loss(config.use_supervised_loss, 
                                                      config.use_contrastive_loss, 
                                                      logits_per_imu, 
                                                      logits_per_video, 
                                                      pred, 
                                                      y, 
                                                      lossfcn,
                                                      sequence_wise=config.contrast_on_sequence,
                                                      epoch=current_epoch,
                                                      imu_features=imu_features,
                                                      video_features=video_features,
                                                      model=model,
                                                      config=config)

        for j, contrastive_loss_virtual_batch in enumerate(contrastive_loss):
            save_graph = True if j < (len(contrastive_loss) - 1) or config.use_supervised_loss else False
            contrastive_loss_virtual_batch.backward(retain_graph=save_graph)
        if config.use_supervised_loss:
            supervised_loss.backward(retain_graph=False)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        # Calculate and store additional metrics for logging at the end of epoch
        cosine_similarity_train, recall_train, acc_train = compute_metrics(logits_per_imu, 
                                                                            logits_per_video, 
                                                                            imu_features, 
                                                                            video_features,
                                                                            config.metrics_on_perceiver,
                                                                            temperature=model.logit_scale,
                                                                            k=config.k,
                                                                            text_embeds=text_embeds,
                                                                            y=y,
                                                                            use_perceiver_on_video_only=config.use_perceiver_on_video_only, 
                                                                            current_epoch=None,
                                                                            config=config)
        
        cosine_similarity_trains.append(cosine_similarity_train.cpu().item())
        for k_val, recall_value in recall_train.items():
            recall_trains[f"Train Recall at {k_val}"].append(recall_value)
        acc_trains.append(acc_train)

        #print(loss, loss.shape, loss.squeeze().shape)
        epoch_train_loss += loss.squeeze().detach()
        epoch_train_contrastive_loss += sum(contrastive_loss).squeeze().detach()
        epoch_train_supervised_loss += supervised_loss.detach()
        
    # Calculate average of the metrics
    avg_cosine_similarity = sum(cosine_similarity_trains) / len(cosine_similarity_trains)
    avg_acc = sum(acc_trains) / len(acc_trains)
    recall_train_averaged = {key: sum(value) / len(value) for key, value in recall_trains.items()}
    
    # Log the metrics at the end of the epoch
    dict_to_log = {
        "Train Loss": (epoch_train_loss / len(dataloader)).cpu().item(),
        "Train Contrastive Loss": (epoch_train_contrastive_loss / len(dataloader)).cpu().item(),
        "Train Supervised Loss": (epoch_train_supervised_loss / len(dataloader)).cpu().item(),
        "Train Cosine Similarity": avg_cosine_similarity,
        "Train Zero-Shot Accuracy": avg_acc
    }
    dict_to_log.update(recall_train_averaged)
    wandb.log(dict_to_log)

    return (epoch_train_loss / len(dataloader)).cpu().item()


def validate(model, dataloader, lossfcn, device, config, text_embeds, current_epoch):
    model.eval()
    epoch_test_loss = torch.tensor(0.0).to(device)
    epoch_test_contrastive_loss = torch.tensor(0.0).to(device)
    epoch_test_supervised_loss = torch.tensor(0.0).to(device)

    # Initialize lists to store batch-wise metrics for averaging later
    cosine_similarity_tests = []
    recall_tests = {f"Test Recall at {k_val}": [] for k_val in config.k}
    acc_tests = []

    with torch.no_grad():
        for i, (imu, video_class, video_perceiver, y) in enumerate(dataloader):
            imu, video_class, video_perceiver, y = imu.to(device), video_class.to(device), video_perceiver.to(device), y.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits_per_imu, logits_per_video, pred, imu_features, video_features = model(imu=imu, 
                                                                                             video_class=video_class,
                                                                                             video_perceiver=video_perceiver,
                                                                                             use_perceiver=config.use_perceiver, 
                                                                                             supervised_on_perceiver=config.supervised_on_perceiver,
                                                                                             use_perceiver_on_video_only=config.use_perceiver_on_video_only,
                                                                                             metrics_on_perceiver=config.metrics_on_perceiver,
                                                                                             virtual_batch_size=config.virtual_batch_size,
                                                                                             sequence_wise=config.contrast_on_sequence)
                loss, contrastive_loss, supervised_loss = compute_loss(config.use_supervised_loss, 
                                                          config.use_contrastive_loss, 
                                                          logits_per_imu, 
                                                          logits_per_video, 
                                                          pred, 
                                                          y, 
                                                          lossfcn,
                                                          sequence_wise=config.contrast_on_sequence,
                                                          epoch=current_epoch,
                                                          imu_features=imu_features,
                                                          video_features=video_features,
                                                          model=model,
                                                          config=config)

            # Calculate and log additional metrics for validation
            cosine_similarity_test, recall_test, acc_test = compute_metrics(logits_per_imu, 
                                                                              logits_per_video, 
                                                                              imu_features, 
                                                                              video_features,
                                                                              config.metrics_on_perceiver,
                                                                              temperature=model.logit_scale,
                                                                              k=config.k,
                                                                              text_embeds=text_embeds,
                                                                              y=y,
                                                                              use_perceiver_on_video_only=config.use_perceiver_on_video_only, 
                                                                              current_epoch=None,
                                                                              config=config)
            
            cosine_similarity_tests.append(cosine_similarity_test.cpu().item())
            for k_val, recall_value in recall_test.items():
                recall_tests[f"Test Recall at {k_val}"].append(recall_value)
            acc_tests.append(acc_test)

            epoch_test_loss += loss.squeeze().detach()
            epoch_test_contrastive_loss += sum(contrastive_loss).squeeze().detach()
            epoch_test_supervised_loss += supervised_loss.detach()
            
        recall_test_averaged = {key: sum(value) / len(value) for key, value in recall_tests.items()}
            
        dict_to_log = {
            "Test Loss": (epoch_test_loss / len(dataloader)).cpu().item(),
            "Test Contrastive Loss": (epoch_test_contrastive_loss / len(dataloader)).cpu().item(),
            "Test Supervised Loss": (epoch_test_supervised_loss / len(dataloader)).cpu().item(),
            "Test Cosine Similarity": sum(cosine_similarity_tests) / len(cosine_similarity_tests),
            "Test Zero-Shot Accuracy": sum(acc_tests) / len(acc_tests)
        }
        dict_to_log.update(recall_test_averaged)
        wandb.log(dict_to_log)

    return sum(cosine_similarity_tests) / len(cosine_similarity_tests), (epoch_test_loss / len(dataloader)).cpu().item()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta

    def __call__(self, val_cosim, val_loss):
        if math.isnan(val_loss):
            return True
        
        if self.best_score is None:
            self.best_score = (val_cosim, val_loss)
            return False

        if val_cosim < self.best_score[0] + self.delta and val_loss > self.best_score[1] - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            if val_cosim > self.best_score[0] or val_loss < self.best_score[1]:
                self.best_score = (val_cosim, val_loss)
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
    parser.add_argument('--use_perceiver_on_video_only', type=bool, default=False)
    parser.add_argument('--T_max', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--virtual_batch_size', type=int, default=16)
    parser.add_argument('--contrast_on_sequence', type=bool, default=False)
    parser.add_argument('--loss_remapping', type=bool, default=False)
    parser.add_argument('--num_encoders', type=int, default=1)
    parser.add_argument('--token_num', type=int, default=0)
    parser.add_argument('--num_batches', type=int, default=-1)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='../Checkpoints')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--early_stopping_delta', type=float, default=0.001)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--metrics_on_perceiver', type=bool, default=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_training_steps', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--root_dirs', type=str, default='["../data/acc_watch_clip", "../data/acc_phone_clip", "../data/gyro_clip", "../data/orientation_clip"]')
    parser.add_argument('--supervised_on_perceiver', type=bool, default=False)
    parser.add_argument('--projected_embeds', type=bool, default=False)
    parser.add_argument('--use_contrastive_loss', type=bool, default=True)
    parser.add_argument('--use_supervised_loss', type=bool, default=True)
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument('--video_root_dir', type=str, default='../data/video')
    parser.add_argument('--padding', type=str_or_int, default='same')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--config', type=str, required=False, help='Path to a specific config file')
    args = parser.parse_args()

    file_config = load_config(args.config) if args.config else {} # If loading config from yaml rather than args
    config = merge_configs(file_config, args)
    
    # Set k config parameter in train.py b/c of issues with lists in yaml
    config['k'] = [1, 5, 10]
    config['token_num'] = [i for i in range(256)]
    config['combined_encoders'] = False


    if config.get('use_wandb', True):
        wandb.init(project="Contrastive Learning Ablations", config=config)
    else:
        wandb.init(project="Contrastive Learning Ablations", config=config, mode="disabled")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dirs = ['../data/acc_watch_clip', '../data/acc_phone_clip', '../data/gyro_clip', '../data/orientation_clip']
    video_root_dir = '../data/video'
    
    if wandb.config.projected_embeds:
        path = './saved_weights/dataset_perceiver_embeddings_fixed_projected.pkl'
        projection=768
    else: 
        path = None
        projection=1024
        
    if wandb.config.contrast_on_sequence or True:
        path = './saved_weights/dataset_embeddings_with_perceiver_pooled.pkl'
        #path = './saved_weights/dataset_perceiver_embeddings_no_pooling.pkl'
        num_class_tokens = 257 #65 #257
    else:
        num_class_tokens = 1
    
    dataset = load_all_data(root_dirs, video_root_dir, path=path)
    torch.cuda.empty_cache()
    
    train_dataloader, test_dataloader, label_mapping = prepare_dataloader_precomputed_perceiver_embeddings(dataset, batch_size=wandb.config.batch_size, num_workers=wandb.config.num_workers, num_batches=wandb.config.num_batches)

    model = perceivingContrastive(width=wandb.config.width, patch_size=wandb.config.patch_size, padding=wandb.config.padding, dropout=wandb.config.dropout, projection=projection, num_class_tokens=num_class_tokens, loss_remapping=wandb.config.loss_remapping, num_encoders=wandb.config.num_encoders, config=wandb.config, num_layers=wandb.config.num_layers,num_heads=wandb.config.num_heads).to(device)
    print('Number of Model Parameters', sum([param.nelement() for param in model.parameters()]), '\n', 'Trainable params: ', sum([param.nelement() for param in model.parameters() if param.requires_grad]))
    #print(summary(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config.learning_rate)
    if wandb.config.use_contrastive_loss in ['infonce', True]:
        lossfcn = torch.nn.CrossEntropyLoss()
    else:
        lossfcn = torch.nn.MSELoss(reduction='none')
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*wandb.config.num_epochs, eta_min=1e-5)

    early_stopping = EarlyStopping(patience=wandb.config.early_stopping_patience, delta=wandb.config.early_stopping_delta)
    
    _, text_embeds = compute_label_embeddings()

    for epoch in range(wandb.config.num_epochs):
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, lr_scheduler, lossfcn, device, wandb.config, text_embeds, current_epoch=epoch)
        wandb.log({"Epoch": epoch, "Avg Train Loss": avg_train_loss})

        sweep_subdir = os.path.join(wandb.config.checkpoint_dir, wandb.run.sweep_id) if wandb.run and wandb.run.sweep_id else wandb.config.checkpoint_dir
        run_subdir = os.path.join(sweep_subdir, wandb.run.name) if wandb.run else sweep_subdir
        os.makedirs(run_subdir, exist_ok=True)

        checkpoint_filename = f"{run_subdir}/epoch{epoch}_avg_train_loss_{avg_train_loss:.3f}.pt"
    
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.imu_encoders.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_filename)

        avg_test_cosim, avg_test_loss = validate(model, test_dataloader, lossfcn, device, wandb.config, text_embeds, current_epoch=epoch)

        if early_stopping(avg_test_cosim,avg_test_loss):
            if epoch > 50:
                print("Early stopping triggered.")
                break

    wandb.finish()