import torch
from torch.utils.data import DataLoader
import wandb
from .utils import label_to_numeric, loadDataFromFile
from .model import perceivingContrastive
from .data import dictionaryFilepathLoader, collate_fn

def get_labels(imu, label_to_numeric, device):
    """Get labels from IMU data and convert them to numerical form."""
    labels = [label_to_numeric[imu[j][0].split('/')[-1][:-4].lower()] for j in range(len(imu))]
    return torch.tensor(labels, dtype=torch.long).to(device)

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, lossfcn, device):
    """Train the model for one epoch."""
    model.train()
    epoch_train_loss = 0
    for i, (video, imu, interp) in enumerate(dataloader):
        optimizer.zero_grad()
        labels = get_labels(imu, label_to_numeric, device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(video, imu, interp)
            loss = lossfcn(pred, labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        epoch_train_loss += loss.detach().cpu().item()
        wandb.log({"Step": i*(epoch+1), "Loss": loss.detach().cpu().item()})

    return epoch_train_loss / len(dataloader)

def validate(model, dataloader, lossfcn, device):
    """Validate the model."""
    model.eval()
    epoch_test_loss = 0
    with torch.no_grad():
        for video, imu, interp in dataloader:
            labels = get_labels(imu, label_to_numeric, device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(video, imu, interp)
                loss = lossfcn(pred, labels)

            epoch_test_loss += loss.detach().cpu().item()

    return epoch_test_loss / len(dataloader)

if __name__ == "__main__":
    # Initialize Weights and Biases
    config = {
        "patch": 16,
        "width": 1024,
        "heads": 8,
        "layers": 8,
        "window": 256,
        "batch": 512
    }
    wandb.init(project='Perceiver Resampler Contrastive Training', resume=False, config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data = loadDataFromFile()
    train_dataloader = DataLoader(dictionaryFilepathLoader(train_data), batch_size=512, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(dictionaryFilepathLoader(test_data), batch_size=512, shuffle=True, collate_fn=collate_fn)

    model = perceivingContrastive().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    lossfcn = torch.nn.CrossEntropyLoss()
    num_training_steps = len(train_dataloader)*5
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=num_training_steps,eta_min=1e-5)
    
    for epoch in range(100):
        avg_train_loss = train_one_epoch(model, train_dataloader, optim, lr_scheduler, lossfcn, device)
        wandb.log({"Epoch": epoch, "Avg Train Loss": avg_train_loss})

        # Checkpointing
        checkpoint_filename = f'./Checkpoints/LowPassFilter/WindowSizeFix/128/P16W1024H8L8/epoch{epoch}_avg_train_loss_{avg_train_loss:.3f}.pt'
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': avg_train_loss,
                }, checkpoint_filename)

        # Validation loop
        avg_test_loss = validate(model, test_dataloader, lossfcn, device)
        wandb.log({"Epoch": epoch, "Avg Test Loss": avg_test_loss})

    wandb.finish()
