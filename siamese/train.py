import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

import copy
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Create a writer with all default settings





# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def create_writer(log_dir: str,
                  experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join(log_dir, timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join(log_dir, timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)




def train_step(model, dataloader, loss_fn, optimizer, device):
    """One training epoch for Siamese model."""
    model.train()
    epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

    for img0, img1, label in dataloader:
        img0, img1, label = img0.to(device), img1.to(device), label.to(device).float().unsqueeze(1)

        # Forward
        output = model(img0, img1)  # logits
        loss = loss_fn(output, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = (torch.sigmoid(output) > 0.5).float()
        correct = preds.eq(label).sum().item()

        batch_size = label.size(0)
        epoch_loss += loss.item() * batch_size
        epoch_correct += correct
        epoch_total += batch_size

    return epoch_loss / epoch_total, epoch_correct / epoch_total


def val_step(model, dataloader, loss_fn, device):
    """One validation epoch for Siamese model."""
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for img0, img1, label in dataloader:
            img0, img1, label = img0.to(device), img1.to(device), label.to(device).float().unsqueeze(1)

            output = model(img0, img1)
            loss = loss_fn(output, label)

            preds = (torch.sigmoid(output) > 0.5).float()
            correct = preds.eq(label).sum().item()

            batch_size = label.size(0)
            val_loss += loss.item() * batch_size
            val_correct += correct
            val_total += batch_size

    return val_loss / val_total, val_correct / val_total


def train(model, train_loader, valid_loader, optimizer, loss_fn, device,
          epochs=100, patience=10, writer=None):
    """
    Train Siamese model with BCE loss, validation, early stopping, and optional TensorBoard logging.
    """
    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = val_step(model, valid_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] --> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_step=epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, global_step=epoch)
            writer.close()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            print("  ✅ New best model saved")
        else:
            early_stop_counter += 1
            print(f"  ⚠️ No improvement. Early stop counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("⏹️ Early stopping triggered!")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)


    return history, model


