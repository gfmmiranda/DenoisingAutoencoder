import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import wandb
from src.utils import play_denoised_sample


def run_epoch(
        model, 
        loader, 
        criterion, 
        device, 
        optimizer=None, 
        scaler=None, 
        train=True
        ):
    epoch_loss = 0.0
    mode = 'Train' if train else 'Val'

    if train:
        model.train()
    else:
        model.eval()

    loop = tqdm(loader, desc=f"{mode} Epoch", leave=False)

    for noisy, clean in loop:
        noisy = noisy.unsqueeze(1).to(device)
        clean = clean.unsqueeze(1).to(device)

        with torch.set_grad_enabled(train):
            with autocast():
                output = model(noisy)
                loss = criterion(output, clean)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def train_model(
        model, 
        optimizer, 
        criterion,
        scheduler,
        device, 
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        num_epochs=10,
        audio_preview=False,
        early_stopping_patience=None,
        trial=None,
        save_dir="checkpoints",
        log_to_wandb=False
    ):

    train_losses = []
    val_losses = []
    scaler = GradScaler()
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        avg_train_loss = run_epoch(model, train_loader, criterion, device, optimizer, scaler, train=True)
        avg_val_loss = run_epoch(model, val_loader, criterion, device, train=False)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if early_stopping_patience is not None and early_stopping_patience > 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                model_save_path = os.path.join(save_dir, f"model_trial_{trial.number if trial else 'default'}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved: {model_save_path} (val loss: {best_val_loss:.4f})")

                if trial:
                    trial.set_user_attr("best_model_path", model_save_path)
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        if audio_preview and epoch % 10 == 0:
            print(f"\nPreviewing model output at epoch {epoch}:")
            play_denoised_sample(model, val_dataset, index=0)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']:.6f}")

        if log_to_wandb:
            # Log to wand
            wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            }, step=epoch)

            torch.cuda.empty_cache()

    return model, train_losses, val_losses, best_val_loss