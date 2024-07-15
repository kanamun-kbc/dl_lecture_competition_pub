import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import get_dataloaderscd
from src.models import EnhancedWaveNet
from src.utils import set_seed

@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    os.chdir(hydra.utils.get_original_cwd())
    
    logdir = os.path.join(hydra.utils.get_original_cwd(), "outputs", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    print(f"Data directory: {cfg.data_dir}")
    
    # デバッグフラグを設定
    debug = False  # 必要に応じて True に変更

    train_loader, val_loader, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, augment=True, debug=debug)

    num_channels = train_loader.dataset.num_channels

    model = EnhancedWaveNet(cfg.num_classes, num_subjects=10, num_channels=num_channels).to(cfg.device)
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    if cfg.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min')
    elif cfg.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=cfg.learning_rate, steps_per_epoch=len(train_loader), epochs=cfg.epochs)

    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=1854, top_k=10).to(cfg.device)
    early_stopping_counter = 0
    patience = 5

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")
        model.train()
        train_loss, train_acc = [], []

        for X, y, subject_idx in tqdm(train_loader):
            X, y, subject_idx = X.to(cfg.device), y.to(cfg.device), subject_idx.to(cfg.device)
            optimizer.zero_grad()
            y_pred = model(X, subject_idx)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(accuracy(y_pred, y).item())

        model.eval()
        val_loss, val_acc = [], []

        for X, y, subject_idx in tqdm(val_loader):
            X, y, subject_idx = X.to(cfg.device), y.to(cfg.device), subject_idx.to(cfg.device)
            with torch.no_grad():
                y_pred = model(X, subject_idx)
                val_loss.append(F.cross_entropy(y_pred, y).item())
                val_acc.append(accuracy(y_pred, y).item())

        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_val_loss = sum(val_loss) / len(val_loss)
        mean_train_acc = sum(train_acc) / len(train_acc)
        mean_val_acc = sum(val_acc) / len(val_acc)

        if cfg.scheduler == 'plateau':
            scheduler.step(mean_val_loss)
        else:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{cfg.epochs} | train loss: {mean_train_loss:.4f} | val loss: {mean_val_loss:.4f} | train acc: {mean_train_acc:.4f} | val acc: {mean_val_acc:.4f}")

        if mean_val_acc > max_val_acc:
            max_val_acc = mean_val_acc
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
            cprint("New best model saved!", "green")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                cprint("Early stopping triggered!", "red")
                break

    model.load_state_dict(torch.load(os.path.join(logdir, 'best_model.pth')))
    model.eval()
    preds = []

    for X, subject_idx in tqdm(test_loader):
        with torch.no_grad():
            preds.append(model(X.to(cfg.device), subject_idx.to(cfg.device)).cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, 'test_preds.npy'), preds)
    cprint(f"Test predictions saved to {logdir}", "cyan")

if __name__ == "__main__":
    main()
