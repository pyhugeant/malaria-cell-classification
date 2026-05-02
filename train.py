import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from src.datasets.malaria_dataset import build_dataloaders
from src.models.resnet50_classifier import build_model
from src.eval.classification_metrics import compute_classification_metrics


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    if cfg["project"]["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["project"]["device"])


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    metrics = compute_classification_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/malaria_resnet50.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    torch.manual_seed(cfg["project"]["seed"])

    device = get_device(cfg)
    print("Device:", device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    run_dir = Path(cfg["project"]["output_dir"]) / cfg["project"]["name"]
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{cfg['train']['epochs']}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }

        history.append(row)
        print(row)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "best_val_f1": best_val_f1,
                },
                ckpt_dir / "best.pt"
            )

            print("Saved best model.")
        else:
            patience_counter += 1

        if patience_counter >= cfg["train"]["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    print("\nTesting best model...")

    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device
    )

    with open(run_dir / "test_summary.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Final test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()

def plot_training_curves(history, save_path):
    import pandas as pd

    df = pd.DataFrame(history)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    # F1
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_f1"], label="train")
    plt.plot(df["epoch"], df["val_f1"], label="val")
    plt.title("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(model, loader, device, save_path):
    import numpy as np

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = np.zeros((2, 2), dtype=int)

    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Parasitized", "Uninfected"],
                yticklabels=["Parasitized", "Uninfected"])

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(save_path)
    plt.close()

# save cures
plot_training_curves(history, run_dir / "training_curve.png")

# save confusion matrix（test set）
plot_confusion_matrix(model, test_loader, device, run_dir / "confusion_matrix.png")
