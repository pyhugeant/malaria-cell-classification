import yaml
import torch
import pandas as pd
from pathlib import Path

from src.datasets.malaria_dataset import build_dataloaders
from src.models.resnet50_classifier import build_model


def main():
    # ===== config =====
    cfg = yaml.safe_load(open("configs/malaria_resnet50.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== dataloader =====
    _, _, test_loader = build_dataloaders(cfg)

    # ===== model =====
    checkpoint_path = Path(cfg["project"]["output_dir"]) / cfg["project"]["name"] / "checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                results.append({
                    "true_label": labels[i].item(),
                    "pred_label": preds[i].item(),
                    "prob_parasitized": probs[i][0].item(),
                    "prob_uninfected": probs[i][1].item(),
                })

    df = pd.DataFrame(results)

    save_path = Path(cfg["project"]["output_dir"]) / cfg["project"]["name"] / "test_predictions.csv"
    df.to_csv(save_path, index=False)

    print("Saved to:", save_path)


if __name__ == "__main__":
    main()
