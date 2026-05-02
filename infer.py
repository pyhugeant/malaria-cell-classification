import argparse
from pathlib import Path

import yaml
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet50_classifier import build_model


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_image(model, image_path, transform, device, idx_to_class):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    pred_class = idx_to_class[pred_idx]
    confidence = probs[pred_idx].item()

    return pred_class, confidence, probs.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/malaria_resnet50.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # 根据你的 sanity check：
    # {'Parasitized': 0, 'Uninfected': 1}
    idx_to_class = {
        0: "Parasitized",
        1: "Uninfected",
    }

    transform = get_transform(cfg["data"]["image_size"])

    pred_class, confidence, probs = predict_image(
        model=model,
        image_path=args.image,
        transform=transform,
        device=device,
        idx_to_class=idx_to_class,
    )

    print("\nPrediction result")
    print("-----------------")
    print(f"Image: {args.image}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Probability Parasitized: {probs[0]:.4f}")
    print(f"Probability Uninfected: {probs[1]:.4f}")


if __name__ == "__main__":
    main()
