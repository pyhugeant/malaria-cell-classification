from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def get_transforms(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def build_dataloaders(cfg):
    root_dir = Path(cfg["data"]["root_dir"])
    image_size = cfg["data"]["image_size"]

    dataset = datasets.ImageFolder(
        root=root_dir,
        transform=get_transforms(image_size, train=True)
    )

    print("Class mapping:", dataset.class_to_idx)

    total = len(dataset)
    n_train = int(total * cfg["data"]["train_ratio"])
    n_val = int(total * cfg["data"]["val_ratio"])
    n_test = total - n_train - n_val

    generator = torch.Generator().manual_seed(cfg["project"]["seed"])

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator
    )

    val_set.dataset.transform = get_transforms(image_size, train=False)
    test_set.dataset.transform = get_transforms(image_size, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    return train_loader, val_loader, test_loader