import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cell_state_classifier import CellStateCNN

def build_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return train_tf, eval_tf


def build_loaders(data_root, img_size, batch_size, num_workers):
    train_tf, eval_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def compute_class_weights_from_dataset(dataset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float)
    for _, label in dataset.samples:
        counts[label] += 1.0
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return counts, weights


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            correct, count = accuracy_from_logits(logits, labels)
            total_correct += correct
            total_count += count

            preds = logits.argmax(dim=1).cpu()
            targets = labels.cpu()
            for t, p in zip(targets, preds):
                conf[t, p] += 1

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc, conf


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct, count = accuracy_from_logits(logits, labels)
        total_correct += correct
        total_count += count

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train a cell-state CNN classifier.")
    parser.add_argument("--data-root", default="data/cnn_cells/labeled", help="Dataset root with train/val/test")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=96, help="Square resize for images")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--out-dir", default="model/runs/cnn", help="Output directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        data_root, args.img_size, args.batch_size, args.num_workers
    )

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    print(f"num_classes={num_classes}")
    print(f"class_to_idx={class_to_idx}")

    model = CellStateCNN(num_classes=num_classes).to(device)
    class_counts, class_weights = compute_class_weights_from_dataset(train_ds, num_classes)
    print(f"class_counts={class_counts.tolist()}")
    print(f"class_weights={class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device, num_classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "epoch": epoch,
                "val_acc": val_acc,
            }
            torch.save(ckpt, out_dir / "best.pt")

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_acc, test_conf = evaluate(model, test_loader, criterion, device, num_classes)

    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "class_to_idx.json").open("w") as f:
        json.dump(class_to_idx, f, indent=2)
    with (out_dir / "idx_to_class.json").open("w") as f:
        json.dump(idx_to_class, f, indent=2)
    with (out_dir / "test_confusion_matrix.json").open("w") as f:
        json.dump(test_conf.tolist(), f)

    print(f"saved={out_dir}")


if __name__ == "__main__":
    main()
