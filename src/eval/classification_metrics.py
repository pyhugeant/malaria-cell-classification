import torch


def compute_classification_metrics(logits, labels, positive_label=0):
    preds = torch.argmax(logits, dim=1)

    correct = (preds == labels).sum().item()
    total = labels.numel()

    tp = ((preds == positive_label) & (labels == positive_label)).sum().item()
    tn = ((preds != positive_label) & (labels != positive_label)).sum().item()
    fp = ((preds == positive_label) & (labels != positive_label)).sum().item()
    fn = ((preds != positive_label) & (labels == positive_label)).sum().item()

    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    specificity = tn / max(tn + fp, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_label": positive_label,
        "positive_class": "Parasitized",
    }
