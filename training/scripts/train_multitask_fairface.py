import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
AGE_MIDPOINTS = [1.0, 6.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
GENDER_CLASSES = ["Male", "Female"]
BACKBONE_OPTIONS = {"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"}


@dataclass
class Sample:
    image_path: Path
    age_idx: int
    gender_idx: int
    age_value: float
    race: str
    source_name: str


class FaceDataset(Dataset):
    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        return (
            image,
            torch.tensor(sample.age_idx, dtype=torch.long),
            torch.tensor(sample.gender_idx, dtype=torch.long),
            torch.tensor(sample.age_value, dtype=torch.float32),
        )


class MultiTaskMobileNet(nn.Module):
    def __init__(self, pretrained: bool = True, backbone_name: str = "mobilenet_v3_large", dropout: float = 0.3):
        super().__init__()
        if backbone_name not in BACKBONE_OPTIONS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        self.backbone_name = backbone_name
        self.backbone = backbone
        self.neck = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
        )
        self.age_head = nn.Linear(256, len(AGE_CLASSES))
        self.gender_head = nn.Linear(256, len(GENDER_CLASSES))

    def forward(self, x):
        features = self.backbone(x)
        fused = self.neck(features)
        return self.age_head(fused), self.gender_head(fused)


def build_model_from_config(config: dict, pretrained: bool | None = None):
    use_pretrained = bool(config["use_pretrained"]) if pretrained is None else pretrained
    return MultiTaskMobileNet(
        pretrained=use_pretrained,
        backbone_name=str(config.get("backbone", "mobilenet_v3_large")),
        dropout=float(config.get("dropout", 0.3)),
    )


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_sources(config: dict) -> list[dict]:
    if "data_sources" in config:
        return config["data_sources"]
    return [
        {
            "name": "fairface",
            "dataset_root": config["dataset_root"],
            "train_csv": config["train_csv"],
            "val_csv": config["val_csv"],
            "source_weight": 1.0,
        }
    ]


def read_samples(csv_path: Path, dataset_root: Path, limit: int | None, source_name: str):
    rows = []
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = dataset_root / row["file"]
            if not image_path.exists():
                continue
            age_idx = AGE_CLASSES.index(row["age"])
            rows.append(
                Sample(
                    image_path=image_path,
                    age_idx=age_idx,
                    gender_idx=GENDER_CLASSES.index(row["gender"]),
                    age_value=AGE_MIDPOINTS[age_idx],
                    race=row.get("race", "Unknown"),
                    source_name=source_name,
                )
            )
    if limit:
        rows = rows[:limit]
    return rows


def build_age_distribution(age_idx: torch.Tensor, sigma: float, device: torch.device):
    centers = torch.tensor(AGE_MIDPOINTS, device=device, dtype=torch.float32)
    targets = centers[age_idx].unsqueeze(1)
    distances = (centers.unsqueeze(0) - targets) / sigma
    distribution = torch.softmax(-(distances**2), dim=1)
    return distribution


def classification_accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def one_off_accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return ((predictions - targets).abs() <= 1).float().mean().item()


def expected_age(logits):
    probabilities = torch.softmax(logits, dim=1)
    centers = torch.tensor(AGE_MIDPOINTS, device=logits.device, dtype=torch.float32)
    return (probabilities * centers.unsqueeze(0)).sum(dim=1)


def mae_years(logits, age_values):
    predictions = expected_age(logits)
    return (predictions - age_values).abs().mean().item()


def cs_tolerance(logits, age_values, tolerance: float):
    predictions = expected_age(logits)
    return ((predictions - age_values).abs() <= tolerance).float().mean().item()


def distribution_loss(logits, age_idx, sigma, age_ce_weight, age_kl_weight):
    ce_loss = F.cross_entropy(logits, age_idx)
    target_dist = build_age_distribution(age_idx, sigma=sigma, device=logits.device)
    log_probs = F.log_softmax(logits, dim=1)
    kl_loss = F.kl_div(log_probs, target_dist, reduction="batchmean")
    return age_ce_weight * ce_loss + age_kl_weight * kl_loss


def regression_consistency_loss(logits, age_values):
    predicted_age = expected_age(logits)
    return F.l1_loss(predicted_age, age_values)


def distillation_loss(student_logits, teacher_logits, temperature: float):
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    target_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean") * (temperature**2)


def run_epoch(
    model,
    loader,
    device,
    optimizer,
    gender_criterion,
    age_sigma,
    age_ce_weight,
    age_kl_weight,
    age_reg_weight,
    gender_weight,
    teacher_model,
    distill_temperature,
    distill_age_weight,
    distill_gender_weight,
    train: bool,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_age_acc = 0.0
    total_age_one_off = 0.0
    total_gender_acc = 0.0
    total_age_mae = 0.0
    total_cs5 = 0.0
    steps = 0

    for images, age_targets, gender_targets, age_values in loader:
        images = images.to(device, non_blocking=True)
        age_targets = age_targets.to(device, non_blocking=True)
        gender_targets = gender_targets.to(device, non_blocking=True)
        age_values = age_values.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            age_logits, gender_logits = model(images)
            age_loss = distribution_loss(
                age_logits,
                age_targets,
                sigma=age_sigma,
                age_ce_weight=age_ce_weight,
                age_kl_weight=age_kl_weight,
            )
            reg_loss = regression_consistency_loss(age_logits, age_values)
            gender_loss = gender_criterion(gender_logits, gender_targets)
            loss = age_loss + age_reg_weight * reg_loss + gender_weight * gender_loss

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_age_logits, teacher_gender_logits = teacher_model(images)
                age_distill = distillation_loss(age_logits, teacher_age_logits, temperature=distill_temperature)
                gender_distill = distillation_loss(gender_logits, teacher_gender_logits, temperature=distill_temperature)
                loss = loss + distill_age_weight * age_distill + distill_gender_weight * gender_distill

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item()
        total_age_acc += classification_accuracy(age_logits, age_targets)
        total_age_one_off += one_off_accuracy(age_logits, age_targets)
        total_gender_acc += classification_accuracy(gender_logits, gender_targets)
        total_age_mae += mae_years(age_logits, age_values)
        total_cs5 += cs_tolerance(age_logits, age_values, tolerance=5.0)
        steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "age_acc": total_age_acc / max(steps, 1),
        "age_one_off_acc": total_age_one_off / max(steps, 1),
        "gender_acc": total_gender_acc / max(steps, 1),
        "age_mae_proxy": total_age_mae / max(steps, 1),
        "age_cs5_proxy": total_cs5 / max(steps, 1),
    }


def save_checkpoint(path: Path, model, config, metrics):
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "metrics": metrics,
        "age_classes": AGE_CLASSES,
        "age_midpoints": AGE_MIDPOINTS,
        "gender_classes": GENDER_CLASSES,
    }
    torch.save(payload, path)


def make_sampler(samples: List[Sample], race_boosts: dict, source_boosts: dict, gender_boosts: dict):
    counts = [0] * len(AGE_CLASSES)
    race_counts = {}
    source_counts = {}
    for sample in samples:
        counts[sample.age_idx] += 1
        race_counts[sample.race] = race_counts.get(sample.race, 0) + 1
        source_counts[sample.source_name] = source_counts.get(sample.source_name, 0) + 1

    weights = []
    for sample in samples:
        age_count = max(counts[sample.age_idx], 1)
        base_weight = 1.0 / math.sqrt(age_count)
        race_weight = float(race_boosts.get(sample.race, 1.0))
        source_weight = float(source_boosts.get(sample.source_name, 1.0))
        gender_weight = float(gender_boosts.get(GENDER_CLASSES[sample.gender_idx], 1.0))
        weights.append(base_weight * race_weight * source_weight * gender_weight)

    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
    return sampler, counts, race_counts, source_counts


def build_transforms(image_size: int):
    transform_train = transforms.Compose([
        transforms.Resize((image_size + 12, image_size + 12)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)],
            p=0.7,
        ),
        transforms.RandomGrayscale(p=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.01, 0.06), ratio=(0.3, 3.0)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_val


def load_all_samples(config: dict, split: str):
    samples = []
    sources = resolve_sources(config)
    for source in sources:
        limit_key = "max_train_samples" if split == "train" else "max_val_samples"
        csv_key = "train_csv" if split == "train" else "val_csv"
        limit = source.get(limit_key, config.get(limit_key))
        source_samples = read_samples(
            csv_path=Path(source[csv_key]),
            dataset_root=Path(source["dataset_root"]),
            limit=limit,
            source_name=str(source.get("name", Path(source["dataset_root"]).name)),
        )
        samples.extend(source_samples)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(config["image_size"])
    transform_train, transform_val = build_transforms(image_size)

    train_samples = load_all_samples(config, split="train")
    val_samples = load_all_samples(config, split="val")

    race_boosts = config.get("race_boosts", {})
    source_boosts = {str(item.get("name", "default")): float(item.get("source_weight", 1.0)) for item in resolve_sources(config)}
    gender_boosts = config.get("gender_boosts", {})

    sampler, class_counts, race_counts, source_counts = make_sampler(train_samples, race_boosts, source_boosts, gender_boosts)

    train_loader = DataLoader(
        FaceDataset(train_samples, transform_train),
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(config.get("persistent_workers", True)),
    )
    val_loader = DataLoader(
        FaceDataset(val_samples, transform_val),
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=bool(config.get("persistent_workers", True)),
    )

    model = MultiTaskMobileNet(
        pretrained=bool(config["use_pretrained"]),
        backbone_name=str(config.get("backbone", "mobilenet_v3_large")),
        dropout=float(config.get("dropout", 0.3)),
    ).to(device)
    teacher_model = None
    teacher_checkpoint_path = config.get("teacher_checkpoint")
    if teacher_checkpoint_path:
        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu")
        teacher_config = teacher_checkpoint.get("config", config)
        teacher_model = build_model_from_config(teacher_config, pretrained=False).to(device)
        teacher_model.load_state_dict(teacher_checkpoint["model_state"])
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    gender_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config.get("min_learning_rate", 1e-6)),
    )

    freeze_epochs = int(config.get("freeze_backbone_epochs", 0))
    best_primary = float("-inf")
    history = []

    for epoch in range(int(config["epochs"])):
        backbone_trainable = epoch >= freeze_epochs
        for param in model.backbone.parameters():
            param.requires_grad = backbone_trainable

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            gender_criterion,
            age_sigma=float(config.get("age_label_sigma", 6.0)),
            age_ce_weight=float(config.get("age_ce_weight", 1.0)),
            age_kl_weight=float(config.get("age_kl_weight", 0.7)),
            age_reg_weight=float(config.get("age_reg_weight", 0.25)),
            gender_weight=float(config.get("gender_loss_weight", 0.25)),
            teacher_model=teacher_model,
            distill_temperature=float(config.get("distill_temperature", 2.0)),
            distill_age_weight=float(config.get("distill_age_weight", 0.0)),
            distill_gender_weight=float(config.get("distill_gender_weight", 0.0)),
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer,
            gender_criterion,
            age_sigma=float(config.get("age_label_sigma", 6.0)),
            age_ce_weight=float(config.get("age_ce_weight", 1.0)),
            age_kl_weight=float(config.get("age_kl_weight", 0.7)),
            age_reg_weight=float(config.get("age_reg_weight", 0.25)),
            gender_weight=float(config.get("gender_loss_weight", 0.25)),
            teacher_model=teacher_model,
            distill_temperature=float(config.get("distill_temperature", 2.0)),
            distill_age_weight=float(config.get("distill_age_weight", 0.0)),
            distill_gender_weight=float(config.get("distill_gender_weight", 0.0)),
            train=False,
        )
        scheduler.step()

        primary_score = val_metrics["age_one_off_acc"] + 0.15 * val_metrics["gender_acc"] - 0.002 * val_metrics["age_mae_proxy"]
        summary = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "backbone_trainable": backbone_trainable,
            "lr": optimizer.param_groups[0]["lr"],
            "primary_score": primary_score,
        }
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

        save_checkpoint(output_dir / "last.pt", model, config, summary)
        if primary_score > best_primary:
            best_primary = primary_score
            save_checkpoint(output_dir / "best.pt", model, config, summary)

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    best_val = max(history, key=lambda item: item["primary_score"])["val"] if history else {}
    meta = {
        "device": str(device),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "age_classes": AGE_CLASSES,
        "age_midpoints": AGE_MIDPOINTS,
        "gender_classes": GENDER_CLASSES,
        "class_counts": class_counts,
        "race_counts": race_counts,
        "source_counts": source_counts,
        "race_boosts": race_boosts,
        "source_boosts": source_boosts,
        "gender_boosts": gender_boosts,
        "backbone": config.get("backbone", "mobilenet_v3_large"),
        "image_size": image_size,
        "teacher_checkpoint": teacher_checkpoint_path,
        "distill_temperature": float(config.get("distill_temperature", 2.0)),
        "distill_age_weight": float(config.get("distill_age_weight", 0.0)),
        "distill_gender_weight": float(config.get("distill_gender_weight", 0.0)),
        "best_primary_score": best_primary,
        "best_age_acc": best_val.get("age_acc", 0.0),
        "best_age_one_off_acc": best_val.get("age_one_off_acc", 0.0),
        "best_gender_acc": best_val.get("gender_acc", 0.0),
        "best_age_mae_proxy": best_val.get("age_mae_proxy", 0.0),
        "best_age_cs5_proxy": best_val.get("age_cs5_proxy", 0.0),
        "exact_age_metrics_supported": False,
        "note": "FairFace provides age ranges instead of exact ages; MAE/CS metrics here are midpoint-based proxy metrics.",
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
