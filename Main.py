import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


class TransformSubset(Dataset):
    def __init__(self, image_folder, indices, transform, two_views=False, strong_transform=None):
        self.image_folder = image_folder
        self.indices = list(indices)
        self.transform = transform
        self.two_views = two_views
        self.strong_transform = strong_transform or transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        path, label = self.image_folder.samples[self.indices[item]]
        image = self.image_folder.loader(path)
        if self.two_views:
            return self.transform(image), self.strong_transform(image), label
        return self.transform(image), label


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stratified_target_split(dataset, labeled_ratio, test_ratio, seed):
    rng = random.Random(seed)
    class_to_indices = {}
    for index, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(index)

    labeled, unlabeled, test = [], [], []
    for _, indices in sorted(class_to_indices.items()):
        rng.shuffle(indices)
        n_test = max(1, int(round(len(indices) * test_ratio)))
        test_part = indices[:n_test]
        train_part = indices[n_test:]
        n_labeled = max(1, int(round(len(train_part) * labeled_ratio)))
        labeled_part = train_part[:n_labeled]
        unlabeled_part = train_part[n_labeled:]
        test.extend(test_part)
        labeled.extend(labeled_part)
        unlabeled.extend(unlabeled_part)

    rng.shuffle(labeled)
    rng.shuffle(unlabeled)
    rng.shuffle(test)
    return labeled, unlabeled, test


def split_labeled_train_val(dataset, labeled_indices, val_ratio, seed):
    if val_ratio <= 0:
        return list(labeled_indices), []

    rng = random.Random(seed + 1009)
    class_to_indices = {}
    for index in labeled_indices:
        _, label = dataset.samples[index]
        class_to_indices.setdefault(label, []).append(index)

    train_indices, val_indices = [], []
    for _, indices in sorted(class_to_indices.items()):
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue
        n_val = max(1, int(round(len(indices) * val_ratio)))
        n_val = min(n_val, len(indices) - 1)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_transforms(img_size, profile="fault_light", normalize_profile="default"):
    if normalize_profile == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if profile == "generic":
        weak_crop = (0.85, 1.0)
        strong_crop = (0.70, 1.0)
        blur_weak = 0.2
        blur_strong = 0.35
        affine_prob = 0.5
        erase_prob = 0.25
    else:
        weak_crop = (0.92, 1.0)
        strong_crop = (0.85, 1.0)
        blur_weak = 0.05
        blur_strong = 0.10
        affine_prob = 0.20
        erase_prob = 0.0

    weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=weak_crop),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=blur_weak),
        transforms.ToTensor(),
        normalize,
    ])
    strong = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=strong_crop),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=blur_strong),
        transforms.RandomApply([transforms.RandomAffine(degrees=6, translate=(0.03, 0.03))], p=affine_prob),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=erase_prob, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return weak, strong, test


class PrivacyViTBackbone(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        privacy_noise=0.02,
        privacy_dropout=0.1,
    ):
        super().__init__()
        self.privacy_noise = privacy_noise
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.privacy_drop = nn.Dropout(privacy_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.feat_dim = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, use_privacy=True):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        if self.training and use_privacy:
            x = x + torch.randn_like(x) * self.privacy_noise
            x = self.privacy_drop(x)
        x = self.encoder(x)
        return self.norm(x[:, 0])


class PrivacyConvBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, privacy_noise=0.02, privacy_dropout=0.1):
        super().__init__()
        self.privacy_noise = privacy_noise
        self.privacy_drop = nn.Dropout2d(privacy_dropout)
        channels = [64, 96, 128, embed_dim]
        layers = []
        prev = in_channels
        for i, width in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers.extend([
                nn.Conv2d(prev, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
            ])
            prev = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.feat_dim = embed_dim

    def forward(self, x, use_privacy=True):
        x = self.features(x)
        if self.training and use_privacy:
            x = x + torch.randn_like(x) * self.privacy_noise
            x = self.privacy_drop(x)
        x = self.pool(x).flatten(1)
        return self.norm(x)


class PrivacyResNetBackbone(nn.Module):
    def __init__(self, name="resnet18", embed_dim=256, pretrained=True, privacy_noise=0.02, privacy_dropout=0.1):
        super().__init__()
        self.privacy_noise = privacy_noise
        self.privacy_drop = nn.Dropout(privacy_dropout)
        if name == "resnet50":
            weight_enum = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weight_enum)
        elif name == "resnet18":
            weight_enum = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weight_enum)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {name}")
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.projector = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.feat_dim = embed_dim

    def forward(self, x, use_privacy=True):
        feat = self.encoder(x)
        if self.training and use_privacy:
            feat = feat + torch.randn_like(feat) * self.privacy_noise
            feat = self.privacy_drop(feat)
        return self.projector(feat)


def build_backbone(args, privacy_noise=None, privacy_dropout=None):
    backbone_type = getattr(args, "backbone_type", "vit")
    privacy_noise = getattr(args, "privacy_noise", 0.02) if privacy_noise is None else privacy_noise
    privacy_dropout = getattr(args, "privacy_dropout", 0.1) if privacy_dropout is None else privacy_dropout
    if backbone_type == "conv":
        return PrivacyConvBackbone(
            in_channels=3,
            embed_dim=getattr(args, "embed_dim", 128),
            privacy_noise=privacy_noise,
            privacy_dropout=privacy_dropout,
        )
    if backbone_type in {"resnet18", "resnet50"}:
        return PrivacyResNetBackbone(
            name=backbone_type,
            embed_dim=getattr(args, "embed_dim", 256),
            pretrained=getattr(args, "pretrained_backbone", True),
            privacy_noise=privacy_noise,
            privacy_dropout=privacy_dropout,
        )
    if backbone_type == "vit":
        return PrivacyViTBackbone(
            img_size=getattr(args, "img_size", 64),
            patch_size=getattr(args, "patch_size", 8),
            in_channels=3,
            embed_dim=getattr(args, "embed_dim", 128),
            depth=getattr(args, "depth", 4),
            heads=getattr(args, "heads", 4),
            privacy_noise=privacy_noise,
            privacy_dropout=privacy_dropout,
        )
    raise ValueError(f"Unsupported backbone_type: {backbone_type}")


class ReinforcedPrivacyDA(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = backbone
        self.policy_norm = nn.LayerNorm(backbone.feat_dim)
        self.actor = nn.Sequential(
            nn.Linear(backbone.feat_dim, backbone.feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.feat_dim, num_classes),
        )
        self.critic = nn.Sequential(
            nn.Linear(backbone.feat_dim, backbone.feat_dim // 2),
            nn.GELU(),
            nn.Linear(backbone.feat_dim // 2, 1),
        )

    def forward(self, x, use_privacy=True):
        feat = self.backbone(x, use_privacy=use_privacy)
        feat_norm = self.policy_norm(feat)
        return self.actor(feat_norm), self.critic(feat_norm).squeeze(1), feat_norm


def supervised_contrastive_loss(features, labels, temperature=0.1):
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positives = same_class & ~eye
    valid = positives.sum(dim=1) > 0
    if not valid.any():
        return features.new_tensor(0.0)
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    mean_log_prob_pos = (positives * log_prob).sum(dim=1) / positives.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def class_prototype_loss(source_feat, source_label, target_feat, target_label, num_classes, temperature=0.1):
    source_feat = F.normalize(source_feat, dim=1)
    target_feat = F.normalize(target_feat, dim=1)
    prototypes = []
    valid_classes = []
    for cls in range(num_classes):
        mask = source_label == cls
        if mask.any():
            prototypes.append(F.normalize(source_feat[mask].mean(dim=0, keepdim=True), dim=1))
            valid_classes.append(cls)
    if not prototypes:
        return source_feat.new_tensor(0.0)
    prototypes = torch.cat(prototypes, dim=0)
    valid_classes = torch.tensor(valid_classes, device=source_feat.device)
    keep = (target_label.unsqueeze(1) == valid_classes.unsqueeze(0)).any(dim=1)
    if not keep.any():
        return source_feat.new_tensor(0.0)
    remapped = (target_label[keep].unsqueeze(1) == valid_classes.unsqueeze(0)).float().argmax(dim=1)
    logits = target_feat[keep] @ prototypes.T / temperature
    return F.cross_entropy(logits, remapped)


def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    return -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)


def source_actor_critic_loss(logits, values, labels, ce_weight, entropy_weight):
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    reward = (action == labels).float() * 2.0 - 1.0
    advantage = reward - values.detach()
    actor = -(dist.log_prob(action) * advantage).mean()
    critic = F.mse_loss(values, reward)
    ce = F.cross_entropy(logits, labels)
    entropy = dist.entropy().mean()
    return actor + 0.5 * critic + ce_weight * ce - entropy_weight * entropy, {
        "src_reward": reward.mean().item(),
        "src_ce": ce.item(),
    }


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent

# Edit these values and run this file directly. Defaults come from the shared engine.
RUN_CONFIG = {
    "task_names": ["PU_N15_M07_F10_to_N09_M07_F10"],
    "methods": ["Ours"], "seeds": [42],
    "output_dir": CODE_ROOT / "runs" / "PU_main_v2",
    "epochs": 60, "batch_size": 128, "gpu_ids": None,
}


def main(config=None):
    from unified_experiment import main as run_unified
    settings = {**RUN_CONFIG, **(config or {})}
    if "seed" in settings:
        settings["seeds"] = [settings.pop("seed")]
    return run_unified(settings)


if __name__ == "__main__":
    main()
