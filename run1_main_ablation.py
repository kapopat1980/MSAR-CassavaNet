import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.msar_cassavanet import MSARCassavaNet
from utils.losses import FocalLoss

def get_ablation_model(variant_name: str) -> nn.Module:
    """Instantiates architecture variants for component ablation study."""
    if variant_name == "full":
        return MSARCassavaNet(use_se=True, use_spatial_gate=True, use_multiscale=True)
    elif variant_name == "no_se":
        return MSARCassavaNet(use_se=False, use_spatial_gate=True, use_multiscale=True)
    elif variant_name == "no_spatial":
        return MSARCassavaNet(use_se=True, use_spatial_gate=False, use_multiscale=True)
    elif variant_name == "no_multiscale":
        return MSARCassavaNet(use_se=True, use_spatial_gate=True, use_multiscale=False)
    elif variant_name == "base_only":
        return MSARCassavaNet(use_se=False, use_spatial_gate=False, use_multiscale=False)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSAR-CassavaNet Main Ablation Experiment")
    parser.add_argument("--variant", type=str, default="full", choices=["full", "no_se", "no_spatial", "no_multiscale", "base_only"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = get_ablation_model(args.variant)
    print(f"Model variant [{args.variant}] initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
