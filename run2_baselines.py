import argparse
import torch.nn as nn
import torchvision.models as models

def build_baseline_model(model_name: str, pretrained: bool = False, num_classes: int = 5) -> nn.Module:
    """Builds lightweight/standard baselines matching evaluation protocol."""
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported baseline architecture: {model_name}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Models Benchmark")
    parser.add_argument("--model", type=str, default="mobilenet_v3_small")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    net = build_baseline_model(args.model, pretrained=args.pretrained)
    print(f"Loaded baseline {args.model} (pretrained={args.pretrained}). Params: {sum(p.numel() for p in net.parameters()):,}")
