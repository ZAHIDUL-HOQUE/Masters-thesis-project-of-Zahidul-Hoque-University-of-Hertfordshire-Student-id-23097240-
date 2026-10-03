"""
Vehicle Exterior Damage Detection - Model Definition & Weights Loader
Architecture: Prototypical Mask R-CNN (ResNet-50-FPN backbone)
Dataset: CarDD (Car Damage Detection Dataset)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Target damage classes corresponding to CarDD dataset mapping
DAMAGE_CLASSES = [
    "Background",
    "Dent",
    "Scratch",
    "Crack",
    "Glass Shatter",
    "Lamp Broken",
    "Tire Flat"
]

NUM_CLASSES = len(DAMAGE_CLASSES)

# Vibrant, visually distinct color palette for damage visualization (RGB 0-255)
CLASS_COLORS = {
    "Dent": (255, 122, 0),         # Vibrant Amber / Orange
    "Scratch": (0, 210, 255),       # Bright Cyan / Neon Blue
    "Crack": (168, 85, 247),       # Neon Purple / Violet
    "Glass Shatter": (255, 42, 109), # Radiant Coral / Crimson
    "Lamp Broken": (250, 204, 21),   # Golden Yellow
    "Tire Flat": (16, 185, 129),     # Emerald Green
    "Background": (150, 150, 150)
}

CLASS_HEX_COLORS = {
    "Dent": "#FF7A00",
    "Scratch": "#00D2FF",
    "Crack": "#A855F7",
    "Glass Shatter": "#FF2A6D",
    "Lamp Broken": "#FACC15",
    "Tire Flat": "#10B981"
}

# Severity weights used for calculating overall vehicle repair impact
DAMAGE_SEVERITY_WEIGHTS = {
    "Dent": 1.5,
    "Scratch": 1.0,
    "Crack": 2.0,
    "Glass Shatter": 3.0,
    "Lamp Broken": 2.5,
    "Tire Flat": 2.5
}


class PrototypicalBoxPredictor(nn.Module):
    """
    Prototypical Classification Head for Mask R-CNN.
    Computes metric distance to learnable class prototype vectors in feature space.
    """
    def __init__(self, in_channels: int, num_classes: int, scale: float = 20.0):
        super().__init__()
        self.scale = scale
        self.cls_prototypes = nn.Parameter(torch.randn(num_classes, in_channels))
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

        # Xavier Uniform Initialization
        nn.init.xavier_uniform_(self.cls_prototypes)
        nn.init.xavier_uniform_(self.bbox_pred.weight)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor):
        x_norm = F.normalize(x, p=2, dim=1)
        proto_norm = F.normalize(self.cls_prototypes, p=2, dim=1)

        # Compute Euclidean distance between feature representations and prototypes
        dists = torch.cdist(x_norm, proto_norm, p=2)
        squared_dists = dists.pow(2)

        # Scale negative squared distance into classification logits
        logits = -squared_dists * self.scale
        bbox_deltas = self.bbox_pred(x)

        return logits, bbox_deltas


def get_prototypical_model(num_classes: int = NUM_CLASSES, proto_scale: float = 20.0, nms_thresh: float = 0.25):
    """
    Constructs the Prototypical Mask R-CNN model matching the CarDD training checkpoint.
    """
    model = maskrcnn_resnet50_fpn(weights=None)
    model.roi_heads.nms_thresh = nms_thresh

    # Replace box predictor with Prototypical Box Predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = PrototypicalBoxPredictor(
        in_features_box, num_classes, scale=proto_scale
    )

    # Replace mask predictor with MaskRCNNPredictor matching CarDD classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


_CACHED_MODEL = None
_CACHED_DEVICE = None


def load_damage_model(weights_path: str = "latest_model.pth", device: torch.device = None):
    """
    Loads and caches the pretrained Prototypical Mask R-CNN weights.
    Returns (model, device).
    """
    global _CACHED_MODEL, _CACHED_DEVICE

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if _CACHED_MODEL is not None and _CACHED_DEVICE == device:
        return _CACHED_MODEL, _CACHED_DEVICE

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    print(f"Loading Prototypical Mask R-CNN from '{weights_path}' on device '{device}'...")
    model = get_prototypical_model(num_classes=NUM_CLASSES)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    load_res = model.load_state_dict(state_dict)
    print(f"Model loaded successfully: {load_res}")

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_DEVICE = device

    return model, device
