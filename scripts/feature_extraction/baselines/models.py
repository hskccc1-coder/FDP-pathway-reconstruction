from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ModelBundle:
    model: nn.Module
    initialization: str


class SMPAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        return {"mask": self.model(x), "edge": None, "distance": None}


class SEANetAdapter(nn.Module):
    """Adapter around the authors' SEANet implementation.

    The upstream model returns mask logits plus sigmoid edge/distance maps. Losses
    in this project consume logits, so the auxiliary probabilities are converted
    back to stable logits here.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _to_logits(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        value = value.clamp(eps, 1.0 - eps)
        return torch.logit(value)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        output = self.model(x)
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise RuntimeError("Official SEANet must return (mask, edges, distance).")
        mask, edges, distance = output[:3]
        if isinstance(edges, (tuple, list)):
            edge = [self._to_logits(v) for v in edges]
        else:
            edge = self._to_logits(edges)
        return {
            "mask": mask,
            "edge": edge,
            "distance": self._to_logits(distance),
        }


def _load_seanet_module(source_dir: Path):
    models_py = source_dir / "models.py"
    if not models_py.is_file():
        raise FileNotFoundError(
            f"SEANet official models.py was not found: {models_py}. "
            "Clone https://github.com/long123524/SEANet_torch or update seanet.source_dir."
        )
    module_name = "comparison_official_seanet_models"
    spec = importlib.util.spec_from_file_location(module_name, models_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import SEANet from {models_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _replace_seanet_classifier(model: nn.Module, num_classes: int) -> None:
    head = getattr(model, "conv_final1", None)
    if not isinstance(head, nn.Conv2d):
        raise AttributeError("Unsupported SEANet source: conv_final1 is missing or not Conv2d.")
    model.conv_final1 = nn.Conv2d(
        head.in_channels,
        num_classes,
        kernel_size=head.kernel_size,
        stride=head.stride,
        padding=head.padding,
        dilation=head.dilation,
        groups=head.groups,
        bias=head.bias is not None,
    )


def _official_seanet_weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if tuple(module.weight.shape) == (1, 5, 1, 1):
            nn.init.constant_(module.weight, 0.2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _load_official_vgg16_mat(model: nn.Module, mat_path: Path) -> int:
    """Load matching tensors from the VGG16 .mat used by the official repo."""
    import scipy.io as sio

    raw: dict[str, Any] = sio.loadmat(str(mat_path))
    state = model.state_dict()
    matched = 0
    for raw_name, raw_value in raw.items():
        if raw_name.startswith("__"):
            continue
        name = raw_name.replace("-", ".")
        if name not in state:
            continue
        array = np.asarray(raw_value).squeeze()
        tensor = torch.as_tensor(array, dtype=state[name].dtype)
        if tensor.shape == state[name].shape:
            state[name] = tensor
            matched += 1
    if matched == 0:
        raise RuntimeError(f"No SEANet VGG16 weights matched tensors in {mat_path}")
    model.load_state_dict(state, strict=False)
    return matched


SEANET_VGG16_CONVS = [
    "conv1_1", "conv1_2",
    "conv2_1", "conv2_2",
    "conv3_1", "conv3_2", "conv3_3",
    "conv4_1", "conv4_2", "conv4_3",
    "conv5_1", "conv5_2", "conv5_3",
]


def _load_torchvision_vgg16(model: nn.Module) -> int:
    """Copy the 13 ImageNet VGG16 convolutions into SEANet's VGG16 encoder.

    SEANet's conv5 layers are dilated, but kernel shapes match VGG16 exactly, so the
    pretrained kernels are reused unchanged, as in the HED/RCF lineage SEANet builds on.
    """
    from torchvision.models import VGG16_Weights, vgg16

    source = [m for m in vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features if isinstance(m, nn.Conv2d)]
    if len(source) != len(SEANET_VGG16_CONVS):
        raise RuntimeError(f"Expected 13 VGG16 convolutions, found {len(source)}")
    with torch.no_grad():
        for name, conv in zip(SEANET_VGG16_CONVS, source):
            target = getattr(model, name)
            if target.weight.shape != conv.weight.shape:
                raise RuntimeError(f"SEANet {name} shape {tuple(target.weight.shape)} != VGG16 {tuple(conv.weight.shape)}")
            target.weight.copy_(conv.weight)
            target.bias.copy_(conv.bias)
    return len(SEANET_VGG16_CONVS)


def freeze_encoder_batchnorm(model: nn.Module) -> int:
    """Fix BatchNorm statistics and affine terms of a pretrained SMP encoder.

    With batch size 2 the running statistics cannot be re-estimated reliably, so the
    ImageNet values are kept. Must be re-applied after every model.train() call.
    """
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is None:
        return 0
    count = 0
    for module in encoder.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            count += 1
    return count


def build_model(
    name: str,
    num_classes: int,
    *,
    pretrained: bool,
    encoder_name: str = "resnet34",
    deeplab_output_stride: int = 16,
    deeplab_atrous_rates: list[int] | None = None,
    seanet_source_dir: str | None = None,
    seanet_pretrained_source: str = "torchvision_vgg16_imagenet",
    seanet_vgg16_mat: str | None = None,
) -> ModelBundle:
    key = name.lower().replace("_", "").replace("+", "plus")
    if key in {"unetpp", "unetplusplus"}:
        import segmentation_models_pytorch as smp

        weights = "imagenet" if pretrained else None
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=num_classes,
        )
        return ModelBundle(SMPAdapter(model), f"{encoder_name}:{weights or 'random'}")

    if key in {"deeplabv3plus", "deeplabv3p"}:
        import inspect

        import segmentation_models_pytorch as smp

        extra: dict[str, Any] = {}
        if deeplab_atrous_rates is not None:
            if "decoder_atrous_rates" not in inspect.signature(smp.DeepLabV3Plus.__init__).parameters:
                raise RuntimeError(
                    "Installed segmentation_models_pytorch cannot set ASPP rates; "
                    "install the version pinned in requirements.txt"
                )
            extra["decoder_atrous_rates"] = tuple(int(rate) for rate in deeplab_atrous_rates)
        weights = "imagenet" if pretrained else None
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            encoder_output_stride=int(deeplab_output_stride),
            in_channels=3,
            classes=num_classes,
            **extra,
        )
        rates = extra.get("decoder_atrous_rates", "smp-default")
        return ModelBundle(
            SMPAdapter(model),
            f"{encoder_name}:{weights or 'random'}:os{deeplab_output_stride}:aspp{rates}",
        )

    if key == "seanet":
        if not seanet_source_dir:
            raise ValueError("seanet.source_dir is required for the official SEANet model.")
        module = _load_seanet_module(Path(seanet_source_dir))
        if not hasattr(module, "SEANet"):
            raise AttributeError("Official SEANet models.py does not export SEANet.")
        raw_model = module.SEANet()
        raw_model.apply(_official_seanet_weights_init)
        _replace_seanet_classifier(raw_model, num_classes)
        _official_seanet_weights_init(raw_model.conv_final1)
        initialization = "official-SEANet:random"
        if pretrained:
            if seanet_pretrained_source == "torchvision_vgg16_imagenet":
                count = _load_torchvision_vgg16(raw_model)
                initialization = f"official-SEANet:torchvision-VGG16-IMAGENET1K_V1({count}-convs)"
            elif seanet_pretrained_source == "official_vgg16_mat":
                if not seanet_vgg16_mat or not Path(seanet_vgg16_mat).is_file():
                    raise FileNotFoundError(f"SEANet VGG16 .mat not found: {seanet_vgg16_mat}")
                count = _load_official_vgg16_mat(raw_model, Path(seanet_vgg16_mat))
                initialization = f"official-SEANet:VGG16-mat({count}-tensors)"
            else:
                raise ValueError(f"Unknown seanet.pretrained_source: {seanet_pretrained_source}")
        return ModelBundle(SEANetAdapter(raw_model), initialization)

    raise ValueError(f"Unknown model: {name}")
