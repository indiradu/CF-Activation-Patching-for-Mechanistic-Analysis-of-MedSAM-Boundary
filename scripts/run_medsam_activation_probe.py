#!/usr/bin/env python3
"""Activation summaries and patching for matched MedSAM counterfactual pairs."""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import transform

from counterfactual_medsam.manifest import parse_box_xyxy
from counterfactual_medsam.metrics import evaluate_mask_pair
from counterfactual_medsam.regions import probing_region, read_mask, region_mask, region_mean, resize_mask


DEFAULT_LAYERS = "block3,block6,block9,neck"
DEFAULT_REGIONS = "whole,boundary,interior,background"
DEFAULT_CONTROLS = "matched_cf_to_original"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--medsam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--counterfactual-column", default="counterfactual_path")
    parser.add_argument("--reference-column", default="lung_mask_path")
    parser.add_argument("--original-prediction-column", default="")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--regions", default=DEFAULT_REGIONS)
    parser.add_argument("--controls", default=DEFAULT_CONTROLS)
    parser.add_argument("--nonmatched-shift", type=int, default=1)
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-patching", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    normalized = value.replace(":", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def load_medsam(medsam_repo: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    return model


def layer_module(model: torch.nn.Module, layer: str) -> torch.nn.Module:
    normalized = layer.lower().replace("_", "").replace(".", "")
    if normalized.startswith("block"):
        index = int(normalized.removeprefix("block"))
        return model.image_encoder.blocks[index]
    if normalized == "neck":
        return model.image_encoder.neck
    raise ValueError(f"Unsupported layer: {layer}")


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def preprocess_image(image: np.ndarray, device: str) -> torch.Tensor:
    image_1024 = transform.resize(
        image,
        (1024, 1024),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    image_1024 = (image_1024 - image_1024.min()) / np.clip(
        image_1024.max() - image_1024.min(),
        a_min=1e-8,
        a_max=None,
    )
    return torch.tensor(image_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)


def output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract tensor from hook output type: {type(output)!r}")


def replace_output_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    raise TypeError(f"Cannot replace hook output type: {type(output)!r}")


def is_channel_last(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 4 and tensor.shape[1] == tensor.shape[2] and tensor.shape[-1] > tensor.shape[1]


def spatial_shape(tensor: torch.Tensor) -> tuple[int, int]:
    if is_channel_last(tensor):
        return int(tensor.shape[1]), int(tensor.shape[2])
    if tensor.ndim == 4:
        return int(tensor.shape[2]), int(tensor.shape[3])
    raise ValueError(f"Expected 4D activation, got {tuple(tensor.shape)}")


def activation_map(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().abs()
    if is_channel_last(tensor):
        return tensor.mean(dim=-1).squeeze(0).cpu().numpy()
    return tensor.mean(dim=1).squeeze(0).cpu().numpy()


def random_like_mask(mask: np.ndarray, seed: int) -> np.ndarray:
    """Sample a random spatial region with the same token count as mask."""

    mask = mask.astype(bool)
    count = int(mask.sum())
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    if count >= mask.size:
        return np.ones_like(mask, dtype=bool)

    rng = np.random.default_rng(seed)
    flat = np.zeros(mask.size, dtype=bool)
    flat[rng.choice(mask.size, size=count, replace=False)] = True
    return flat.reshape(mask.shape)


def resolve_region_mask(
    reference_mask: np.ndarray,
    region: str,
    feature_shape: tuple[int, int],
    boundary_width: int,
    original_prediction: np.ndarray | None,
    random_seed: int,
) -> np.ndarray:
    region = region.lower()
    random_prefix = "random_like_"
    is_random_like = region.startswith(random_prefix)
    base_region = region.removeprefix(random_prefix) if is_random_like else region
    full_region = probing_region(
        reference_mask,
        base_region,
        boundary_width=boundary_width,
        prediction=original_prediction,
    )
    feature_region = resize_mask(full_region, feature_shape)
    if is_random_like:
        return random_like_mask(feature_region, seed=random_seed)
    return feature_region


def blend_activation(
    original: torch.Tensor,
    donor: torch.Tensor,
    reference_mask: np.ndarray,
    region: str,
    boundary_width: int,
    original_prediction: np.ndarray | None = None,
    random_seed: int = 0,
) -> torch.Tensor:
    if region == "whole":
        return donor
    h, w = spatial_shape(original)
    feature_region = resolve_region_mask(
        reference_mask,
        region,
        feature_shape=(h, w),
        boundary_width=boundary_width,
        original_prediction=original_prediction,
        random_seed=random_seed,
    )
    mask = torch.as_tensor(feature_region, dtype=original.dtype, device=original.device)
    if is_channel_last(original):
        mask = mask[None, :, :, None]
    else:
        mask = mask[None, None, :, :]
    return torch.where(mask.bool(), donor, original)


@torch.no_grad()
def encode_with_hooks(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str,
    capture_layers: list[str],
    patch_layer: str | None = None,
    donor_activation: torch.Tensor | None = None,
    patch_region: str = "whole",
    reference_mask: np.ndarray | None = None,
    original_prediction: np.ndarray | None = None,
    boundary_width: int = 5,
    random_seed: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    activations: dict[str, torch.Tensor] = {}
    handles = []

    for layer in capture_layers:
        module = layer_module(model, layer)

        def capture_hook(_module, _inputs, output, *, name=layer):
            activations[name] = output_tensor(output).detach().clone()

        handles.append(module.register_forward_hook(capture_hook))

    if patch_layer is not None:
        if donor_activation is None or reference_mask is None:
            raise ValueError("Patching requires donor_activation and reference_mask")
        module = layer_module(model, patch_layer)

        def patch_hook(_module, _inputs, output):
            original = output_tensor(output)
            patched = blend_activation(
                original=original,
                donor=donor_activation.to(original.device),
                reference_mask=reference_mask,
                region=patch_region,
                boundary_width=boundary_width,
                original_prediction=original_prediction,
                random_seed=random_seed,
            )
            return replace_output_tensor(output, patched)

        handles.append(module.register_forward_hook(patch_hook))

    try:
        image_tensor = preprocess_image(image, device=device)
        embedding = model.image_encoder(image_tensor)
    finally:
        for handle in handles:
            handle.remove()
    return embedding, activations


@torch.no_grad()
def decode_mask(
    model: torch.nn.Module,
    image_embedding: torch.Tensor,
    box_xyxy: tuple[int, int, int, int],
    original_size: tuple[int, int],
) -> np.ndarray:
    height, width = original_size
    box_np = np.array([box_xyxy], dtype=np.float32)
    box_1024 = box_np / np.array([width, height, width, height], dtype=np.float32) * 1024
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=image_embedding.device)[:, None, :]

    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    pred = F.interpolate(
        torch.sigmoid(low_res_logits),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def summarize_layer(
    *,
    case_id: str,
    layer: str,
    original_activation: torch.Tensor,
    counterfactual_activation: torch.Tensor,
    reference_mask: np.ndarray,
    boundary_width: int,
) -> dict[str, float | str]:
    diff_map = activation_map(counterfactual_activation - original_activation)
    feature_mask = resize_mask(reference_mask, diff_map.shape)
    boundary = region_mask(feature_mask, "boundary", boundary_width=boundary_width)
    interior = region_mask(feature_mask, "interior", boundary_width=boundary_width)
    background = region_mask(feature_mask, "background", boundary_width=boundary_width)
    return {
        "case_id": case_id,
        "layer": layer,
        "mean_abs_diff": float(diff_map.mean()),
        "max_abs_diff": float(diff_map.max()),
        "boundary_mean_abs_diff": region_mean(diff_map, boundary),
        "interior_mean_abs_diff": region_mean(diff_map, interior),
        "background_mean_abs_diff": region_mean(diff_map, background),
    }


def main() -> None:
    args = parse_args()
    layers = split_csv(args.layers)
    regions = split_csv(args.regions)
    controls = split_csv(args.controls)

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_mask_dir = args.output_dir / "patch_masks"
    if args.run_patching:
        patch_mask_dir.mkdir(parents=True, exist_ok=True)

    model = load_medsam(args.medsam_repo, args.checkpoint, args.device)
    summary_rows = []
    patch_rows = []

    supported_controls = {
        "matched_cf_to_original",
        "reverse_original_to_cf",
        "nonmatched_cf_to_original",
    }
    unknown_controls = sorted(set(controls) - supported_controls)
    if unknown_controls:
        raise ValueError(f"Unsupported controls: {unknown_controls}")
    if len(df) < 2 and "nonmatched_cf_to_original" in controls:
        raise ValueError("nonmatched_cf_to_original requires at least two rows")

    for position, (_row_idx, row) in enumerate(df.iterrows()):
        case_id = str(row[args.case_column])
        counterfactual_path = str(row.get(args.counterfactual_column, "")).strip()
        if not counterfactual_path or counterfactual_path.lower() == "nan":
            raise ValueError(f"Missing counterfactual path for {case_id}")

        original = read_image(Path(str(row[args.image_column])))
        counterfactual = read_image(Path(counterfactual_path))
        reference_mask = read_mask(Path(str(row[args.reference_column])))
        original_prediction = None
        if args.original_prediction_column:
            value = str(row.get(args.original_prediction_column, "")).strip()
            if value and value.lower() != "nan":
                original_prediction = read_mask(Path(value))
        box = parse_box_xyxy(row[args.box_column])

        _, original_acts = encode_with_hooks(model, original, args.device, capture_layers=layers)
        _, counterfactual_acts = encode_with_hooks(model, counterfactual, args.device, capture_layers=layers)
        nonmatched_case_id = ""
        nonmatched_counterfactual = None
        nonmatched_counterfactual_acts = None
        if "nonmatched_cf_to_original" in controls:
            donor_idx = (position + args.nonmatched_shift) % len(df)
            if donor_idx == position:
                donor_idx = (position + 1) % len(df)
            donor_row = df.iloc[donor_idx]
            nonmatched_case_id = str(donor_row[args.case_column])
            donor_counterfactual_path = str(donor_row.get(args.counterfactual_column, "")).strip()
            if not donor_counterfactual_path or donor_counterfactual_path.lower() == "nan":
                raise ValueError(f"Missing non-matched counterfactual path for {nonmatched_case_id}")
            nonmatched_counterfactual = read_image(Path(donor_counterfactual_path))
            _, nonmatched_counterfactual_acts = encode_with_hooks(
                model,
                nonmatched_counterfactual,
                args.device,
                capture_layers=layers,
            )

        for layer in layers:
            summary_rows.append(
                summarize_layer(
                    case_id=case_id,
                    layer=layer,
                    original_activation=original_acts[layer],
                    counterfactual_activation=counterfactual_acts[layer],
                    reference_mask=reference_mask,
                    boundary_width=args.boundary_width,
                )
            )

        if args.run_patching:
            control_inputs = {
                "matched_cf_to_original": {
                    "target_image": original,
                    "target_name": "original",
                    "donor_case_id": case_id,
                    "donor_name": "matched_counterfactual",
                    "donor_acts": counterfactual_acts,
                },
                "reverse_original_to_cf": {
                    "target_image": counterfactual,
                    "target_name": "counterfactual",
                    "donor_case_id": case_id,
                    "donor_name": "matched_original",
                    "donor_acts": original_acts,
                },
            }
            if nonmatched_counterfactual_acts is not None:
                control_inputs["nonmatched_cf_to_original"] = {
                    "target_image": original,
                    "target_name": "original",
                    "donor_case_id": nonmatched_case_id,
                    "donor_name": "nonmatched_counterfactual",
                    "donor_acts": nonmatched_counterfactual_acts,
                }

            for control in controls:
                control_input = control_inputs[control]
                target_image = control_input["target_image"]
                target_name = str(control_input["target_name"])
                donor_case_id = str(control_input["donor_case_id"])
                donor_name = str(control_input["donor_name"])
                donor_acts = control_input["donor_acts"]
                for layer in layers:
                    for region in regions:
                        random_seed = stable_seed(case_id, control, layer, region)
                        patched_embedding, _ = encode_with_hooks(
                            model,
                            target_image,
                            args.device,
                            capture_layers=[],
                            patch_layer=layer,
                            donor_activation=donor_acts[layer],
                            patch_region=region,
                            reference_mask=reference_mask,
                            original_prediction=original_prediction,
                            boundary_width=args.boundary_width,
                            random_seed=random_seed,
                        )
                        pred = decode_mask(model, patched_embedding, box, target_image.shape[:2])
                        pred_path = patch_mask_dir / f"{case_id}__{control}__{layer}__{region}.png"
                        Image.fromarray(pred * 255).save(pred_path)
                        metrics = evaluate_mask_pair(
                            prediction_path=pred_path,
                            reference_path=Path(str(row[args.reference_column])),
                        )
                        patch_rows.append(
                            {
                                "case_id": case_id,
                                "control": control,
                                "target_image": target_name,
                                "donor_image": donor_name,
                                "donor_case_id": donor_case_id,
                                "layer": layer,
                                "region": region,
                                "prediction_path": str(pred_path),
                                **metrics,
                            }
                        )

        print(f"[{position + 1}/{len(df)}] {case_id}", flush=True)
        pd.DataFrame(summary_rows).to_csv(args.output_dir / "activation_summary_partial.csv", index=False)
        if args.run_patching:
            pd.DataFrame(patch_rows).to_csv(args.output_dir / "patching_metrics_partial_live.csv", index=False)

    summary_path = args.output_dir / "activation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote activation summary to {summary_path}")

    if args.run_patching:
        patch_path = args.output_dir / "patching_metrics.csv"
        pd.DataFrame(patch_rows).to_csv(patch_path, index=False)
        print(f"Wrote patching metrics to {patch_path}")


if __name__ == "__main__":
    main()
