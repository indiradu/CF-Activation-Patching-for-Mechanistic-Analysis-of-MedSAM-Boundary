#!/usr/bin/env python3
"""Run MedSAM over a manifest while loading the model only once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import transform

from counterfactual_medsam.manifest import parse_box_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--medsam-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--box-column", default="box_xyxy")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--prediction-column", default="medsam_original_mask_path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_medsam(medsam_repo: Path, checkpoint: Path, device: str):
    sys.path.insert(0, str(medsam_repo))
    from segment_anything import sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    return model


def read_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    return image


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


@torch.no_grad()
def run_one(model, image: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    image_tensor = preprocess_image(image, device=str(next(model.parameters()).device))
    image_embedding = model.image_encoder(image_tensor)

    box_np = np.array([box_xyxy], dtype=np.float32)
    box_1024 = box_np / np.array([width, height, width, height], dtype=np.float32) * 1024
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=image_embedding.device)[:, None, :]

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not (args.medsam_repo / "segment_anything").exists():
        raise FileNotFoundError(f"MedSAM repo missing segment_anything package: {args.medsam_repo}")

    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_medsam(args.medsam_repo, args.checkpoint, args.device)

    rows = []
    for idx, row in df.iterrows():
        case_id = str(row[args.case_column])
        image_path = Path(str(row[args.image_column]))
        box = parse_box_xyxy(row[args.box_column])
        output_path = args.output_dir / f"{case_id}.png"

        image = read_image(image_path)
        pred = run_one(model, image, box)
        Image.fromarray(pred * 255).save(output_path)

        updated = row.to_dict()
        updated[args.prediction_column] = str(output_path)
        rows.append(updated)
        print(f"[{idx + 1}/{len(df)}] {case_id} -> {output_path}", flush=True)

    updated_manifest = args.output_dir / "manifest_with_predictions.csv"
    pd.DataFrame(rows).to_csv(updated_manifest, index=False)
    print(f"Wrote updated manifest to {updated_manifest}", flush=True)


if __name__ == "__main__":
    main()
