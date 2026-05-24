#!/usr/bin/env python3
"""Generate PRISM-style diffusion counterfactual chest X-rays from a manifest.

The public PRISM repository documents a Stable-Diffusion-based counterfactual
generation path, but its current main branch does not include all helper
modules imported by `generate_cf_images.py`. This script provides a small,
manifest-driven diffusion image-to-image runner for our MedSAM experiments so
we can test whether diffusion counterfactuals reproduce the CF-Seg activation
patching pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


DEFAULT_PROMPT = "frontal chest x-ray of a patient with no findings"
DEFAULT_NEGATIVE_PROMPT = "pleural effusion, lung opacity, edema, support devices, cropped anatomy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--model-id", default="IrohXu/stable-diffusion-mimic-cxr-v0.1")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--counterfactual-column", default="prism_counterfactual_path")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_pipeline(model_id: str, device: str, dtype: torch.dtype):
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def read_image(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    return image.resize((512, 512), Image.Resampling.BICUBIC), original_size


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch_dtype(args.torch_dtype)
    pipe = load_pipeline(args.model_id, args.device, dtype)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    paths = []
    for index, row in df.iterrows():
        case_id = str(row[args.case_column])
        image, original_size = read_image(Path(str(row[args.image_column])))
        generated = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
        out_path = args.output_dir / f"{case_id}.png"
        generated.resize(original_size, Image.Resampling.BICUBIC).save(out_path)
        paths.append(str(out_path))
        print(f"[{index + 1}/{len(df)}] {case_id} -> {out_path}", flush=True)

    df = df.copy()
    df[args.counterfactual_column] = paths
    df["prism_model_id"] = args.model_id
    df["prism_prompt"] = args.prompt
    df["prism_negative_prompt"] = args.negative_prompt
    df["prism_strength"] = args.strength
    df["prism_guidance_scale"] = args.guidance_scale
    df["prism_num_inference_steps"] = args.num_inference_steps
    df.to_csv(args.output_manifest, index=False)
    print(f"Wrote manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
