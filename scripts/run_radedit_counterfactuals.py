#!/usr/bin/env python3
"""Generate RadEdit lung-region counterfactuals from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps


DEFAULT_PROMPT = "No acute cardiopulmonary process"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--mask-column", default="lung_mask_path")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--counterfactual-column", default="radedit_counterfactual_path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--weight", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--skip-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_radedit_pipeline(device: str, dtype: torch.dtype):
    from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline
    from diffusers import UNet2DConditionModel
    from transformers import AutoModel, AutoTokenizer

    unet = UNet2DConditionModel.from_pretrained(
        "microsoft/radedit",
        subfolder="unet",
        torch_dtype=dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=dtype,
    )
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )
    generation_pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )
    radedit_pipeline = DiffusionPipeline.from_pipe(
        generation_pipeline,
        custom_pipeline="microsoft/radedit",
        trust_remote_code=True,
    )
    radedit_pipeline.to(device)
    return radedit_pipeline


def read_inputs(image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image, Image.Image, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    input_image = image.resize((512, 512), Image.Resampling.BICUBIC)
    edit_mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    edit_mask = edit_mask.point(lambda value: 255 if value > 0 else 0)
    keep_mask = ImageOps.invert(edit_mask)
    return input_image, edit_mask, keep_mask, original_size


def output_image(result) -> Image.Image:
    if hasattr(result, "images"):
        return result.images[0]
    if isinstance(result, (list, tuple)):
        return result[0]
    if isinstance(result, Image.Image):
        return result
    raise TypeError(f"Unexpected RadEdit output type: {type(result)!r}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    pipe = load_radedit_pipeline(args.device, torch_dtype(args.torch_dtype))
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    paths = []
    for index, row in df.iterrows():
        case_id = str(row[args.case_column])
        image, edit_mask, keep_mask, original_size = read_inputs(
            Path(str(row[args.image_column])),
            Path(str(row[args.mask_column])),
        )
        result = pipe(
            args.prompt,
            weights=[args.weight],
            image=image,
            edit_mask=edit_mask,
            keep_mask=keep_mask,
            num_inference_steps=args.num_inference_steps,
            invert_prompt="",
            skip_ratio=args.skip_ratio,
            output_type="pil",
        )
        edited = output_image(result).resize(original_size, Image.Resampling.BICUBIC)
        out_path = args.output_dir / f"{case_id}.png"
        edited.save(out_path)
        paths.append(str(out_path))
        print(f"[{index + 1}/{len(df)}] {case_id} -> {out_path}", flush=True)

    df = df.copy()
    df[args.counterfactual_column] = paths
    df["radedit_prompt"] = args.prompt
    df["radedit_weight"] = args.weight
    df["radedit_num_inference_steps"] = args.num_inference_steps
    df["radedit_skip_ratio"] = args.skip_ratio
    df.to_csv(args.output_manifest, index=False)
    print(f"Wrote manifest to {args.output_manifest}")


if __name__ == "__main__":
    main()
