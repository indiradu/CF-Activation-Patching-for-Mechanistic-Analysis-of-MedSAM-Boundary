"""Manifest construction and prompt-box utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


MANIFEST_COLUMNS = [
    "case_id",
    "image_id",
    "subject_id",
    "study_id",
    "dicom_id",
    "split",
    "view_position",
    "has_pleural_effusion",
    "has_no_finding",
    "image_path",
    "lung_mask_path",
    "mask_source",
    "counterfactual_path",
    "counterfactual_source",
    "box_xyxy",
    "notes",
]


def mimic_jpg_path(mimic_root: Path, subject_id: int | str, study_id: int | str, dicom_id: str) -> Path:
    subject = str(subject_id)
    study = str(study_id)
    prefix = subject[:2]
    return mimic_root / "files" / f"p{prefix}" / f"p{subject}" / f"s{study}" / f"{dicom_id}.jpg"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in metadata.columns:
        if col.lower() == "viewposition":
            rename[col] = "view_position"
    metadata = metadata.rename(columns=rename)
    required = {"subject_id", "study_id", "dicom_id", "view_position"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")
    return metadata


def _normalize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "study_id", "Pleural Effusion", "No Finding"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels missing required columns: {sorted(missing)}")
    return labels


def _load_mask_index(mask_index_csv: Path | None) -> pd.DataFrame | None:
    if mask_index_csv is None:
        return None
    mask_index = _read_csv(mask_index_csv)
    id_col = None
    for candidate in ("dicom_id", "image_id"):
        if candidate in mask_index.columns:
            id_col = candidate
            break
    if id_col is None or "mask_path" not in mask_index.columns:
        raise ValueError("Mask index must contain `dicom_id` or `image_id`, plus `mask_path`")
    mask_index = mask_index.rename(columns={id_col: "dicom_id", "mask_path": "lung_mask_path"})
    keep = ["dicom_id", "lung_mask_path"]
    if "mask_source" in mask_index.columns:
        keep.append("mask_source")
    else:
        mask_index["mask_source"] = "mask_index"
        keep.append("mask_source")
    return mask_index[keep]


def _select_cohort(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    pe = df["Pleural Effusion"].fillna(0).eq(1)
    nf = df["No Finding"].fillna(0).eq(1)
    if cohort == "pleural_effusion":
        return df[pe].copy()
    if cohort == "no_finding":
        return df[nf].copy()
    if cohort == "both":
        return df[pe | nf].copy()
    raise ValueError(f"Unknown cohort: {cohort}")


def build_mimic_manifest(
    *,
    mimic_root: Path,
    metadata_csv: Path,
    labels_csv: Path,
    output_csv: Path,
    split_csv: Path | None = None,
    mask_index_csv: Path | None = None,
    cohort: str = "pleural_effusion",
    views: Iterable[str] = ("AP", "PA"),
    limit: int | None = 50,
) -> pd.DataFrame:
    metadata = _normalize_metadata(_read_csv(metadata_csv))
    labels = _normalize_labels(_read_csv(labels_csv))
    df = metadata.merge(labels, on=["subject_id", "study_id"], how="inner")

    if split_csv is not None:
        split = _read_csv(split_csv)
        if {"dicom_id", "split"}.issubset(split.columns):
            df = df.merge(split[["dicom_id", "split"]], on="dicom_id", how="left")
        else:
            raise ValueError("Split CSV must contain `dicom_id` and `split`")
    else:
        df["split"] = "unspecified"

    view_set = {view.upper() for view in views}
    df["view_position"] = df["view_position"].fillna("").astype(str).str.upper()
    df = df[df["view_position"].isin(view_set)]
    df = _select_cohort(df, cohort)
    df = df.sort_values(["subject_id", "study_id", "dicom_id"])

    if limit is not None:
        df = df.head(limit)
    df = df.reset_index(drop=True)

    manifest = pd.DataFrame()
    manifest["case_id"] = [f"mimic_{dicom_id}" for dicom_id in df["dicom_id"].astype(str)]
    manifest["image_id"] = df["dicom_id"]
    manifest["subject_id"] = df["subject_id"]
    manifest["study_id"] = df["study_id"]
    manifest["dicom_id"] = df["dicom_id"]
    manifest["split"] = df["split"]
    manifest["view_position"] = df["view_position"]
    manifest["has_pleural_effusion"] = df["Pleural Effusion"].fillna(0).eq(1).astype(int)
    manifest["has_no_finding"] = df["No Finding"].fillna(0).eq(1).astype(int)
    manifest["image_path"] = [
        str(mimic_jpg_path(mimic_root, row.subject_id, row.study_id, row.dicom_id))
        for row in df.itertuples(index=False)
    ]
    manifest["lung_mask_path"] = ""
    manifest["mask_source"] = ""
    manifest["counterfactual_path"] = ""
    manifest["counterfactual_source"] = ""
    manifest["box_xyxy"] = ""
    manifest["notes"] = ""

    mask_index = _load_mask_index(mask_index_csv)
    if mask_index is not None:
        manifest = manifest.merge(mask_index, on="dicom_id", how="left", suffixes=("", "_from_index"))
        manifest["lung_mask_path"] = manifest["lung_mask_path_from_index"].fillna("")
        manifest["mask_source"] = manifest["mask_source_from_index"].fillna("")
        manifest = manifest.drop(columns=["lung_mask_path_from_index", "mask_source_from_index"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest[MANIFEST_COLUMNS]
    manifest.to_csv(output_csv, index=False)
    return manifest


def _nih_image_map(image_root: Path, image_ids: set[str]) -> dict[str, Path]:
    direct = {image_id: image_root / image_id for image_id in image_ids if (image_root / image_id).exists()}
    missing = image_ids.difference(direct)
    if not missing:
        return direct

    found = dict(direct)
    for path in image_root.rglob("*"):
        if path.name in missing and path.is_file():
            found[path.name] = path
            if len(found) == len(image_ids):
                break
    return found


def _load_nih_mask_index(mask_index_csv: Path | None) -> pd.DataFrame | None:
    if mask_index_csv is None:
        return None
    mask_index = _read_csv(mask_index_csv)
    rename = {}
    for col in mask_index.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized in {"image_index", "image_id"}:
            rename[col] = "image_id"
        if normalized == "mask_path":
            rename[col] = "lung_mask_path"
    mask_index = mask_index.rename(columns=rename)
    if "image_id" not in mask_index.columns or "lung_mask_path" not in mask_index.columns:
        raise ValueError("NIH mask index must contain `image_id` or `Image Index`, plus `mask_path`")
    if "mask_source" not in mask_index.columns:
        mask_index["mask_source"] = "mask_index"
    return mask_index[["image_id", "lung_mask_path", "mask_source"]]


def build_nih_chestxray14_manifest(
    *,
    image_root: Path,
    labels_csv: Path,
    output_csv: Path,
    mask_index_csv: Path | None = None,
    cohort: str = "effusion",
    views: Iterable[str] = ("AP", "PA"),
    limit: int | None = 50,
) -> pd.DataFrame:
    labels = _read_csv(labels_csv)
    rename = {}
    for col in labels.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized == "image_index":
            rename[col] = "image_id"
        elif normalized == "finding_labels":
            rename[col] = "finding_labels"
        elif normalized == "patient_id":
            rename[col] = "subject_id"
        elif normalized == "view_position":
            rename[col] = "view_position"
    labels = labels.rename(columns=rename)
    required = {"image_id", "finding_labels", "subject_id", "view_position"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"NIH labels missing required columns: {sorted(missing)}")

    labels["view_position"] = labels["view_position"].fillna("").astype(str).str.upper()
    view_set = {view.upper() for view in views}
    labels = labels[labels["view_position"].isin(view_set)].copy()

    finding_labels = labels["finding_labels"].fillna("").astype(str)
    has_effusion = finding_labels.str.contains("Effusion", regex=False)
    has_no_finding = finding_labels.str.contains("No Finding", regex=False)
    if cohort == "effusion":
        labels = labels[has_effusion].copy()
    elif cohort == "no_finding":
        labels = labels[has_no_finding].copy()
    elif cohort == "both":
        labels = labels[has_effusion | has_no_finding].copy()
    else:
        raise ValueError(f"Unknown cohort: {cohort}")

    labels = labels.sort_values(["subject_id", "image_id"])
    if limit is not None:
        labels = labels.head(limit)
    labels = labels.reset_index(drop=True)

    image_ids = set(labels["image_id"].astype(str))
    image_map = _nih_image_map(image_root, image_ids)

    manifest = pd.DataFrame()
    manifest["case_id"] = [f"nih_{Path(image_id).stem}" for image_id in labels["image_id"].astype(str)]
    manifest["image_id"] = labels["image_id"]
    manifest["subject_id"] = labels["subject_id"]
    manifest["study_id"] = ""
    manifest["dicom_id"] = ""
    manifest["split"] = "unspecified"
    manifest["view_position"] = labels["view_position"]
    manifest["has_pleural_effusion"] = labels["finding_labels"].fillna("").str.contains("Effusion", regex=False).astype(int)
    manifest["has_no_finding"] = labels["finding_labels"].fillna("").str.contains("No Finding", regex=False).astype(int)
    manifest["image_path"] = [str(image_map.get(str(image_id), "")) for image_id in labels["image_id"]]
    manifest["lung_mask_path"] = ""
    manifest["mask_source"] = ""
    manifest["counterfactual_path"] = ""
    manifest["counterfactual_source"] = ""
    manifest["box_xyxy"] = ""
    manifest["notes"] = np.where(manifest["image_path"].eq(""), "image_path_not_found", "")

    mask_index = _load_nih_mask_index(mask_index_csv)
    if mask_index is not None:
        manifest = manifest.merge(mask_index, on="image_id", how="left", suffixes=("", "_from_index"))
        manifest["lung_mask_path"] = manifest["lung_mask_path_from_index"].fillna("")
        manifest["mask_source"] = manifest["mask_source_from_index"].fillna("")
        manifest = manifest.drop(columns=["lung_mask_path_from_index", "mask_source_from_index"])

    manifest = manifest[manifest["image_path"].ne("")]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = manifest[MANIFEST_COLUMNS]
    manifest.to_csv(output_csv, index=False)
    return manifest


def parse_box_xyxy(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if len(parts) != 4:
            raise ValueError(f"Expected four box coordinates, got {value!r}")
        return tuple(int(round(float(part))) for part in parts)  # type: ignore[return-value]
    if isinstance(value, (tuple, list)) and len(value) == 4:
        return tuple(int(round(float(part))) for part in value)  # type: ignore[return-value]
    raise ValueError(f"Cannot parse box from {value!r}")


def mask_to_box_xyxy(mask: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot create a bounding box from an empty mask")
    height, width = mask.shape[:2]
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(width - 1, int(xs.max()) + margin)
    y2 = min(height - 1, int(ys.max()) + margin)
    return x1, y1, x2, y2


def _read_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def add_boxes_from_masks(
    df: pd.DataFrame,
    *,
    mask_column: str = "lung_mask_path",
    box_column: str = "box_xyxy",
    margin: int = 12,
) -> pd.DataFrame:
    boxes = []
    for value in df[mask_column]:
        if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
            boxes.append("")
            continue
        box = mask_to_box_xyxy(_read_mask(Path(str(value))), margin=margin)
        boxes.append(",".join(str(v) for v in box))
    out = df.copy()
    out[box_column] = boxes
    return out
