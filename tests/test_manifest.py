from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from counterfactual_medsam.manifest import (
    build_nih_chestxray14_manifest,
    mask_to_box_xyxy,
    mimic_jpg_path,
    parse_box_xyxy,
)


def test_mimic_jpg_path() -> None:
    path = mimic_jpg_path(Path("/mimic"), 10000032, 50414267, "abc")
    assert path == Path("/mimic/files/p10/p10000032/s50414267/abc.jpg")


def test_parse_box_xyxy() -> None:
    assert parse_box_xyxy("1,2,3,4") == (1, 2, 3, 4)
    assert parse_box_xyxy("1 2 3 4") == (1, 2, 3, 4)


def test_mask_to_box_xyxy_with_margin() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[3:6, 4:9] = True
    assert mask_to_box_xyxy(mask, margin=1) == (3, 2, 9, 6)


def test_build_nih_chestxray14_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_root / "0001.png")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_root / "0002.png")

    labels_csv = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(
        [
            {
                "Image Index": "0001.png",
                "Finding Labels": "Effusion",
                "Patient ID": 1,
                "View Position": "PA",
            },
            {
                "Image Index": "0002.png",
                "Finding Labels": "No Finding",
                "Patient ID": 2,
                "View Position": "PA",
            },
        ]
    ).to_csv(labels_csv, index=False)

    output_csv = tmp_path / "manifest.csv"
    manifest = build_nih_chestxray14_manifest(
        image_root=image_root,
        labels_csv=labels_csv,
        output_csv=output_csv,
        cohort="effusion",
    )

    assert len(manifest) == 1
    assert manifest.iloc[0]["case_id"] == "nih_0001"
    assert manifest.iloc[0]["has_pleural_effusion"] == 1
