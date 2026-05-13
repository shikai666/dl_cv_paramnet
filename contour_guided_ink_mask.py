# -*- coding: utf-8 -*-
"""
Contour-guided ink recovery (overlap-mask only)

Use the model's raw contour as a cue to select plausible ink components
from the original grayscale image.

Supported contour filename formats:
  - 1741_raw_contour.png
  - 1741_contour.png

Ignored files:
  - *_overlay.png
  - other unrelated files

Inputs:
  - original images directory (grayscale / infrared images)
  - raw contour directory containing files like:
      1741_raw_contour.png
      or
      1741_contour.png

Outputs:
  - *_overlap_mask.png : evaluation mask used for comparison with GT
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np


VALID_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def find_image_by_stem(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in VALID_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p

        p2 = image_dir / f"{stem}{ext.upper()}"
        if p2.exists():
            return p2

    return None


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def make_ink_mask(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Returns a loose ink candidate mask (uint8, 0/1).
    Assumes text is darker than background.
    """
    if method == "otsu":
        _, th = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return (th > 0).astype(np.uint8)

    if method == "adaptive":
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            11,
        )
        return (th > 0).astype(np.uint8)

    raise ValueError(f"Unknown ink method: {method}")


def build_contour_band(raw_contour_u8: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    """
    Convert raw contour image to binary band mask, with optional dilation.
    """
    band = (raw_contour_u8 > 127).astype(np.uint8)

    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        work = cv2.dilate(band * 255, k, iterations=int(dilate_iter))
        band = (work > 0).astype(np.uint8)

    return band


def keep_ink_components_supported_by_contour(
    ink01: np.ndarray,
    contour_band01: np.ndarray,
    min_area: int = 8,
    min_overlap_pixels: int = 5,
    min_overlap_ratio: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only ink connected components that are supported by the contour band.

    A component is kept if:
      overlap_pixels >= min_overlap_pixels
      OR
      overlap_pixels / component_area >= min_overlap_ratio

    Returns:
      keep_mask01, overlap_mask01
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink01.astype(np.uint8), connectivity=8
    )

    keep = np.zeros_like(ink01, dtype=np.uint8)
    overlap_mask = np.zeros_like(ink01, dtype=np.uint8)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue

        comp = (labels == i).astype(np.uint8)
        overlap = int((comp & contour_band01).sum())
        overlap_ratio = float(overlap) / float(max(1, area))

        if overlap >= int(min_overlap_pixels) or overlap_ratio >= float(min_overlap_ratio):
            keep[comp > 0] = 1

            ov = (comp & contour_band01).astype(np.uint8)
            overlap_mask[ov > 0] = 1

    return keep, overlap_mask


def recover_overlap_mask(
    gray: np.ndarray,
    raw_contour_u8: np.ndarray,
    ink_method: str = "otsu",
    contour_band_dilate_iter: int = 1,
    comp_min_area: int = 8,
    min_overlap_pixels: int = 5,
    min_overlap_ratio: float = 0.03,
) -> np.ndarray:
    """
    Generate overlap mask only.
    """
    ink01 = make_ink_mask(gray, method=ink_method)
    contour_band01 = build_contour_band(
        raw_contour_u8, dilate_iter=contour_band_dilate_iter
    )

    _, overlap01 = keep_ink_components_supported_by_contour(
        ink01=ink01,
        contour_band01=contour_band01,
        min_area=comp_min_area,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_ratio=min_overlap_ratio,
    )

    return overlap01


def extract_stem_from_contour_filename(contour_path: Path) -> Optional[str]:
    """
    Support:
      xxx_raw_contour.png -> xxx
      xxx_contour.png     -> xxx

    Ignore:
      xxx_overlay.png
      others
    """
    stem = contour_path.stem

    if stem.endswith("_raw_contour"):
        return stem[: -len("_raw_contour")]

    if stem.endswith("_contour"):
        return stem[: -len("_contour")]

    return None


def list_contour_files(contour_dir: Path) -> List[Path]:
    """
    Collect supported contour files:
      *_raw_contour.png
      *_contour.png

    Exclude duplicates when both exist for the same stem,
    preferring *_raw_contour.png.
    """
    raw_files = sorted(contour_dir.glob("*_raw_contour.png"))
    contour_files = sorted(contour_dir.glob("*_contour.png"))

    selected = {}
    for p in raw_files:
        stem = extract_stem_from_contour_filename(p)
        if stem is not None:
            selected[stem] = p

    for p in contour_files:
        stem = extract_stem_from_contour_filename(p)
        if stem is not None and stem not in selected:
            selected[stem] = p

    return [selected[k] for k in sorted(selected.keys())]


def process_one(
    image_path: Path,
    contour_path: Path,
    output_dir: Path,
    ink_method: str,
    contour_band_dilate_iter: int,
    comp_min_area: int,
    min_overlap_pixels: int,
    min_overlap_ratio: float,
) -> None:
    gray = load_gray(image_path)
    raw_contour = load_gray(contour_path)

    overlap01 = recover_overlap_mask(
        gray=gray,
        raw_contour_u8=raw_contour,
        ink_method=ink_method,
        contour_band_dilate_iter=contour_band_dilate_iter,
        comp_min_area=comp_min_area,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_ratio=min_overlap_ratio,
    )

    stem = extract_stem_from_contour_filename(contour_path)
    if stem is None:
        raise ValueError(f"Unsupported contour filename: {contour_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_overlap_mask.png"), overlap01 * 255)


def process_dir(
    image_dir: Path,
    contour_dir: Path,
    output_dir: Path,
    ink_method: str,
    contour_band_dilate_iter: int,
    comp_min_area: int,
    min_overlap_pixels: int,
    min_overlap_ratio: float,
) -> None:
    contour_paths = list_contour_files(contour_dir)
    if not contour_paths:
        raise FileNotFoundError(
            f"No supported contour files found in {contour_dir}. "
            f"Expected '*_raw_contour.png' or '*_contour.png'."
        )

    missing: List[str] = []
    count = 0

    for contour_path in contour_paths:
        stem = extract_stem_from_contour_filename(contour_path)
        if stem is None:
            continue

        image_path = find_image_by_stem(image_dir, stem)

        if image_path is None:
            missing.append(stem)
            continue

        process_one(
            image_path=image_path,
            contour_path=contour_path,
            output_dir=output_dir,
            ink_method=ink_method,
            contour_band_dilate_iter=contour_band_dilate_iter,
            comp_min_area=comp_min_area,
            min_overlap_pixels=min_overlap_pixels,
            min_overlap_ratio=min_overlap_ratio,
        )
        count += 1

    print(f"Done. processed={count}, output_dir={output_dir}")

    if missing:
        print("Missing original images for these stems:")
        for s in missing[:20]:
            print(" -", s)
        if len(missing) > 20:
            print(f" ... and {len(missing) - 20} more")


def main() -> None:
    ap = argparse.ArgumentParser("Generate overlap-only masks for evaluation")
    ap.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="directory with original grayscale / infrared images",
    )
    ap.add_argument(
        "--contour_dir",
        type=str,
        required=True,
        help="directory with contour files: '*_raw_contour.png' or '*_contour.png'",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory to save *_overlap_mask.png",
    )
    ap.add_argument(
        "--ink_method",
        type=str,
        default="otsu",
        choices=["otsu", "adaptive"],
    )
    ap.add_argument("--contour_band_dilate_iter", type=int, default=1)
    ap.add_argument("--comp_min_area", type=int, default=8)
    ap.add_argument("--min_overlap_pixels", type=int, default=5)
    ap.add_argument("--min_overlap_ratio", type=float, default=0.03)

    args = ap.parse_args()

    process_dir(
        image_dir=Path(args.image_dir),
        contour_dir=Path(args.contour_dir),
        output_dir=Path(args.output_dir),
        ink_method=args.ink_method,
        contour_band_dilate_iter=args.contour_band_dilate_iter,
        comp_min_area=args.comp_min_area,
        min_overlap_pixels=args.min_overlap_pixels,
        min_overlap_ratio=args.min_overlap_ratio,
    )


if __name__ == "__main__":
    main()
