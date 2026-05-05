"""Image enhancement: CLAHE on LAB color space + optional bilateral denoise."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame: np.ndarray, denoise: bool = False) -> np.ndarray:
    """Apply CLAHE on the L channel of LAB; optionally bilateral-filter the output.

    Handles harsh outdoor lighting, dust, and glare on construction sites.
    """
    if frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if denoise:
        out = cv2.bilateralFilter(out, d=9, sigmaColor=75, sigmaSpace=75)
    return out


def _cli(path: str) -> None:
    src = Path(path)
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(path)
    out = enhance(img, denoise=True)
    dst = src.with_name(f"{src.stem}_enhanced{src.suffix}")
    cv2.imwrite(str(dst), out)
    print(f"wrote {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m cv_pipeline.enhancement <image_path>", file=sys.stderr)
        sys.exit(1)
    _cli(sys.argv[1])
