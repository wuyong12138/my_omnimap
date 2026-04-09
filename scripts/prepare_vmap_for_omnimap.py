from __future__ import annotations

import re
import zipfile
from collections import Counter
from io import TextIOWrapper
from pathlib import Path

import numpy as np
from PIL import Image


# Edit these paths on the target machine as needed.
SCENE_ROOT = Path(r"D:\pycharm\omnimap_code\vamp")
SCENE_NAME = SCENE_ROOT.name
RGB_DIR = SCENE_ROOT / "imap" / "00" / "rgb"
DEPTH_DIR = SCENE_ROOT / "imap" / "00" / "depth"
POSE_ZIP = SCENE_ROOT / "imap" / "00" / "camera_poses.zip"
POSE_TXT = SCENE_ROOT / "imap" / "00" / "traj_w_c.txt"

RGB_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEPTH_EXTENSIONS = {".png"}
INVALID_DEPTH_THRESHOLD = 65000


def numeric_stem(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def list_numbered_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    return sorted(files, key=lambda path: (numeric_stem(path) is None, numeric_stem(path), path.name))


def format_index_preview(indices: set[int], limit: int = 10) -> str:
    if not indices:
        return "[]"
    ordered = sorted(indices)
    preview = ordered[:limit]
    suffix = "" if len(ordered) <= limit else " ..."
    return "[" + ", ".join(str(value) for value in preview) + suffix + "]"


def parse_pose_matrix(text: str, source_name: str) -> np.ndarray:
    values = np.fromstring(text, sep=" ")
    if values.size != 16:
        raise ValueError(f"Expected 16 values in {source_name}, got {values.size}")
    return values.reshape(4, 4)


def write_traj_from_zip(zip_path: Path, output_path: Path) -> int:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Missing pose zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        text_names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not text_names:
            raise FileNotFoundError(f"No .txt pose file found inside {zip_path}")
        text_names.sort(key=lambda name: (numeric_stem(Path(name)) is None, numeric_stem(Path(name)), name))

        pose_matrices: list[np.ndarray] = []
        for text_name in text_names:
            with archive.open(text_name) as handle:
                text = TextIOWrapper(handle, encoding="utf-8").read()
            pose_matrices.append(parse_pose_matrix(text, text_name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for matrix in pose_matrices:
            flat = matrix.reshape(-1)
            handle.write(" ".join(f"{value:.12g}" for value in flat))
            handle.write("\n")
    return len(pose_matrices)


def read_pose_count(pose_txt: Path) -> int:
    if not pose_txt.is_file():
        raise FileNotFoundError(f"Missing pose txt: {pose_txt}")
    with pose_txt.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def inspect_depth_files(depth_files: list[Path]) -> tuple[int, int, Counter[str]]:
    invalid_pixels = 0
    total_pixels = 0
    mode_counter: Counter[str] = Counter()

    for path in depth_files:
        with Image.open(path) as image:
            array = np.array(image)
        mode_counter[str(array.dtype)] += 1
        total_pixels += int(array.size)
        invalid_pixels += int(np.count_nonzero((array == 0) | (array >= INVALID_DEPTH_THRESHOLD)))

    return invalid_pixels, total_pixels, mode_counter


def main() -> None:
    pose_count = write_traj_from_zip(POSE_ZIP, POSE_TXT)

    rgb_files = list_numbered_files(RGB_DIR, RGB_EXTENSIONS)
    depth_files = list_numbered_files(DEPTH_DIR, DEPTH_EXTENSIONS)
    rgb_indices = {numeric_stem(path) for path in rgb_files if numeric_stem(path) is not None}
    depth_indices = {numeric_stem(path) for path in depth_files if numeric_stem(path) is not None}
    pose_line_count = read_pose_count(POSE_TXT)

    missing_in_depth = rgb_indices - depth_indices
    missing_in_rgb = depth_indices - rgb_indices

    invalid_pixels, total_pixels, mode_counter = inspect_depth_files(depth_files)
    invalid_ratio = (invalid_pixels / total_pixels) if total_pixels else 0.0

    print("=== OmniMap vmap preparation report ===")
    print(f"scene: {SCENE_NAME}")
    print(f"rgb_dir: {RGB_DIR}")
    print(f"depth_dir: {DEPTH_DIR}")
    print(f"pose_zip: {POSE_ZIP}")
    print(f"pose_txt: {POSE_TXT}")
    print()
    print(f"traj_w_c.txt rows written: {pose_count}")
    print(f"rgb files: {len(rgb_files)}")
    print(f"depth files: {len(depth_files)}")
    print(f"pose rows: {pose_line_count}")
    print(f"count_match: {len(rgb_files) == len(depth_files) == pose_line_count}")
    print(f"rgb_depth_index_match: {rgb_indices == depth_indices}")
    if missing_in_depth:
        print(f"missing_in_depth: {format_index_preview(missing_in_depth)}")
    if missing_in_rgb:
        print(f"missing_in_rgb: {format_index_preview(missing_in_rgb)}")
    print(f"depth_dtypes: {dict(mode_counter)}")
    print(f"invalid_depth_pixels: {invalid_pixels}")
    print(f"invalid_depth_ratio: {invalid_ratio:.6f}")


if __name__ == "__main__":
    main()
