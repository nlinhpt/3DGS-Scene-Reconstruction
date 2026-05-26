#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import re
import logging
import json
from argparse import ArgumentParser
import shutil
from pathlib import Path

from benchmark_sfm import evaluate_sfm, export_trajectory_tum

try:
    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        pairs_from_exhaustive,
        pairs_from_sequential,
        reconstruction,
    )
    import pycolmap
except ImportError:
    logging.error(
        "Install hloc and pycolmap first: pip install -e . "
        "(in the Hierarchical-Localization directory)"
    )
    exit(1)

# ============================================================
# Argument Parser
# ============================================================
parser = ArgumentParser("Colmap converter with LoMa and Matching Strategies")
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--skip_matching", action="store_true")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)

parser.add_argument(
    "--pairing",
    default="sequential", # Đã sửa thành sequential
    choices=["retrieval", "exhaustive", "sequential"],
    help="Image pairing strategy",
)
parser.add_argument(
    "--num_matched",
    default=50,
    type=int,
    help="Number of retrieved image pairs per image (for retrieval)",
)
parser.add_argument(
    "--overlap",
    default=10,
    type=int,
    help="Number of overlapping image pairs (for sequential). "
         "Should match temporal_window used in LK flow preprocessing.",
)
args = parser.parse_args()

colmap_command = (
    '"{}"'.format(args.colmap_executable)
    if len(args.colmap_executable) > 0
    else "colmap"
)
magick_command = (
    '"{}"'.format(args.magick_executable)
    if len(args.magick_executable) > 0
    else "magick"
)
use_gpu = 1 if not args.no_gpu else 0


# ============================================================
# Helper: stable numeric sort for image filenames
# ============================================================
def _numeric_key(p: str) -> int:
    m = re.search(r"(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0


if not args.skip_matching:

    # ── Path setup ───────────────────────────────────────────────────────
    source_path = Path(args.source_path)
    input_dir = source_path / "input"
    image_dir = source_path / "images"

    if image_dir.exists() and not input_dir.exists():
        print("--- [Setup] Renaming 'images' to 'input' for original frames ---")
        image_dir.rename(input_dir)

    if not input_dir.exists():
        logging.error(
            "Source directory does not contain an 'input' or 'images' folder!"
        )
        exit(1)

    output_dir = source_path / "distorted"
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_pairs = output_dir / f"pairs-{args.pairing}.txt"
    sfm_dir = output_dir / "sparse" / "0"
    
    # Đã sửa để tránh ghi đè khi chạy nhiều strategy khác nhau
    features = output_dir / f"features-{args.pairing}.h5"
    matches = output_dir / f"matches-{args.pairing}.h5"

    # Only used for retrieval pairing
    global_features_path = output_dir / "global_features.h5"

    feature_conf = extract_features.confs["loma_inloc"]
    matcher_conf = match_features.confs["loma"]

    # ── Build stable, sorted image list ─────────────────────────────────
    valid_ext = {".jpg", ".jpeg", ".png"}
    image_list_paths = sorted(
        [
            str(p.relative_to(input_dir))
            for p in input_dir.iterdir()
            if p.suffix.lower() in valid_ext
        ],
        key=_numeric_key,
    )

    if not image_list_paths:
        logging.error("No valid images found in %s", input_dir)
        exit(1)

    print(f"--- [Info] Found {len(image_list_paths)} images ---")

    # ============================================================
    # Image Pairing Strategy
    # ============================================================
    if args.pairing == "retrieval":
        print("--- [hloc] Extracting global features via NetVLAD ---")
        retrieval_conf = extract_features.confs["netvlad"]
        extract_features.main(
            retrieval_conf,
            input_dir,
            image_list=image_list_paths,
            feature_path=global_features_path,
        )
        print(f"--- [hloc] Generating image pairs from retrieval (k={args.num_matched}) ---")
        pairs_from_retrieval.main(
            global_features_path, sfm_pairs, num_matched=args.num_matched
        )

    elif args.pairing == "exhaustive":
        print("--- [hloc] Generating exhaustive image pairs ---")
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_list_paths)

    elif args.pairing == "sequential":
        print(f"--- [hloc] Generating sequential image pairs (overlap={args.overlap}) ---")
        pairs_from_sequential.main(
            sfm_pairs, image_list=image_list_paths, overlap=args.overlap
        )

    # Log pair count
    with open(sfm_pairs) as f:
        n_pairs = sum(1 for _ in f)
    print(f"--- [Info] {n_pairs} image pairs generated ({args.pairing} strategy) ---")

    # ============================================================
    # Local Feature Extraction & Matching
    # ============================================================
    print("--- [LoMa] Extracting local features (DaD detector + DeDoDe descriptor) ---")
    extract_features.main(feature_conf, input_dir, feature_path=features)

    print("--- [LoMa] Matching features via LightGlue ---")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    # ============================================================
    # COLMAP Incremental Mapping
    # ============================================================
    mapper_options = {
        "ba_global_function_tolerance": 1e-6,
        "ba_global_max_num_iterations": 100,
        # "tri_min_angle": 1.0,  # uncomment for sparse/textureless scenes
    }

    print("--- [hloc/COLMAP] Running Incremental Mapping & Global Bundle Adjustment ---")
    try:
        pycolmap.logging.set_log_level(pycolmap.logging.INFO)
    except AttributeError:
        pass

    reconstruction.main(
        sfm_dir,
        input_dir,
        sfm_pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.SINGLE,   # FIX: use enum, not string
        image_options={"camera_model": args.camera},
        mapper_options=mapper_options,              # FIX: was declared but never passed
        verbose=True,
    )

    # ── Sanity check: reconstruction non-empty ───────────────────────────
    recon_files = list(sfm_dir.glob("*.bin")) + list(sfm_dir.glob("*.txt"))
    if not recon_files:
        logging.error(
            "Reconstruction at %s appears empty. "
            "Check feature extraction and matching quality.",
            sfm_dir,
        )
        exit(1)

    # ============================================================
    # SfM Quality Evaluation
    # ============================================================
    print("--- [Benchmark] Evaluating SfM reconstruction quality ---")
    try:
        metrics = evaluate_sfm(sfm_dir, input_dir)
        print(metrics)
        with open(output_dir / "sfm_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logging.warning("SfM evaluation skipped (non-critical): %s", e)


# ============================================================
# Image Undistortion & Model Organization
# ============================================================
source_path = Path(args.source_path)
output_dir = source_path / "distorted"
input_model_path = output_dir / "sparse" / "0"

# FIX: guard against missing reconstruction when --skip_matching is used
if not input_model_path.exists():
    logging.error(
        "No reconstruction found at %s. "
        "Run without --skip_matching first.",
        input_model_path,
    )
    exit(1)

print("--- [COLMAP] Undistorting images ---")
img_undist_cmd = (
    colmap_command + " image_undistorter "
    f"--image_path {source_path / 'input'} "
    f"--input_path {input_model_path} "
    f"--output_path {args.source_path} "
    "--output_type COLMAP"
)

exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error("Undistorter failed with code %d. Exiting.", exit_code)
    exit(exit_code)

sparse_root = source_path / "sparse"
if sparse_root.exists():
    dest_proto = sparse_root / "0"
    dest_proto.mkdir(exist_ok=True)
    for file in os.listdir(sparse_root):
        if file == "0":
            continue
        src = sparse_root / file
        dst = dest_proto / file
        if src.is_file():
            shutil.move(str(src), str(dst))

# ============================================================
# Multi-scale Image Pyramid Generation
# ============================================================
if args.resize:
    print("Copying and resizing undistorted images...")
    undistorted_image_dir = source_path / "images"

    for scale, pct in [(2, "50%"), (4, "25%"), (8, "12.5%")]:
        scale_dir = source_path / f"images_{scale}"
        scale_dir.mkdir(exist_ok=True)
        for file in os.listdir(undistorted_image_dir):
            src = undistorted_image_dir / file
            dst = scale_dir / file
            shutil.copy2(src, dst)
            exit_code = os.system(f'{magick_command} mogrify -resize {pct} "{dst}"')
            if exit_code != 0:
                logging.error("%s resize failed with code %d. Exiting.", pct, exit_code)
                exit(exit_code)

# ============================================================
# Trajectory Export
# ============================================================
print("--- [Benchmark] Exporting camera trajectory in TUM format ---")
model_dir = source_path / "sparse" / "0"
tum_output = source_path / "trajectory.tum"


# NOTE — Quaternion order:
#   COLMAP lưu quaternion dưới dạng [qw, qx, qy, qz] (img.qvec).
#   TUM format yêu cầu  [tx ty tz qx qy qz qw]  (qw ở CUỐI).
#   benchmark_sfm.export_trajectory_tum đã xử lý đúng thứ tự này.
#   Nếu ATE/RPE từ evo hoặc TUM benchmark tools bị lớn bất thường,
#   kiểm tra lại thứ tự quaternion là nguyên nhân đầu tiên cần xem.

try:
    export_trajectory_tum(model_dir, tum_output)
except Exception as e:
    logging.warning("Trajectory export failed (non-critical): %s", e)

print("Done.")