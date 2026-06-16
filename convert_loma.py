"""
convert.py — LoMa-based SfM preprocessing pipeline for 3D Gaussian Splatting.

Pipeline overview:
    Phase 1 — Image pairing → Local feature extraction (LoMa) →
              Feature matching (LightGlue) → Sparse reconstruction (COLMAP)
    Phase 2 — Image undistortion → produces images/ and sparse/0/ for 3DGS
    Phase 3 — Camera trajectory export in TUM format for evaluation
    Phase 4 — (optional) Multi-scale image pyramid for coarse-to-fine 3DGS training

Output structure expected by 3DGS:
    source_path/
    ├── images/          ← undistorted images (3DGS reads these)
    └── sparse/
        └── 0/
            ├── cameras.bin
            ├── images.bin   ← image names MUST match filenames in images/
            └── points3D.bin

Copyright (C) 2023, Inria — GRAPHDECO research group
License: see LICENSE.md (non-commercial, research and evaluation use only)
Contact: george.drettakis@inria.fr
"""

import os
import re
import logging
import json
import shutil
from argparse import ArgumentParser
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
        "hloc and pycolmap are required. "
        "Install with: pip install -e .  (inside Hierarchical-Localization/)"
    )
    exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Argument Parser
# ═══════════════════════════════════════════════════════════════════════

parser = ArgumentParser("Colmap converter with LoMa")

parser.add_argument("--no_gpu", action="store_true",
                    help="Disable GPU for feature extraction.")

parser.add_argument("--skip_matching", action="store_true",
                    help="Skip Phase 1 entirely (pairing, extraction, matching, "
                         "reconstruction). Assumes distorted/sparse/0/ already exists "
                         "from a previous run. Useful when re-running only undistortion "
                         "or evaluation.")

parser.add_argument("--source_path", "-s", required=True, type=str,
                    help="Root directory of the scene. Must contain an 'images/' or "
                         "'input/' subfolder with raw frames.")

parser.add_argument("--camera", default="OPENCV", type=str,
                    help="COLMAP camera model used during reconstruction. "
                         "Options: SIMPLE_PINHOLE, PINHOLE, OPENCV (default), "
                         "OPENCV_FISHEYE, FULL_OPENCV. "
                         "OPENCV models radial + tangential distortion and is "
                         "recommended for most smartphone/DSLR footage.")

parser.add_argument("--colmap_executable", default="", type=str,
                    help="Full path to the COLMAP binary. Leave empty to use the "
                         "'colmap' command from PATH.")

parser.add_argument("--resize", action="store_true",
                    help="Generate multi-scale image pyramids (images_2/, images_4/, "
                         "images_8/) from the undistorted images. Useful for "
                         "coarse-to-fine 3DGS training.")

parser.add_argument("--magick_executable", default="", type=str,
                    help="Full path to the ImageMagick binary. Leave empty to use "
                         "'magick' from PATH.")

parser.add_argument(
    "--pairing",
    default="sequential",
    choices=["sequential", "exhaustive", "retrieval"],
    help=(
        "Image pair generation strategy for feature matching.\n"
        "\n"
        "  sequential  (default)\n"
        "    Pairs each image with its --overlap nearest neighbours in the sorted\n"
        "    list. O(n * overlap) pairs total. Best for video sequences where\n"
        "    consecutive frames share high visual overlap. Requires images to be\n"
        "    named with numeric indices (e.g. frame_001.jpg) so they sort correctly.\n"
        "\n"
        "  exhaustive\n"
        "    Pairs every image with every other image. O(n²) pairs total.\n"
        "    Highest recall but very slow for large datasets (>200 images).\n"
        "    Recommended for small unordered photo collections.\n"
        "\n"
        "  retrieval\n"
        "    Uses NetVLAD global descriptors to retrieve the --num_matched most\n"
        "    visually similar images for each query. O(n * num_matched) pairs.\n"
        "    Good balance between recall and speed for large unordered datasets.\n"
        "    Requires an extra NetVLAD feature extraction pass.\n"
    ),
)

parser.add_argument(
    "--num_matched",
    default=50,
    type=int,
    help="Number of image pairs retrieved per query image. "
         "Only used when --pairing retrieval is selected. "
         "Higher values improve recall at the cost of matching time.",
)

parser.add_argument(
    "--overlap",
    default=10,
    type=int,
    help="Number of neighbouring frames to pair with each image. "
         "Only used when --pairing sequential is selected. "
         "Should be >= the temporal window used in any LK-flow preprocessing step. "
         "Larger values improve robustness against motion blur or low-texture frames "
         "but increase matching time.",
)

args = parser.parse_args()

# Build shell commands — quote paths to handle spaces
colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"

source_path = Path(args.source_path)


# ═══════════════════════════════════════════════════════════════════════
# Input directory normalisation
#
# 3DGS expects its training images at  source_path/images/  after the
# pipeline finishes. COLMAP image_undistorter writes its output to that
# same path. To avoid overwriting the original frames we rename the raw
# image folder from  images/  to  input/  before doing anything else.
#
# If the caller already placed frames in  input/  this step is a no-op.
# ═══════════════════════════════════════════════════════════════════════

image_dir = source_path / "images"   # conventional name used by many datasets
input_dir = source_path / "input"    # name we keep raw frames under throughout

if image_dir.exists() and not input_dir.exists():
    print("--- [Setup] Renaming 'images' → 'input' to avoid COLMAP output conflict ---")
    image_dir.rename(input_dir)

if not input_dir.exists():
    logging.error(
        "No 'input/' or 'images/' folder found under %s. "
        "Place raw frames there before running.", source_path
    )
    exit(1)


# ═══════════════════════════════════════════════════════════════════════
# Path constants
#
# Keep all path definitions here so the relationships between Phase 1–4
# outputs are easy to audit at a glance.
# ═══════════════════════════════════════════════════════════════════════

# Working directory for intermediate SfM artefacts (features, matches, sparse model)
output_dir = source_path / "distorted"

# Sparse model produced by LoMa + COLMAP, still in the *original* (distorted)
# camera coordinate system. This is the authoritative reconstruction used for:
#   - trajectory export (contains ALL registered images)
#   - image undistortion input
sfm_dir = output_dir / "sparse" / "0"

# After undistortion, COLMAP writes a *new* sparse model and the undistorted
# images here. These two paths are what 3DGS reads at train time.
final_sparse_dir = source_path / "sparse" / "0"   # undistorted sparse model
final_images_dir = source_path / "images"          # undistorted images


# ═══════════════════════════════════════════════════════════════════════
# Helper: stable numeric sort for image filenames
#
# iterdir() returns files in arbitrary filesystem order. Sequential pairing
# is order-sensitive: frame_002 must come after frame_001. This key extracts
# the first integer found in the filename stem so that sort() produces the
# correct temporal order regardless of zero-padding or prefix strings.
#
# Examples:
#   "frame_001.jpg"  → 1
#   "img042.png"     → 42
#   "DSC_0007.JPG"   → 7
# ═══════════════════════════════════════════════════════════════════════

def _numeric_key(p: str) -> int:
    m = re.search(r"(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — Feature extraction, matching, sparse reconstruction
#
# Steps:
#   1. Build a sorted image list from input/ (numeric order).
#   2. Generate image pairs according to the chosen --pairing strategy.
#   3. Extract LoMa local features for every image in the list.
#      LoMa combines:
#        • DaD  (Descriptor-and-Detector) — learned keypoint detector
#        • DeDoDe descriptor               — learned patch descriptor
#      Together they outperform SIFT on textureless / repetitive surfaces.
#   4. Match features between all generated pairs using LightGlue.
#      LightGlue uses a transformer attention mechanism to match keypoints
#      with global context, producing far fewer outliers than ratio-test NN.
#   5. Run COLMAP incremental mapper (via hloc) to build the sparse model.
#      Bundle Adjustment options are tuned for improved convergence:
#        • ba_global_function_tolerance 1e-6 — early stop when gain < 1e-6
#          (prevents over-fitting to noise while saving compute)
#        • ba_global_max_num_iterations 100  — more iterations than the
#          COLMAP default (50) for better accuracy on large/complex scenes
# ═══════════════════════════════════════════════════════════════════════

if not args.skip_matching:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-strategy file names prevent accidental cache reuse when switching
    # between pairing modes on the same scene.
    sfm_pairs       = output_dir / f"pairs-{args.pairing}.txt"
    features        = output_dir / f"features-{args.pairing}.h5"
    matches         = output_dir / f"matches-{args.pairing}.h5"
    global_features = output_dir / "global_features.h5"  # only written for retrieval

    # ── Feature extractor & matcher configs ─────────────────────────────
    #
    # loma_inloc : resize_max=1600, suited for indoor / high-res footage.
    # loma_aachen: resize_max=1024, suited for outdoor / lower-res footage.
    # Both configs use the same DaD+DeDoDe extractor; only the preprocessing
    # resolution differs.  The extractor implementation lives in:
    #   hloc/extractors/loma.py
    feature_conf = extract_features.confs["loma_inloc"]

    # loma matcher — uses LightGlue as the matching backend.
    # To switch to a heavier LoMa variant (larger model, higher recall):
    #   matcher_conf = {
    #       "output": "matches-loma-r",
    #       "model": {"name": "loma", "arch": "LoMa-R"},
    #   }
    # Available architectures: LoMa-B (default) | LoMa-L | LoMa-G | LoMa-R
    
    # matcher_conf = match_features.confs["loma"]
    matcher_conf = {
    "model": {
        "name": "loma",
        "arch": "LoMa-B",
        "filter_threshold": 0.4, 
    },
}

    # ── Build image list ─────────────────────────────────────────────────
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

    # ── Image pair generation ────────────────────────────────────────────
    #
    # The pairs file lists one (image_a, image_b) per line and controls
    # which image combinations will be feature-matched. Choosing the right
    # strategy is a trade-off between matching recall and computation time.
    if args.pairing == "retrieval":
        # Step 1: extract compact global descriptors (NetVLAD) for each image.
        # Step 2: for each image, retrieve the k most similar images by
        #         global descriptor distance → write those pairs to sfm_pairs.
        # Recommended for large unordered datasets (>500 images) where
        # exhaustive matching would be prohibitively slow.
        print("--- [hloc] Extracting global features (NetVLAD) for retrieval ---")
        retrieval_conf = extract_features.confs["netvlad"]
        extract_features.main(
            retrieval_conf, input_dir,
            image_list=image_list_paths,
            feature_path=global_features,
        )
        print(f"--- [hloc] Retrieving top-{args.num_matched} pairs per image ---")
        pairs_from_retrieval.main(
            global_features, sfm_pairs, num_matched=args.num_matched
        )

    elif args.pairing == "exhaustive":
        # Every image is paired with every other image → O(n²) pairs.
        # Highest recall. Only feasible for small datasets (<200 images).
        print("--- [hloc] Generating exhaustive pairs ---")
        pairs_from_exhaustive.main(sfm_pairs, image_list=image_list_paths)

    else:  # sequential (default)
        # Each image is paired with --overlap neighbours in the sorted list.
        # O(n * overlap) pairs. Ideal for video frames where temporally
        # close images share high visual overlap.
        # IMPORTANT: image list must be in correct temporal order — ensured
        # by _numeric_key above.
        print(f"--- [hloc] Generating sequential pairs (overlap={args.overlap}) ---")
        pairs_from_sequential.main(
            sfm_pairs,
            image_list=image_list_paths,
            overlap=args.overlap,
        )

    with open(sfm_pairs) as f:
        n_pairs = sum(1 for _ in f)
    print(f"--- [Info] {n_pairs} pairs generated ({args.pairing} strategy) ---")

    # ── Local feature extraction ─────────────────────────────────────────
    #
    # Passing image_list ensures only the sorted, validated set of images
    # is processed. Without it, hloc would scan the entire input_dir and
    # might silently fail on unexpected files, producing incomplete features
    # that cause COLMAP to skip those images during reconstruction.
    print("--- [LoMa] Extracting local features (DaD detector + DeDoDe descriptor) ---")
    extract_features.main(
        feature_conf, input_dir,
        image_list=image_list_paths,
        feature_path=features,
    )

    # ── Feature matching ─────────────────────────────────────────────────
    #
    # Matches are computed only for image pairs listed in sfm_pairs.
    # LightGlue reads both feature sets jointly, using cross-attention to
    # produce context-aware matches with built-in outlier filtering.
    print("--- [LoMa] Matching features via LightGlue ---")
    match_features.main(
        matcher_conf, sfm_pairs,
        features=features,
        matches=matches,
    )

    # ── Sparse reconstruction ────────────────────────────────────────────
    #
    # hloc wraps the COLMAP incremental mapper. It:
    #   1. Imports features and matches into a COLMAP database.
    #   2. Runs incremental SfM: registers images one by one, triangulates
    #      new 3D points, and periodically runs local + global BA.
    #   3. Writes the final sparse model (cameras/images/points3D) to sfm_dir.
    #
    # camera_mode=SINGLE: all images share one camera model. Correct for
    #   footage from a single device with fixed focal length.
    #
    # mapper_options:
    #   ba_global_function_tolerance: stop global BA when improvement per
    #     iteration drops below this threshold (avoids over-fitting to noise).
    #   ba_global_max_num_iterations: allow more BA iterations than COLMAP's
    #     default (50) for better convergence on long/complex sequences.
    print("--- [hloc/COLMAP] Running incremental mapping + global Bundle Adjustment ---")
    reconstruction.main(
        sfm_dir,
        input_dir,          # must match the root used for feature extraction
        sfm_pairs,
        features,
        matches,
        image_list=image_list_paths,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={"camera_model": args.camera},
        mapper_options={
            "ba_global_function_tolerance": 1e-6,
            "ba_global_max_num_iterations": 100,
            "num_threads": 1, # add num_threads 
        },
        verbose=True,
    )

    # Guard: an empty sparse model means reconstruction completely failed.
    # Common causes: too few matches, wrong pairing strategy, corrupt images.
    if not (list(sfm_dir.glob("*.bin")) + list(sfm_dir.glob("*.txt"))):
        logging.error(
            "Reconstruction at %s is empty. "
            "Check feature extraction and matching quality.", sfm_dir
        )
        exit(1)

    # ── SfM quality evaluation ───────────────────────────────────────────
    #
    # evaluate_sfm computes metrics from the sparse model without needing
    # ground truth. Typical metrics include registration rate, mean
    # reprojection error, and mean track length.
    # The result is saved to distorted/sfm_metrics.json for logging and
    # comparison across different pipeline configurations.
    print("--- [Benchmark] Evaluating SfM reconstruction quality ---")
    try:
        metrics = evaluate_sfm(sfm_dir, input_dir)
        print(metrics)
        with open(output_dir / "sfm_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        logging.warning("SfM evaluation skipped (non-critical): %s", e)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Image undistortion
#
# COLMAP image_undistorter takes the original (distorted) frames and the
# sparse model, and produces:
#   source_path/images/    — undistorted frames  (3DGS train images)
#   source_path/sparse/0/  — undistorted sparse model (3DGS camera poses)
#
# Critical invariant for 3DGS:
#   The image names stored inside sparse/0/images.bin must exactly match
#   the filenames on disk inside images/. COLMAP guarantees this when
#   --image_path points to the same folder used during reconstruction
#   (input_dir in this pipeline).
#
# --output_type COLMAP writes binary .bin model files, which is the format
# expected by the 3DGS dataloader (scene/dataset_readers.py).
# ═══════════════════════════════════════════════════════════════════════

if not sfm_dir.exists():
    logging.error(
        "No reconstruction found at %s. "
        "Run without --skip_matching first.", sfm_dir
    )
    exit(1)

print("--- [COLMAP] Undistorting images ---")
exit_code = os.system(
    f"{colmap_command} image_undistorter "
    f"--image_path {input_dir} "      # raw frames (distorted)
    f"--input_path {sfm_dir} "        # sparse model with distortion params
    f"--output_path {source_path} "   # 3DGS scene root
    f"--output_type COLMAP"
)
if exit_code != 0:
    logging.error("Undistorter failed with code %d. Exiting.", exit_code)
    exit(exit_code)

# ── Sparse model reorganisation ──────────────────────────────────────
#
# Older COLMAP versions occasionally write model files directly into
# sparse/ instead of sparse/0/. This block moves any such loose files
# into sparse/0/ to match the structure 3DGS expects.
# On current COLMAP builds this loop is typically a no-op.
sparse_root = source_path / "sparse"
if sparse_root.exists():
    final_sparse_dir.mkdir(parents=True, exist_ok=True)
    for fname in os.listdir(sparse_root):
        if fname == "0":
            continue
        src = sparse_root / fname
        if src.is_file():
            shutil.move(str(src), str(final_sparse_dir / fname))

# ── Verify 3DGS input completeness ──────────────────────────────────
assert final_images_dir.exists(), f"Missing undistorted images dir: {final_images_dir}"
assert final_sparse_dir.exists(), f"Missing sparse model dir: {final_sparse_dir}"
assert any(final_sparse_dir.glob("*.bin")), \
    f"sparse/0/ exists but contains no .bin files: {final_sparse_dir}"

print(f"--- [OK] 3DGS input ready at {source_path} ---")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — Camera trajectory export (TUM format)
#
# Exports the camera trajectory as a .tum file for pose evaluation with
# standard tools such as evo (https://github.com/MichaelGrupp/evo).
#
# We export from sfm_dir (distorted/sparse/0) rather than final_sparse_dir
# because sfm_dir contains ALL registered images before undistortion
# filtering. Some images may be dropped during undistortion (e.g. if the
# undistorter cannot handle extreme distortion), so using sfm_dir gives
# a more complete and accurate trajectory for ATE / RPE evaluation.
#
# TUM format per line:  timestamp tx ty tz qx qy qz qw
#
# Coordinate convention (aligned with ScanNet ground truth):
#   COLMAP stores poses as T_cw (World → Camera):
#       R_cw = qvec2rotmat(img.qvec),  t_cw = img.tvec
#   ScanNet ground truth stores poses as T_wc (Camera → World).
#   export_trajectory_tum inverts T_cw to obtain T_wc:
#       R_wc = R_cw.T,  t_wc = -R_wc @ t_cw
#   The resulting t_wc is the camera position in world space, and R_wc
#   is converted to quaternion [qx, qy, qz, qw] via scipy (note: COLMAP
#   uses [qw, qx, qy, qz] internally — reordering is handled in export).
#
# If ATE / RPE from evo appears abnormally large, check in order:
#   1. Quaternion order (qw last for TUM)
#   2. Pose inversion (T_wc vs T_cw)
#   3. Timestamp alignment (ScanNet uses integer stems: 0, 1, 2, ...)
# ═══════════════════════════════════════════════════════════════════════

print("--- [Benchmark] Exporting camera trajectory (TUM format) ---")
try:
    export_trajectory_tum(sfm_dir, source_path / "trajectory.tum")
except Exception as e:
    logging.warning("Trajectory export skipped (non-critical): %s", e)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — Multi-scale image pyramid  (optional, --resize flag)
#
# Generates downscaled copies of the undistorted images for coarse-to-fine
# 3DGS training. The 3DGS trainer can load images_2/, images_4/, images_8/
# alongside the full-resolution images/ to progressively refine Gaussians.
#
# Scales produced:
#   images_2/  — 50 %  of original resolution
#   images_4/  — 25 %
#   images_8/  — 12.5 %
#
# Source is always final_images_dir (undistorted), NOT input_dir (raw).
# Training on distorted raw frames would introduce systematic error because
# the sparse model (and therefore Gaussian initialisation) is already in
# the undistorted coordinate system.
# ═══════════════════════════════════════════════════════════════════════

if args.resize:
    print("--- [Resize] Generating multi-scale image pyramid ---")
    for scale, pct in [(2, "50%"), (4, "25%"), (8, "12.5%")]:
        scale_dir = source_path / f"images_{scale}"
        scale_dir.mkdir(exist_ok=True)
        for fname in os.listdir(final_images_dir):
            src = final_images_dir / fname
            dst = scale_dir / fname
            shutil.copy2(src, dst)
            exit_code = os.system(f'{magick_command} mogrify -resize {pct} "{dst}"')
            if exit_code != 0:
                logging.error("%s resize failed with code %d. Exiting.", pct, exit_code)
                exit(exit_code)

print("Done.")