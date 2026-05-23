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
import logging
import json
from argparse import ArgumentParser
import shutil
from pathlib import Path

from benchmark_sfm import evaluate_sfm, export_trajectory_tum

# Additional pair generation strategies kept for future experiments:
# - pairs_from_exhaustive: exhaustive image pairing
# - pairs_from_sequential: sequential image pairing
# - pairs_from_covisibility: covisibility-based pairing
# - pairs_from_poses: pose-based pairing

try:
    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        reconstruction,
    )
    import pycolmap
except ImportError:
    logging.error(
        "Install hloc and pycolmap first: pip install -e . "
        "(in the Hierarchical-Localization directory)"
    )
    exit(1)
    
parser = ArgumentParser("Colmap converter with LoMa and Retrieval-based Matching")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--num_matched", default=50, type=int, help="Number of retrieved image pairs per image")
args = parser.parse_args()


colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


if not args.skip_matching:
 
    # Path setup
    source_path = Path(args.source_path)
    input_dir = source_path / "input"
    image_dir = source_path / "images"
    
    # Restructure directory format dynamically to align with 3DGS expectations.
    # The original raw images are moved to 'input' so that COLMAP's undistorter
    # can dump the processed, undistorted frames directly into 'images'.
    if image_dir.exists() and not input_dir.exists():
        print("--- [Setup] Renaming 'images' to 'input' for original frames ---")
        image_dir.rename(input_dir)
        
    if not input_dir.exists():
        logging.error("Source directory does not contain an 'input' or 'images' folder!")
        exit(1)
        
    output_dir = source_path / "distorted"
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_pairs = output_dir / "pairs-retrieval.txt"
    sfm_dir = output_dir / "sparse" / "0"
    global_features_path = output_dir / "global_features.h5"
    features = output_dir / "features.h5"
    matches = output_dir / "matches.h5"

    # ============================================================
    # Feature Extraction and Matching
    # ============================================================

    """
    LoMa-based COLMAP conversion pipeline for Gaussian Splatting.

    Pipeline:
    1. Extract global features and retrieve image pairs (NetVLAD)
    2. Extract LoMa local features (DaD & DeDoDe)
    3. Match image pairs using LoMa matcher (LightGlue-based)
    4. Run sparse SfM reconstruction with HLoc + COLMAP Backend
    5. Evaluate reconstruction metrics
    6. Undistort images for Gaussian Splatting
    7. Export camera trajectories and multi-scale image pyramids
    """

    feature_conf = extract_features.confs['loma_inloc'] 
    
    # Preset configuration for LoMa feature extraction.
    # Available presets:
    # - loma_aachen : resize_max = 1024 (outdoor/Aachen style datasets)
    # - loma_inloc  : resize_max = 1600 (indoor/high-resolution datasets)
    #
    # These presets ONLY affect:
    # - image preprocessing
    # - resize resolution
    # - output feature filename
    #
    # The actual LoMa extractor implementation is defined in:
    # hloc/extractors/loma.py

    matcher_conf = match_features.confs['loma']
    
    # Default matcher architecture: LoMa-B.
    # To override:
    #
    # matcher_conf = {
    #     "output": "matches-loma-r",
    #     "model": {
    #         "name": "loma",
    #         "arch": "LoMa-R",
    #     },
    # }
    #
    # Available:
    # LoMa-B / LoMa-L / LoMa-G / LoMa-R
    

    print("--- [hloc] Extracting global features via NetVLAD ---")
    valid_ext = [".jpg", ".jpeg", ".png"]
    image_list_paths = [
        str(p.relative_to(input_dir))
        for p in input_dir.iterdir()
        if p.suffix.lower() in valid_ext
    ]
    
    retrieval_conf = extract_features.confs['netvlad']
    extract_features.main(retrieval_conf, input_dir, image_list=image_list_paths, feature_path=global_features_path)

    print(f"--- [hloc] Generating image pairs from retrieval (k={args.num_matched}) ---")
    pairs_from_retrieval.main(global_features_path, sfm_pairs, num_matched=args.num_matched)
    
    print("--- [LoMa] Extracting local features (DaD detector + DeDoDe descriptor) ---")
    extract_features.main(feature_conf, input_dir, feature_path=features)

    print("--- [LoMa] Matching features via LightGlue matcher ---")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches) 

    print("--- [hloc/COLMAP] Running Incremental Mapping & Global Bundle Adjustment ---")
    # Enable pycolmap internal logging to expose Bundle Adjustment reports on terminal
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
        camera_mode="SINGLE",
        image_options={"camera_model": args.camera},
        verbose=True
    )

    # ============================================================
    # SfM Quality Evaluation
    # ============================================================
    print("--- [Benchmark] Evaluating SfM reconstruction quality ---")
    metrics = evaluate_sfm(sfm_dir, input_dir)
    print(metrics)
    
    with open(output_dir / "sfm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


# ============================================================
# Image Undistortion & Model Organization
# ============================================================
print("--- [COLMAP] Undistorting images ---")
input_model_path = output_dir / "sparse" / "0" 

img_undist_cmd = (
    colmap_command + " image_undistorter "
    f"--image_path {source_path / 'input'} "
    f"--input_path {input_model_path} "
    f"--output_path {args.source_path} "
    "--output_type COLMAP"
)

exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

# Move undistorted sparse model to expected location for Gaussian Splatting.
sparse_root = Path(args.source_path) / "sparse"
if sparse_root.exists():
    files = os.listdir(sparse_root)
    dest_proto = sparse_root / "0"
    dest_proto.mkdir(exist_ok=True)

    for file in files:
        if file == '0': continue
        source_file = sparse_root / file
        destination_file = dest_proto / file
        if source_file.is_file():
            shutil.move(str(source_file), str(destination_file))

# ============================================================
# Multi-scale Image Pyramid Generation
# ============================================================
if (args.resize):
    print("Copying and resizing undistorted images...")
    undistorted_image_dir = os.path.join(args.source_path, "images")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(undistorted_image_dir)
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(undistorted_image_dir, file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + ' mogrify -resize 50% "' + destination_file + '"')
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + ' mogrify -resize 25% "' + destination_file + '"')
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + ' mogrify -resize 12.5% "' + destination_file + '"')
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

# ============================================================
# Trajectory Export
# ============================================================
print("--- [Benchmark] Exporting camera trajectory in TUM format ---")
model_dir = Path(args.source_path) / "sparse" / "0"
tum_output = Path(args.source_path) / "trajectory.tum"
export_trajectory_tum(model_dir, tum_output)

print("Done.")
