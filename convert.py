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
from argparse import ArgumentParser
import shutil
from pathlib import Path

# Additional pair generation strategies kept for future experiments:
# - pairs_from_retrieval: retrieval-based image pairing
# - pairs_from_covisibility: covisibility-based pairing
# - pairs_from_poses: pose-based pairing

try:
    from hloc import (
        extract_features,
        match_features,
        pairs_from_exhaustive,
        pairs_from_covisibility,
        pairs_from_poses,
        pairs_from_retrieval,
        reconstruction,
    )
except ImportError:
    logging.error(
        "Install hloc first: pip install -e . "
        "(in the Hierarchical-Localization directory)"
    )
    exit(1)
    
parser = ArgumentParser("Colmap converter with LoMa")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()


colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


if not args.skip_matching:
 
    # Path setup
    source_path = Path(args.source_path)
    #image_dir = source_path
    image_dir = source_path / "images"
    output_dir = source_path / "distorted"
    
    output_dir.mkdir(parents=True, exist_ok=True);

    sfm_pairs = output_dir / "pairs-exhaustive.txt"
    sfm_dir = output_dir / "sparse"/"0"
    features = output_dir / "features.h5"
    matches = output_dir / "matches.h5"

    # ============================================================
    # Feature Extraction and Matching
    # ============================================================

    """
    LoMa-based COLMAP conversion pipeline for Gaussian Splatting.

    Pipeline:
    1. Generate image pairs
    2. Extract LoMa local features
    3. Match image pairs using LoMa matcher
    4. Run sparse SfM reconstruction with HLoc + COLMAP
    5. Undistort images for Gaussian Splatting
    6. Generate multi-scale image pyramids

    Supports future extensions with:
    - retrieval-based pairing
    - covisibility pairing
    - pose-based pairing
    """


    feature_conf = extract_features.confs['loma_aachen'] 
    
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
    

    print("--- [LoMa] Creating image pairs ---")
    # Get list of images from the image_dir and convert to strings
    #image_list_paths = [str(p.relative_to(image_dir)) for p in image_dir.iterdir() if p.is_file()]
    valid_ext = [".jpg", ".jpeg", ".png"]

    image_list_paths = [
        str(p.relative_to(image_dir))
        for p in image_dir.iterdir()
        if p.suffix.lower() in valid_ext
    ]
    # Pair generation strategy for image matching.
    # Current: exhaustive matching (all image pairs).
    # Can be replaced with:
    # - retrieval-based pairs
    # - vocabulary tree matching
    # - overlap-based pairing
    # for better scalability on large indoor datasets.
    
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list_paths) 
    extract_features.main(feature_conf, image_dir, feature_path=features)

    print("--- [LoMa] Matching features ---")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches) 

    print("--- [hloc] Running Reconstruction (Sparse Model) ---")
    reconstruction.main(
    sfm_dir,
    image_dir,
    sfm_pairs,
    features,
    matches,
    camera_mode="SINGLE",
    image_options={"camera_model": args.camera}
)



### Image undistortion
print("--- [COLMAP] Undistorting images ---")
input_model_path = output_dir / "sparse" /"0" # add "0" to match COLMAP's expected input structure for undistorter

img_undist_cmd = (
    colmap_command + " image_undistorter "
    f"--image_path {image_dir} "
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

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")