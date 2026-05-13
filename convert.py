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
try:
    from hloc import extract_features, match_features, pairs_from_exhaustive,pairs_from_sequence, reconstruction
except ImportError:
    logging.error("Install hloc first: pip install -e . (in the Hierarchical-Localization directory)")
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
# test input: --source_path /content/drive/MyDrive/KLTN/test_input --camera SIMPLE_PINHOLE
# args = parser.parse_args([
#     "--source_path", "/content/drive/MyDrive/KLTN/test_input",
#     "--camera", "SIMPLE_PINHOLE"
# ])

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


if not args.skip_matching:
    
    # ## Feature extraction
    # feat_extracton_cmd = colmap_command + " feature_extractor "\
    #     "--database_path " + args.source_path + "/distorted/database.db \
    #     --image_path " + args.source_path + "/input \
    #     --ImageReader.single_camera 1 \
    #     --ImageReader.camera_model " + args.camera + " \
    #     --SiftExtraction.use_gpu " + str(use_gpu)
    # exit_code = os.system(feat_extracton_cmd)
    # if exit_code != 0:
    #     logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
    #     exit(exit_code)
    
    
   
    # Path setup
    source_path = Path(args.source_path)
    image_dir = source_path # Corrected: Images are directly in source_path, not source_path / "input"
    output_dir = source_path / "distorted"
    output_dir.mkdir(parents=True, exist_ok=True);

    sfm_pairs = output_dir / "pairs-exhaustive.txt"
    sfm_dir = output_dir / "sparse"/"0"
    features = output_dir / "features.h5"
    matches = output_dir / "matches.h5"

    ## Feature extraction and matching with hloc
    feature_conf = extract_features.confs['loma_aachen'] # Changed 'loma' to 'loma_aachen', you can choose loma_inloc
    matcher_conf = match_features.confs['loma']

    print("--- [LoMa] Creating image pairs ---")
    # Get list of images from the image_dir and convert to strings
    image_list_paths = [str(p.relative_to(image_dir)) for p in image_dir.iterdir() if p.is_file()]
    
    # Pair generation strategy for image matching.
    # Current: exhaustive matching (all image pairs).
    # Can be replaced with:
    # - sequential pairs (video-like captures)
    # - retrieval-based pairs
    # - vocabulary tree matching
    # - overlap-based pairing
    # for better scalability on large indoor datasets.
    
    pairs_from_exhaustive.main(sfm_pairs, image_list=image_list_paths) 
    #pairs_from_sequence.main(sfm_pairs, image_list=image_list_paths) # Optional: Add sequential pairs for video-like datasets

    print("--- [LoMa] Extracting features ---")
    extract_features.main(feature_conf, image_dir, feature_path=features)

    print("--- [LoMa] Matching features ---")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches) # Corrected: Pass features and matches as keyword arguments

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
input_model_path = output_dir / "sparse"
img_undist_cmd = (colmap_command + " image_undistorter "
    f"--image_path {args.source_path} " # Corrected from /input
    f"--input_path {input_model_path} "
    f"--output_path {args.source_path} "
    #f"--output_path {output_dir} "
    "--output_type COLMAP")

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