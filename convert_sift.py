# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
import logging
from argparse import ArgumentParser
import shutil
from pathlib import Path
from benchmark_sfm import evaluate_sfm, export_trajectory_tum
import json

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument(
    "--matcher",
    type=str,
    default="exhaustive",
    choices=["exhaustive", "sequential"]
)

parser.add_argument(
    "--loop_detection",
    type=int,
    default=0,
    choices=[0, 1],
    help="Enable COLMAP loop detection (1) or disable (0)"
)

args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0



if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # ## Feature matching
    # # sequential_matcher
    # # feat_matching_cmd = colmap_command + " sequential_matcher \
    # #     --database_path " + args.source_path + "/distorted/database.db \
    # #     --SiftMatching.use_gpu " + str(use_gpu)

    # # 
    # feat_matching_cmd = (colmap_command + " exhaustive_matcher \
    #     --database_path " + args.source_path + "/distorted/database.db \
    #     --SiftMatching.use_gpu " + str(use_gpu))
    
    # # feat_matching_cmd = (colmap_command + " sequential_matcher \
    # #     --database_path " + args.source_path + "/distorted/database.db \
    # #     --SiftMatching.use_gpu " + str(use_gpu) + " \
    # #     --SequentialMatching.overlap 10 \
    # #     --SequentialMatching.loop_detection 0")
    
    if args.matcher == "exhaustive":
        print("--- Using Exhaustive Matcher ---")

        feat_matching_cmd = (
            colmap_command + " exhaustive_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)
        )

    elif args.matcher == "sequential":
        print(f"--- Using Sequential Matcher (overlap=10, loop_detection={args.loop_detection}) ---")

        feat_matching_cmd = (
            colmap_command + " sequential_matcher "
            "--database_path " + args.source_path + "/distorted/database.db "
            "--SiftMatching.use_gpu " + str(use_gpu) + " "
            "--SequentialMatching.overlap 10 "
            "--SequentialMatching.loop_detection " + str(args.loop_detection)
        )
    
    print(f"--- [Strict Evaluation] COLMAP matching | overlap=10 | loop_detection={args.loop_detection} ---")
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001\
        --Mapper.num_threads 1")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + " \
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# --- PHẦN CHÈN BENCHMARK VÀ TUM ---
sfm_dir = Path(args.source_path) / "distorted" / "sparse" / "0"
image_dir = Path(args.source_path) / "input"
output_dir = Path(args.source_path)
print("--- [Benchmark]---")
metrics = evaluate_sfm(sfm_dir, image_dir)
print(metrics)

with open(output_dir / "sfm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

export_trajectory_tum(sfm_dir, output_dir / "trajectory.tum")
# -----------------------------------

if(args.resize):
    print("Copying and resizing...")
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    files = os.listdir(args.source_path + "/images")
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        os.system(magick_command + " mogrify -resize 50% " + destination_file)
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        os.system(magick_command + " mogrify -resize 25% " + destination_file)
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        os.system(magick_command + " mogrify -resize 12.5% " + destination_file)

print("Done.")