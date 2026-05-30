from pathlib import Path
from hloc.utils.read_write_model import read_model
import numpy as np

def evaluate_sfm(sfm_dir, image_dir):
    # Đọc model từ file .bin
    cameras, images, points3D = read_model(path=sfm_dir, ext=".bin")

    # 1. Registration Rate
    valid_ext = [".jpg", ".jpeg", ".png"]
    total_images = len([p for p in Path(image_dir).iterdir() if p.suffix.lower() in valid_ext])
    registered_images = len(images)
    registration_rate = (registered_images / total_images * 100) if total_images > 0 else 0.0

    # 2. Sparse Points
    num_sparse_points = len(points3D)

    # 3. Track Length & Reprojection Error
    if num_sparse_points > 0:
        track_lengths = [len(p.image_ids) for p in points3D.values()]
        mean_track_length = np.mean(track_lengths)
        reproj_errors = [p.error for p in points3D.values()]
        mean_reproj_error = np.mean(reproj_errors)
    else:
        mean_track_length = 0.0
        mean_reproj_error = 0.0

    metrics = {
        "registration_rate": round(registration_rate, 2),
        "registered_images": registered_images,
        "total_images": total_images,
        "num_sparse_points": num_sparse_points,
        "mean_track_length": round(float(mean_track_length), 3),
        "mean_reprojection_error": round(float(mean_reproj_error), 4)
    }
    return metrics

def export_trajectory_tum(model_path, output_tum_path):
    """Đọc mô hình COLMAP và xuất quỹ đạo sang định dạng TUM."""
    print("--- [Benchmark] Exporting trajectory to TUM format ---")
    images_bin = model_path / "images.bin"
    images_txt = model_path / "images.txt"
    
    if not images_bin.exists() and not images_txt.exists():
        print(f"Warning: Sparse model 'images' not found in {model_path}. Skipping trajectory export.")
        return

    ext = ".bin" if images_bin.exists() else ".txt"
    _, images, _ = read_model(path=model_path, ext=ext)

    with open(output_tum_path, 'w') as f:
        # Sắp xếp theo tên file ảnh (ví dụ từ 000004.png tăng dần)
        sorted_images = sorted(images.values(), key=lambda x: x.name)
        
        for img in sorted_images:
            # Lấy tên file bỏ đuôi làm timestamp để khớp với Ground Truth
            timestamp = Path(img.name).stem 
            tx, ty, tz = img.tvec
            qw, qx, qy, qz = img.qvec
            
            # Định dạng TUM: timestamp tx ty tz qx qy qz qw
            f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
            
    print(f"Trajectory saved to {output_tum_path}")