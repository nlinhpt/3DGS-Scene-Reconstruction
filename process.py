"""
process.py — LK + Homography geometric keyframe filter cho ScanNet

Chạy:
    python process.py \
        --scene_path /data/scene0011_00 \
        --output_dir scene_processed

Tùy chỉnh ngưỡng:
    python process.py \
        --scene_path /data/scene0011_00 \
        --output_dir scene_processed \
        --min_movement 15.0 \
        --rot_inlier 0.88 \
        --stride 2
"""

import cv2
import numpy as np
import logging
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ============================================================
# Argument Parser
# ============================================================
parser = ArgumentParser("ScanNet geometric keyframe filter")
parser.add_argument("--scene_path",   required=True, type=str,
                    help="Đường dẫn scene ScanNet (chứa color/)")
parser.add_argument("--output_dir",   default="scene_processed", type=str,
                    help="Thư mục output (sẽ tạo output_dir/input/)")
parser.add_argument("--stride",       default=1,    type=int,
                    help="Đọc 1-in-stride frames")
parser.add_argument("--max_corners",  default=1000, type=int,
                    help="goodFeaturesToTrack maxCorners")
parser.add_argument("--min_valid",    default=40,   type=int,
                    help="Số điểm tracked tối thiểu trước khi tính H")
parser.add_argument("--min_movement", default=20.0, type=float,
                    help="Mean pixel movement tối thiểu (STATIONARY threshold)")
parser.add_argument("--rot_inlier",   default=0.92, type=float,
                    help="Homography inlier ratio để coi là PURE_ROTATION")
parser.add_argument("--rot_min_mov",  default=45.0, type=float,
                    help="Pixel movement tối thiểu khi xét PURE_ROTATION")
args = parser.parse_args()


# ============================================================
# Loader
# ============================================================
class ScanNetLoader:
    def __init__(self, scene_dir: str, stride: int = 1):
        color_dir  = Path(scene_dir) / "color"
        self.paths = sorted(
            color_dir.glob("*.jpg"),
            key=lambda p: int("".join(filter(str.isdigit, p.stem)))
        )[::stride]
        if not self.paths:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {color_dir}")

    def __len__(self):
        return len(self.paths)

    def iter_frames(self):
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is not None:
                yield img, p.name


# ============================================================
# AdvancedGeometricFilterLK
# ============================================================
class AdvancedGeometricFilterLK:
    """
    Phân loại chuyển động camera qua LK optical flow + Homography RANSAC.

    Status trả về:
      TRACKING_FAIL  — không đủ điểm hoặc H không tính được
      STATIONARY     — camera đứng yên
      PURE_ROTATION  — xoay tại chỗ, baseline gần 0
      VALID_MOVE     — có translation → keyframe hợp lệ
    """

    TRACKING_FAIL = "TRACKING_FAIL"
    STATIONARY    = "STATIONARY"
    PURE_ROTATION = "PURE_ROTATION"
    VALID_MOVE    = "VALID_MOVE"

    def __init__(
        self,
        max_corners:  int   = 1000,
        min_valid:    int   = 40,
        min_movement: float = 20.0,
        rot_inlier:   float = 0.92,
        rot_min_mov:  float = 45.0,
    ):
        self.min_valid    = min_valid
        self.min_movement = min_movement
        self.rot_inlier   = rot_inlier
        self.rot_min_mov  = rot_min_mov

        self._fp = dict(maxCorners=max_corners, qualityLevel=0.03,
                        minDistance=10, blockSize=7)
        self._lk = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def compute_geometric_status(self, img1: np.ndarray, img2: np.ndarray):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        p1 = cv2.goodFeaturesToTrack(g1, **self._fp)
        if p1 is None:
            return self.TRACKING_FAIL, None, None, None

        p2, st_fwd, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p1, None, **self._lk)
        p1_rec, _, _  = cv2.calcOpticalFlowPyrLK(g2, g1, p2, None, **self._lk)

        fb_error   = np.linalg.norm(p1 - p1_rec, axis=2).ravel()
        valid_mask = (st_fwd.ravel() == 1) & (fb_error < 1.0)
        n_valid    = int(np.sum(valid_mask))

        if n_valid < self.min_valid:
            return self.TRACKING_FAIL, None, None, None

        clean_p1 = p1[valid_mask].reshape(-1, 2)
        clean_p2 = p2[valid_mask].reshape(-1, 2)

        H, inlier_mask = cv2.findHomography(clean_p1, clean_p2, cv2.RANSAC, 3.0)
        if H is None:
            return self.TRACKING_FAIL, None, None, None

        inlier_ratio  = float(np.sum(inlier_mask)) / n_valid
        mean_movement = float(np.mean(np.linalg.norm(clean_p1 - clean_p2, axis=1)))

        if mean_movement < self.min_movement:
            return self.STATIONARY, clean_p1, clean_p2, valid_mask

        if inlier_ratio > self.rot_inlier and mean_movement > self.rot_min_mov:
            return self.PURE_ROTATION, clean_p1, clean_p2, valid_mask

        return self.VALID_MOVE, clean_p1, clean_p2, valid_mask


# ============================================================
# Main
# ============================================================
def main():
    save_dir = Path(args.output_dir) / "input"
    save_dir.mkdir(parents=True, exist_ok=True)

    loader = ScanNetLoader(args.scene_path, stride=args.stride)
    filt   = AdvancedGeometricFilterLK(
        max_corners  = args.max_corners,
        min_valid    = args.min_valid,
        min_movement = args.min_movement,
        rot_inlier   = args.rot_inlier,
        rot_min_mov  = args.rot_min_mov,
    )

    logging.info(f"Tổng frames gốc: {len(loader)}")
    logging.info(f"Output: {save_dir}")

    frame_iter = loader.iter_frames()
    saved      = 0
    stats      = {s: 0 for s in [filt.TRACKING_FAIL, filt.STATIONARY,
                                   filt.PURE_ROTATION, filt.VALID_MOVE]}

    # Frame đầu tiên → luôn lưu làm mỏ neo
    ref_img, ref_name = next(frame_iter)
    saved += 1
    new_ref_name = f"{saved:06d}.jpg"
    cv2.imwrite(str(save_dir / new_ref_name), ref_img)

    for curr_img, curr_name in frame_iter:
        status, p1, p2, vmask = filt.compute_geometric_status(ref_img, curr_img)
        stats[status] += 1

        if status == filt.VALID_MOVE:
            saved += 1
            new_curr_name = f"{saved:06d}.jpg"
            cv2.imwrite(str(save_dir / new_curr_name), curr_img)
            logging.info(
                f"VALID_MOVE  {curr_name} → {new_curr_name}  "
                f"pts={p1.shape[0]}"
            )
            ref_img      = curr_img
            new_ref_name = new_curr_name

        elif status == filt.TRACKING_FAIL:
            # Mất tracking → cập nhật mỏ neo nhưng không lưu
            logging.warning(f"TRACKING_FAIL  {curr_name} — cập nhật mỏ neo")
            ref_img      = curr_img
            new_ref_name = f"{saved:06d}.jpg"

        # STATIONARY / PURE_ROTATION → bỏ qua hoàn toàn

    logging.info("=" * 48)
    logging.info(f"Kết quả preprocessing:")
    logging.info(f"  Gốc            : {len(loader)} frames")
    logging.info(f"  Lưu (VALID)    : {saved} frames")
    logging.info(f"  STATIONARY     : {stats[filt.STATIONARY]}")
    logging.info(f"  PURE_ROTATION  : {stats[filt.PURE_ROTATION]}")
    logging.info(f"  TRACKING_FAIL  : {stats[filt.TRACKING_FAIL]}")
    logging.info(f"  Output         : {save_dir}")


if __name__ == "__main__":
    main()
