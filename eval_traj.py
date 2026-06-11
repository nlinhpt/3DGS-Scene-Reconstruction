# ==============================================================================
#  SfM Trajectory Evaluator – ScanNet / TUM format
#  Reference: Sturm et al. 2012 (ATE / RPE), Umeyama 1991 (Sim(3) alignment)
# ==============================================================================

import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

warnings.filterwarnings('ignore')


class TrajectoryEvaluator:
    """
    End-to-end evaluator for SfM trajectory estimation against ScanNet ground truth.

    Pipeline
    --------
    1. load()         — Load predicted and ground-truth TUM files into memory.
    2. associate()    — Match predicted and GT timestamps (exact + nearest-neighbour).
    3. align()        — Estimate Sim(3) via Umeyama alignment; apply to predictions.
    4. evaluate()     — Compute ATE and RPE metrics.
    5. print_results()— Pretty-print a summary table.
    6. plot()         — Save 6 individual evaluation plots as PNG.
    7. save_json()    — Persist all metrics to a JSON file.

    Parameters
    ----------
    pred_tum_path : Path
        Path to the predicted trajectory in TUM format (exported by SfM pipeline).
    scannet_pose_dir : Path
        Directory containing per-frame ScanNet pose files  <frame_id>.txt  (4×4 T_wc).
    output_dir : Path
        Directory where plots and JSON results will be saved.
    gt_tum_path : Path, optional
        If provided, skip GT conversion and load this file directly.
        If None, GT is converted from scannet_pose_dir on first use.

    Attributes (populated after each step)
    ----------------------------------------
    pred_poses    : dict  — timestamp → {'t': (3,), 'R': (3,3)}
    gt_poses      : dict  — timestamp → {'t': (3,), 'R': (3,3)}
    matches       : list  — [(ts_pred, ts_gt), ...]
    s             : float — estimated scale
    R_align       : (3,3) — alignment rotation
    t_align       : (3,)  — alignment translation
    pred_pts      : (N,3) — raw predicted translations (matched order)
    gt_pts        : (N,3) — GT translations (matched order)
    pred_aligned  : (N,3) — Sim(3)-aligned predicted translations
    ate           : dict  — ATE metrics
    rpe           : dict  — RPE metrics
    """

    # ── Plot style constants ──────────────────────────────────────────────────
    _STYLE = {
        'gt'   : {'color': '#1f77b4', 'marker': 'o', 'ls': '-',  'label': 'Ground Truth'},
        'pred' : {'color': '#d62728', 'marker': 's', 'ls': '--', 'label': 'SfM (aligned)'},
        'rmse' : {'color': '#d62728', 'ls': '--'},
        'lw'   : 1.5,
        'ms'   : 3,
    }

    def __init__(
        self,
        pred_tum_path    : Path,
        scannet_pose_dir : Path,
        output_dir       : Path,
        gt_tum_path      : Path | None = None,
    ) -> None:
        self.pred_tum_path    = Path(pred_tum_path)
        self.scannet_pose_dir = Path(scannet_pose_dir)
        self.output_dir       = Path(output_dir)
        self.gt_tum_path      = Path(gt_tum_path) if gt_tum_path else \
                                 self.output_dir / 'gt.tum'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Populated progressively by each step
        self.pred_poses   : dict        = {}
        self.gt_poses     : dict        = {}
        self.matches      : list        = []
        self.s            : float       = 1.0
        self.R_align      : np.ndarray  = np.eye(3)
        self.t_align      : np.ndarray  = np.zeros(3)
        self.pred_pts     : np.ndarray  = np.empty((0, 3))
        self.gt_pts       : np.ndarray  = np.empty((0, 3))
        self.pred_aligned : np.ndarray  = np.empty((0, 3))
        self.ate          : dict        = {}
        self.rpe          : dict        = {}

    # ── Step 0 – Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _load_tum(path: Path) -> dict:
        """
        Reads a TUM trajectory file into a pose dictionary.

        Parameters
        ----------
        path : Path
            TUM file with lines:  timestamp tx ty tz qx qy qz qw

        Returns
        -------
        poses : dict
            Integer timestamp → {'t': (3,), 'R': (3,3)}.
        """
        poses = {}
        with open(path) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                ts = int(float(parts[0]))
                t  = np.array([float(x) for x in parts[1:4]])
                q  = np.array([float(x) for x in parts[4:8]])   # qx qy qz qw
                R  = Rotation.from_quat(q).as_matrix()
                poses[ts] = {'t': t, 'R': R}
        return poses

    @staticmethod
    def _build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Builds a 4x4 homogeneous transformation matrix from R and t.

        Parameters
        ----------
        R : numpy.ndarray
            (3, 3) shaped rotation matrix.
        t : numpy.ndarray
            (3,) shaped translation vector.

        Returns
        -------
        T : numpy.ndarray
            (4, 4) shaped SE(3) transformation matrix.
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = t
        return T

    # ── Step 1 – Load ─────────────────────────────────────────────────────────

    def convert_scannet_gt_to_tum(self) -> int:
        """
        Converts ScanNet pose files to TUM trajectory format.
        Reads each pose/<frame_id>.txt file (4×4 T_wc matrix) and writes:
        timestamp tx ty tz qx qy qz qw.

        Why a 4×4 matrix?
            ScanNet stores T_wc as a homogeneous transformation matrix,
            encoding both rotation and translation in a single operation.
            We extract:
                R = T[:3, :3]  (3×3 rotation matrix)
                t = T[:3,  3]  (translation vector)

        Returns
        -------
        count : int
            Number of successfully converted frames.
        """
        # Sort pose files numerically by frame index (stem)
        pose_files = sorted(self.scannet_pose_dir.glob('*.txt'),
                            key=lambda p: int(p.stem))
        if not pose_files:
            raise FileNotFoundError(
                f'No pose files found in {self.scannet_pose_dir}')

        count = 0
        with open(self.gt_tum_path, 'w') as f:
            f.write('# timestamp tx ty tz qx qy qz qw\n')

            for pose_file in pose_files:
                T = np.loadtxt(pose_file)               # (4, 4) T_wc matrix

                # Skip invalid poses (inf/nan — may occur in ScanNet sequences)
                if not np.isfinite(T).all():
                    continue

                R = T[:3, :3]                           # Rotation matrix
                t = T[:3,  3]                           # Camera position in world frame

                # Convert rotation matrix to quaternion — scipy: (qx, qy, qz, qw)
                qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()

                # Use integer frame index as timestamp to match pred timestamps
                timestamp = int(pose_file.stem)         # e.g. "000001" → 1

                f.write(f'{timestamp} '
                        f'{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} '
                        f'{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n')
                count += 1

        print(f'Converted {count} GT poses → {self.gt_tum_path}')
        return count

    def load(self) -> 'TrajectoryEvaluator':
        """
        Loads predicted and ground-truth poses from TUM files.
        Converts ScanNet GT to TUM format if not already done.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        assert self.pred_tum_path.exists(), \
            f'Predicted TUM file not found: {self.pred_tum_path}'

        # Convert GT if not already available
        if not self.gt_tum_path.exists():
            self.convert_scannet_gt_to_tum()

        self.pred_poses = self._load_tum(self.pred_tum_path)
        self.gt_poses   = self._load_tum(self.gt_tum_path)

        print(f'Loaded {len(self.pred_poses)} pred frames, '
              f'{len(self.gt_poses)} GT frames')
        return self

    # ── Step 2 – Associate ────────────────────────────────────────────────────

    def associate(self, max_diff: int = 5) -> 'TrajectoryEvaluator':
        """
        Matches predicted and GT timestamps via exact match with nearest-neighbour fallback.

        Strategy
        --------
        1. Exact match (preferred).
        2. Nearest-neighbour within max_diff if no exact match exists.

        Parameters
        ----------
        max_diff : int, optional
            Maximum allowed timestamp distance for nearest-neighbour matching.
            Default is 5.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        gt_timestamps = np.array(sorted(self.gt_poses.keys()))
        self.matches  = []

        for ts_pred in sorted(self.pred_poses.keys()):
            if ts_pred in self.gt_poses:
                # Exact match
                self.matches.append((ts_pred, ts_pred))
            else:
                # Nearest-neighbour fallback
                diffs = np.abs(gt_timestamps - ts_pred)
                idx   = np.argmin(diffs)
                if diffs[idx] <= max_diff:
                    self.matches.append((ts_pred, int(gt_timestamps[idx])))

        print(f'Matched {len(self.matches)}/{len(self.pred_poses)} frames '
              f'(GT has {len(self.gt_poses)} poses)')

        if len(self.matches) < 3:
            raise ValueError(
                'Too few matched pairs. Check image names and GT timestamps.')
        return self

    # ── Step 3 – Align ────────────────────────────────────────────────────────

    @staticmethod
    def _umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                           with_scale: bool = True):
        """
        Estimates the Sim(3) transformation between `src` and `dst` point sets.
        Estimates s_opt, R_opt and t_opt such as s_opt * R_opt @ src + t_opt ~ dst.

        Parameters
        ----------
        src : numpy.ndarray
            (N, 3) shaped array of source (predicted) points.
        dst : numpy.ndarray
            (N, 3) shaped array of destination (ground-truth) points.
            Indexes must be consistent with `src`, i.e. dst[i] corresponds to src[i].
        with_scale : bool, optional
            Whether to estimate scale factor. Default is True.

        Returns
        -------
        s_opt : float
            Estimated scale factor.
        R_opt : numpy.ndarray
            (3, 3) shaped optimal rotation matrix.
        t_opt : numpy.ndarray
            (3,) shaped optimal translation vector.
        """
        assert src.shape == dst.shape and src.ndim == 2
        N = src.shape[0]

        # Center both point sets by subtracting their means
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c  = src - mu_src
        dst_c  = dst - mu_dst

        # Variance of centered src points
        var_src = (src_c ** 2).sum() / N

        # Cross-covariance matrix between centered src and dst
        H = (src_c.T @ dst_c) / N                      # (3, 3)

        # Singular Value Decomposition of the cross-covariance matrix
        U, S_diag, Vt = np.linalg.svd(H)
        V = Vt.T

        # Correction vector to ensure a proper rotation (det = +1, no reflection)
        d = np.ones(3)
        if np.linalg.det(V @ U.T) < 0:
            d[2] = -1
        D = np.diag(d)

        # Optimal rotation matrix
        R_opt = V @ D @ U.T                             # (3, 3)

        # Optimal scale factor
        if with_scale:
            s_opt = (S_diag @ d) / var_src
        else:
            s_opt = 1.0

        # Optimal translation vector
        t_opt = mu_dst - s_opt * R_opt @ mu_src

        return s_opt, R_opt, t_opt

    def align(self, with_scale: bool = True) -> 'TrajectoryEvaluator':
        """
        Estimates Sim(3) alignment via Umeyama and applies it to predicted translations.

        Parameters
        ----------
        with_scale : bool, optional
            Use Sim(3) with scale (True, standard for monocular SfM) or
            rigid SE(3) without scale (False, for stereo / RGB-D). Default is True.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        # Paired predicted and ground-truth translation vectors
        self.pred_pts = np.array(
            [self.pred_poses[tp]['t'] for tp, _ in self.matches])   # (N, 3)
        self.gt_pts   = np.array(
            [self.gt_poses[tg]['t']   for _, tg in self.matches])   # (N, 3)

        # Estimate similarity transform with scale (standard for SfM evaluation)
        self.s, self.R_align, self.t_align = self._umeyama_alignment(
            self.pred_pts, self.gt_pts, with_scale=with_scale)

        # Apply the estimated transform to all predicted points
        self.pred_aligned = (
            self.s * (self.R_align @ self.pred_pts.T).T + self.t_align)  # (N, 3)

        print(f'Scale estimated : {self.s:.4f}')
        print(f'Rotation matrix :\n{self.R_align}')
        print(f'Translation     : {self.t_align}')
        return self

    # ── Step 4 – Evaluate ─────────────────────────────────────────────────────

    def _compute_ate(self) -> dict:
        """
        Computes Absolute Trajectory Error (ATE) after alignment.

        ATE_i      = || p_gt_i - p_pred_aligned_i ||
        ATE_RMSE   = sqrt(mean(ATE_i^2))

        Reference: Sturm et al. 2012, eq. 2

        Returns
        -------
        dict with keys: rmse, mean, median, std, min, max, errors.
        """
        # Per-frame Euclidean translation error
        errors = np.linalg.norm(self.gt_pts - self.pred_aligned, axis=1)  # (N,)

        return {
            'rmse'   : float(np.sqrt(np.mean(errors ** 2))),
            'mean'   : float(np.mean(errors)),
            'median' : float(np.median(errors)),
            'std'    : float(np.std(errors)),
            'min'    : float(np.min(errors)),
            'max'    : float(np.max(errors)),
            'errors' : errors,
        }

    def _compute_rpe(self, delta: int = 1) -> dict:
        """
        Computes Relative Pose Error (RPE) between consecutive frame pairs.

        For each pair (i, i+delta):
            Q_gt   = inv(T_gt_i)   @ T_gt_j
            Q_pred = inv(T_pred_i) @ T_pred_j
            F      = Q_gt @ inv(Q_pred)          (error matrix)

        RPE_trans = || trans(F_i) ||
        RPE_rot   = arccos((tr(R(F_i)) - 1) / 2)   (radians)

        Reference: Sturm et al. 2012, eq. 4

        Parameters
        ----------
        delta : int, optional
            Frame spacing (1 = consecutive, 10 = every 10 frames, ...). Default is 1.

        Returns
        -------
        dict with keys: trans_rmse, trans_mean, trans_median,
                        rot_rmse_deg, rot_mean_deg, trans_errors, rot_errors.
        """
        trans_errors = []
        rot_errors   = []

        for i in range(len(self.matches) - delta):
            tp_i, tg_i = self.matches[i]
            tp_j, tg_j = self.matches[i + delta]

            # Ground-truth SE(3) matrices for frame pair (i, j)
            T_gt_i = self._build_T(self.gt_poses[tg_i]['R'], self.gt_poses[tg_i]['t'])
            T_gt_j = self._build_T(self.gt_poses[tg_j]['R'], self.gt_poses[tg_j]['t'])

            # Apply Sim(3) alignment to predicted poses before building SE(3)
            t_pred_i = self.s * self.R_align @ self.pred_poses[tp_i]['t'] + self.t_align
            R_pred_i = self.R_align @ self.pred_poses[tp_i]['R']
            t_pred_j = self.s * self.R_align @ self.pred_poses[tp_j]['t'] + self.t_align
            R_pred_j = self.R_align @ self.pred_poses[tp_j]['R']

            T_pred_i = self._build_T(R_pred_i, t_pred_i)
            T_pred_j = self._build_T(R_pred_j, t_pred_j)

            # Relative motion over the same frame pair for GT and pred
            Q_gt   = np.linalg.inv(T_gt_i)   @ T_gt_j
            Q_pred = np.linalg.inv(T_pred_i) @ T_pred_j

            # Relative pose error matrix
            F = Q_gt @ np.linalg.inv(Q_pred)

            # Translation error: Euclidean norm of the error translation
            trans_errors.append(np.linalg.norm(F[:3, 3]))

            # Rotation error: geodesic angle from the error rotation matrix
            cos_angle = np.clip((np.trace(F[:3, :3]) - 1) / 2, -1.0, 1.0)
            rot_errors.append(np.arccos(cos_angle))

        trans_errors = np.array(trans_errors)
        rot_errors   = np.array(rot_errors)

        return {
            'trans_rmse'   : float(np.sqrt(np.mean(trans_errors ** 2))),
            'trans_mean'   : float(np.mean(trans_errors)),
            'trans_median' : float(np.median(trans_errors)),
            'rot_rmse_deg' : float(np.degrees(np.sqrt(np.mean(rot_errors ** 2)))),
            'rot_mean_deg' : float(np.degrees(np.mean(rot_errors))),
            'trans_errors' : trans_errors,
            'rot_errors'   : rot_errors,
        }

    def evaluate(self, delta: int = 1) -> 'TrajectoryEvaluator':
        """
        Computes ATE and RPE metrics on the aligned trajectory.

        Parameters
        ----------
        delta : int, optional
            Frame spacing for RPE computation. Default is 1 (consecutive frames).

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        self.ate = self._compute_ate()
        self.rpe = self._compute_rpe(delta=delta)
        return self

    # ── Step 5 – Print results ────────────────────────────────────────────────

    def print_results(self) -> 'TrajectoryEvaluator':
        """
        Prints a formatted summary of ATE and RPE metrics.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        print('\n' + '=' * 55)
        print('   SfM TRAJECTORY EVALUATION RESULTS')
        print('=' * 55)
        print(f'  Evaluated pairs      : {len(self.matches)}')
        print(f'  Estimated scale      : {self.s:.4f}')
        print()
        print('  ATE (Absolute Trajectory Error):')
        print(f'    RMSE     = {self.ate["rmse"]:.4f} m')
        print(f'    Mean     = {self.ate["mean"]:.4f} m')
        print(f'    Median   = {self.ate["median"]:.4f} m')
        print(f'    Std      = {self.ate["std"]:.4f} m')
        print(f'    Max      = {self.ate["max"]:.4f} m')
        print()
        print('  RPE (Relative Pose Error, delta=1):')
        print(f'    Trans RMSE   = {self.rpe["trans_rmse"]:.4f} m')
        print(f'    Trans Mean   = {self.rpe["trans_mean"]:.4f} m')
        print(f'    Rot RMSE     = {self.rpe["rot_rmse_deg"]:.3f} °')
        print(f'    Rot Mean     = {self.rpe["rot_mean_deg"]:.3f} °')
        print('=' * 55)
        return self

    # ── Step 6 – Plot ─────────────────────────────────────────────────────────

    def plot(self) -> 'TrajectoryEvaluator':
        """
        Saves 6 individual evaluation plots as PNG files on a white background.

        Output files
        ------------
        plot_trajectory_3d.png      — 3D trajectory comparison
        plot_trajectory_topdown.png — Top-down X-Z view
        plot_ate_per_frame.png      — Per-frame ATE
        plot_rpe_translation.png    — RPE translation per frame pair
        plot_rpe_rotation.png       — RPE rotation per frame pair
        plot_ate_histogram.png      — ATE error distribution

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        S          = self._STYLE
        frame_ids  = [tp for tp, _ in self.matches]

        # ── 1. 3D Trajectory ─────────────────────────────────────────────────
        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection='3d')
        ax.plot(self.gt_pts[:, 0], self.gt_pts[:, 1], self.gt_pts[:, 2],
                f'{S["gt"]["marker"]}{S["gt"]["ls"]}',
                color=S['gt']['color'], linewidth=S['lw'],
                markersize=S['ms'], label=S['gt']['label'])
        ax.plot(self.pred_aligned[:, 0], self.pred_aligned[:, 1], self.pred_aligned[:, 2],
                f'{S["pred"]["marker"]}{S["pred"]["ls"]}',
                color=S['pred']['color'], linewidth=S['lw'],
                markersize=S['ms'], label=S['pred']['label'])
        ax.set_title('3D Trajectory', fontsize=12)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_trajectory_3d.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 2. Top-down view (XZ plane) ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(self.gt_pts[:, 0], self.gt_pts[:, 2],
                f'{S["gt"]["marker"]}{S["gt"]["ls"]}',
                color=S['gt']['color'], linewidth=S['lw'],
                markersize=S['ms'], label=S['gt']['label'])
        ax.plot(self.pred_aligned[:, 0], self.pred_aligned[:, 2],
                f'{S["pred"]["marker"]}{S["pred"]["ls"]}',
                color=S['pred']['color'], linewidth=S['lw'],
                markersize=S['ms'], label=S['pred']['label'])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
        ax.set_title('Top-down View (X-Z Plane)', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_trajectory_topdown.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 3. ATE per frame ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(frame_ids, self.ate['errors'], color='#e6a817', linewidth=S['lw'])
        ax.axhline(self.ate['rmse'], **S['rmse'],
                   label=f'RMSE = {self.ate["rmse"]:.4f} m')
        ax.set_xlabel('Frame ID'); ax.set_ylabel('ATE (m)')
        ax.set_title('ATE per Frame', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_ate_per_frame.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 4. RPE translation ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.rpe['trans_errors'], color='#2ca02c', linewidth=S['lw'])
        ax.axhline(self.rpe['trans_rmse'], **S['rmse'],
                   label=f'RMSE = {self.rpe["trans_rmse"]:.4f} m')
        ax.set_xlabel('Frame pair'); ax.set_ylabel('RPE translation (m)')
        ax.set_title('RPE – Translation', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_rpe_translation.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 5. RPE rotation ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.degrees(self.rpe['rot_errors']), color='#9467bd', linewidth=S['lw'])
        ax.axhline(self.rpe['rot_rmse_deg'], **S['rmse'],
                   label=f'RMSE = {self.rpe["rot_rmse_deg"]:.3f}°')
        ax.set_xlabel('Frame pair'); ax.set_ylabel('RPE rotation (°)')
        ax.set_title('RPE – Rotation', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_rpe_rotation.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── 6. ATE histogram ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(self.ate['errors'], bins=30,
                color='#1f77b4', alpha=0.8, edgecolor='white')
        ax.axvline(self.ate['rmse'], **S['rmse'],
                   label=f'RMSE = {self.ate["rmse"]:.4f} m')
        ax.set_xlabel('ATE (m)'); ax.set_ylabel('Count')
        ax.set_title('ATE Distribution', fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'plot_ate_histogram.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f'Saved 6 evaluation plots → {self.output_dir}')
        return self

    # ── Step 7 – Save JSON ────────────────────────────────────────────────────

    def save_json(self, filename: str = 'trajectory_metrics.json') -> 'TrajectoryEvaluator':
        """
        Persists all computed metrics to a JSON file.

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'trajectory_metrics.json'.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        results = {
            'num_matched_frames' : len(self.matches),
            'scale_estimated'    : float(self.s),
            'ATE': {
                'rmse_m'   : self.ate['rmse'],
                'mean_m'   : self.ate['mean'],
                'median_m' : self.ate['median'],
                'std_m'    : self.ate['std'],
                'max_m'    : self.ate['max'],
            },
            'RPE_delta1': {
                'trans_rmse_m' : self.rpe['trans_rmse'],
                'trans_mean_m' : self.rpe['trans_mean'],
                'rot_rmse_deg' : self.rpe['rot_rmse_deg'],
                'rot_mean_deg' : self.rpe['rot_mean_deg'],
            },
        }

        out_path = self.output_dir / filename
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f'Metrics saved → {out_path}')
        print(json.dumps(results, indent=2))
        return self

    # ── Convenience: run full pipeline ───────────────────────────────────────

    def run(self, with_scale: bool = True,
            rpe_delta: int = 1) -> 'TrajectoryEvaluator':
        """
        Runs the full evaluation pipeline in one call.

        Equivalent to:
            .load().associate().align().evaluate().print_results().plot().save_json()

        Parameters
        ----------
        with_scale : bool, optional
            Use Sim(3) with scale. Default is True.
        rpe_delta : int, optional
            Frame spacing for RPE. Default is 1.

        Returns
        -------
        self : TrajectoryEvaluator
            For method chaining.
        """
        return (self
                .load()
                .associate()
                .align(with_scale=with_scale)
                .evaluate(delta=rpe_delta)
                .print_results()
                .plot()
                .save_json())