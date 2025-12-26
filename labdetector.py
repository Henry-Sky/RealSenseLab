import random
import threading
import time
from dataclasses import dataclass
from email.iterators import body_line_iterator
from typing import Tuple, Any
import cv2
import numpy as np
from cv2 import Mat, UMat
from insightface.app import FaceAnalysis
from collections import deque

from numpy import ndarray, dtype, integer, floating

from utils.file import BASE_DIR
from pyrealsense2 import rs2_deproject_pixel_to_point, rs2_project_point_to_pixel, distortion, intrinsics

AFTER_FRAMES_CLEAR = 9  # 默认 9 帧未更新 cache 就将其重置
MAX_FACE_CACHES_LENGTH = 9


@dataclass(slots=True)
class FaceInfo:
    bbox : Tuple[int, int, int, int]
    is_photo : bool
    body_lines : np.ndarray | None
    timestamp_ms : int


class LabDetector:
    def __init__(self):
        self._device_intr = self._init_device_intr()
        self._face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=BASE_DIR)
        self._face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self._face_caches = deque(maxlen=MAX_FACE_CACHES_LENGTH)
        self._start_time = time.time()  # 单位 s

    def _init_device_intr(self):
        intr = intrinsics()
        intr.width, intr.height = 1280, 720
        intr.ppx, intr.ppy = 640.0, 360.0
        intr.fx = intr.fy = 848
        intr.model = distortion.brown_conrady
        intr.coeffs = [0] * 5
        return intr

    def get_face_roi(self, frame_bgr8, width: int, height: int) -> tuple[Mat | ndarray | UMat, bool] | None:
        _face_info = self._get_smooth_face_info()
        if not _face_info:
            return None
        _bbox = _face_info.bbox
        tgt_ratio = width / height
        _x1, _y1, _x2, _y2 = _bbox
        img_h, img_w = frame_bgr8.shape[:2]
        fw, fh = _x2 - _x1, _y2 - _y1
        src_ratio = fw / fh
        # 原框更宽 -> 裁宽
        if src_ratio > tgt_ratio:
            new_fw = int(fh * tgt_ratio)
            dw = fw - new_fw
            _x1 += dw // 2
            _x2 -= dw - dw // 2
        # 原框更高 -> 裁高
        else:
            new_fh = int(fw / tgt_ratio)
            dh = fh - new_fh
            _y1 += dh // 2
            _y2 -= dh - dh // 2
        _x1 = max(0, int(_x1))
        _y1 = max(0, int(_y1))
        _x2 = min(int(img_w), int(_x2))
        _y2 = min(int(img_h), int(_y2))
        # 裁剪并缩放，不改颜色
        roi_bgr8 = frame_bgr8[_y1:_y2, _x1:_x2]
        roi_bgr8 = cv2.resize(roi_bgr8, (width, height), interpolation=cv2.INTER_LINEAR)
        return roi_bgr8, _face_info.is_photo

    def detect_once(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray = None, body_calc = False) -> None:
        threading.Thread(target=self._detect_thread(_frame_bgr8, _frame_z16, body_calc), daemon=True).start()

    def _detect_thread(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray, body_calc) -> None:
        faces = self._face_detector.get(_frame_bgr8)
        if faces is None or len(faces) == 0:
            return

        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        best_face = max(faces, key=lambda f: f['det_score'])
        bbox = best_face['bbox']
        _x1, _y1, _x2, _y2 = bbox
        _x1 = int(_x1)
        _y1 = int(_y1)
        _x2 = int(_x2)
        _y2 = int(_y2)
        photo_flag = self._photo_judge(_frame_z16, (_x1, _y1, _x2, _y2)) if _frame_z16 is not None else True
        body_lines = self._detect_body_edges(_frame_bgr8, _frame_z16, bbox) if body_calc and not photo_flag else None
        _face_info = FaceInfo(
            bbox=(_x1, _y1, _x2, _y2),
            is_photo=photo_flag,
            body_lines=body_lines,
            timestamp_ms=timestamp_ms,
        )
        if not self._face_caches:
            self._face_caches.append(_face_info)
        else:
            last_cache: FaceInfo = self._face_caches[-1]
            if _face_info.timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
                self._face_caches.clear()
            self._face_caches.append(_face_info)

    def _get_smooth_face_info(self) -> FaceInfo | None:
        """
        按与最新帧的时间差指数衰减加权，越新框权重越大
        :return: x1, y1, x2, y2
        """
        if not self._face_caches:
            return None
        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        last_cache: FaceInfo = self._face_caches[-1]
        if timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
            self._face_caches.clear()
            return None
        # 拆解 face_cache 的 box 和 timestamp_ms
        boxes = np.array([x.bbox for x in self._face_caches], dtype=np.float32)
        times = np.array([x.timestamp_ms for x in self._face_caches])  # 时间戳列
        photos = np.array([x.is_photo for x in self._face_caches], dtype=np.float32)
        dt = times[-1] - times  # 与最新帧的差(ms)
        alpha = 0.7  # 衰减系数，可调
        weight = alpha ** (dt / 33.)  # 33 ms ≈ 30 fps 基准
        weight /= weight.sum()  # 归一化
        _x1, _y1, _x2, _y2 = (boxes[:, :4].T @ weight).astype(int)  # 只平滑 x1,y1,x2,y2
        smooth_photo = bool(np.round(photos @ weight))
        return FaceInfo(
            bbox=(_x1, _y1, _x2, _y2),
            body_lines=last_cache.body_lines,
            timestamp_ms=timestamp_ms,
            is_photo=last_cache.is_photo,
        )

    def roi_to_points(self, frame_z16, roi_xyxy):
        """返回 (N,3) 单位米，已剔除 NaN"""
        H, W = frame_z16.shape[:2]
        x1, y1, x2, y2 = np.clip(roi_xyxy, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
        roi_z = frame_z16[y1:y2, x1:x2]
        uy, ux = np.where(roi_z > 0)  # 只取>0
        if uy.size == 0:
            return np.empty((0, 3), np.float32)

        u = (x1 + ux).astype(np.float32)
        v = (y1 + uy).astype(np.float32)
        z = roi_z[uy, ux] * 0.001  # mm->m

        pts = np.empty((u.size, 3), np.float32)
        for i in range(u.size):
            pts[i] = rs2_deproject_pixel_to_point(self._device_intr, [u[i], v[i]], z[i])

        # 把 NaN 整行干掉
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts

    def _photo_judge(self, frame_z16, bbox):
        """True:photo, False:person"""
        frame_pts = self.roi_to_points(frame_z16, bbox)
        plane_flag = self.plane_fit_ransac_simplified(frame_pts)
        if plane_flag:
            # print("plane_flag is True")
            return True
        # 分布检查
        xyz_min = np.min(frame_pts, axis=0)
        xyz_max = np.max(frame_pts, axis=0)
        diff_xyz = xyz_max - xyz_min
        if diff_xyz[0] < 0.02 and diff_xyz[1] < 0.02 and diff_xyz[2] < 0.02:
            # print("diff_xyz is too small")
            return True
        return False

    def plane_fit_ransac_simplified(self, points_3d, dist_thresh=0.005, ratio_thresh=0.8, max_iter=30):
        """
        使用简化的RANSAC算法判断3D点是否平面化
        :param points_3d: 输入的3D点集
        :param dist_thresh: 距离阈值，默认值为0.01
        :param ratio_thresh: 内点比例阈值，默认值为0.8
        :param max_iter: 最大迭代次数，默认值为80
        :return: 如果点集平面化则返回True，否则返回False
        """
        # 去掉 NaN/Inf
        points_3d = np.asarray(points_3d)
        points_3d = points_3d[~np.isnan(points_3d).any(axis=1)]
        points_3d = points_3d[~np.isinf(points_3d).any(axis=1)]

        N = len(points_3d)
        if N < 30:
            # print("points_3d is too small")
            return True  # 如果点数太少，直接判定为平面
        best_inliers = 0.0
        target_count = int(ratio_thresh * N)
        idxs = np.arange(N)
        for _ in range(max_iter):
            i1, i2, i3 = random.sample(list(idxs), 3)  # 随机抽样3个点
            p1, p2, p3 = points_3d[i1], points_3d[i2], points_3d[i3]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)  # 计算法向量
            if np.linalg.norm(normal) < 1e-6:
                continue  # 如果法向量太小，跳过
            d = -np.dot(normal, p1)
            denom = np.linalg.norm(normal)
            dist_all = np.abs(np.dot(points_3d, normal) + d) / denom  # 计算所有点到平面的距离
            inliers = np.sum(dist_all <= dist_thresh)  # 计算内点数量
            if inliers > best_inliers:
                best_inliers = inliers
                if best_inliers >= target_count:
                    # print(f"best_inliers is {best_inliers}, target_count is {target_count}, N is {N}")
                    return True  # 如果内点数量超过阈值，判定为平面
        ratio_val = best_inliers / float(N)
        # print(f"ratio_val is {ratio_val}, ratio_thresh is {ratio_thresh}, N is {N}")
        return ratio_val >= ratio_thresh

    def draw_face_rectangle(self, frame_bgr8: np.ndarray, face_info = None) -> None:
        if not face_info:
            face_info = self._get_smooth_face_info()
            if face_info is None:
                return
        x1, y1, x2, y2 = face_info.bbox
        if face_info.is_photo:
            cv2.rectangle(frame_bgr8, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame_bgr8, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def _detect_body_edges(self, frame_bgr8: np.ndarray, frame_z16 : np.ndarray, bbox : Tuple[int, int, int, int]) -> None | np.ndarray:
        print("body detecting")
        img_h, img_w = frame_bgr8.shape[:2]
        intr = self._device_intr  # 获取设备内参
        depth32 = frame_z16.astype(np.float32)
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) // 2)
        cy = int((y1 + y2) // 2)
        seed_z = depth32[cy, cx]
        if seed_z == 0:
            return None
        # floodFill 泛洪得 mask
        mask = np.zeros((img_h + 2, img_w + 2), np.uint8)
        cv2.floodFill(depth32, mask, (cx, cy), 0,
                      loDiff=50, upDiff=50,
                      flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        mask = mask[1:-1, 1:-1]
        if np.count_nonzero(mask) < 500:
            return None
        # 3-D 点云
        ys, xs = np.where(mask)
        zs = frame_z16[ys, xs] / 1000.0
        pts_cam = np.array([rs2_deproject_pixel_to_point(intr, [u, v], z)
                            for u, v, z in zip(xs, ys, zs)], dtype=np.float32)
        # PCA 求主方向 + AABB包围盒8点坐标
        mean, eig_vec = cv2.PCACompute(pts_cam, mean=None)
        eig_vec = eig_vec[:3]
        pts_local = (pts_cam - mean) @ eig_vec.T
        local_min = pts_local.min(axis=0)
        local_max = pts_local.max(axis=0)
        dx, dy, dz = local_max - local_min
        corners_local = np.array([
            [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
            [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]
        ]) + local_min
        corners_cam = (corners_local @ eig_vec) + mean
        # 投影回 2-D 并画框
        pts2d = np.array([rs2_project_point_to_pixel(intr, p)
                          for p in corners_cam]).astype(int)
        edges_idx = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        lines = np.asarray([[pts2d[i], pts2d[j]] for i, j in edges_idx], dtype=np.int32)
        return lines

    def draw_body_rectangle(self, frame_bgr8: np.ndarray, face_info = None) -> None:
        """
        在 1280×720 BGR 图像上绘制人体 3-D AABB 包围框
        参数:
            frame_bgr8: 8-bit BGR,  shape (H, W, 3)
            face_info:  FaceInfo 人脸作为泛洪种子
        """
        if not face_info:
            face_info = self._get_smooth_face_info()
            if not face_info:
                return
        if face_info.is_photo:
            return

        lines = face_info.body_lines
        if lines is not None:
            for (p0, p1) in lines:
                cv2.line(frame_bgr8, tuple(p0), tuple(p1), (255, 0, 0), 2)


if __name__ == '__main__':
    detector = LabDetector()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头')
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_cnt = 0  # 帧计数器
    fps_delay = int(1000 / 30)  # 33 ms ≈ 30 fps

    while True:
        ret, frame = cap.read()
        if not ret: raise RuntimeError("读取摄像头失败")
        detector.detect_once(frame)
        detector.draw_face_rectangle(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

