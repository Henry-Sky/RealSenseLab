import random
import threading
import time
from dataclasses import dataclass
from typing import Tuple
import cv2
from pyrealsense2 import stream
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque
from utils.file import BASE_DIR
from pyrealsense2 import stream, rs2_deproject_pixel_to_point, rs2_project_point_to_pixel, distortion, intrinsics

AFTER_FRAMES_CLEAR = 9  # 默认 9 帧未更新 cache 就将其重置
MAX_FACE_CACHES_LENGTH = 6


@dataclass(slots=True)
class FaceInfo:
    bbox : Tuple[int, int, int, int]
    is_photo : bool
    timestamp_ms : int


class LabDetector:
    def __init__(self):
        self.device_profile = None
        self._face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=BASE_DIR)
        self._face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self._face_caches = deque(maxlen=MAX_FACE_CACHES_LENGTH)
        self._start_time = time.time()  # 单位 s

    def get_face_roi(self, frame_bgr8, width: int, height: int) -> np.ndarray | None:
        if not self._get_smooth_face_info():
            return None
        _bbox = self._get_smooth_face_info().bbox
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
        return roi_bgr8

    def detect_once(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray) -> None:
        threading.Thread(target=self._detect_thread(_frame_bgr8, _frame_z16), daemon=True).start()

    def _detect_thread(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray):
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
        _face_cache = FaceInfo(
            bbox=(_x1, _y1, _x2, _y2),
            is_photo=self._photo_judge(_frame_z16, (_x1, _y1, _x2, _y2)),
            timestamp_ms=timestamp_ms,
        )
        if not self._face_caches:
            self._face_caches.append(_face_cache)
            return
        else:
            last_cache: FaceInfo = self._face_caches[-1]
            if _face_cache.timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
                self._face_caches.clear()
            self._face_caches.append(_face_cache)
            return

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
        is_photo = last_cache.is_photo
        dt = times[-1] - times  # 与最新帧的差(ms)
        alpha = 0.7  # 衰减系数，可调
        weight = alpha ** (dt / 33.)  # 33 ms ≈ 30 fps 基准
        weight /= weight.sum()  # 归一化
        _x1, _y1, _x2, _y2 = (boxes[:, :4].T @ weight).astype(int)  # 只平滑 x1,y1,x2,y2
        return FaceInfo(
            bbox=(_x1, _y1, _x2, _y2),
            timestamp_ms=timestamp_ms,
            is_photo=is_photo,
        )

    def _photo_judge(self, frame_z16, bbox):
        """True:photo, False:person"""
        x1, y1, x2, y2 = bbox
        frame_pts = self.frame_z16_to_points(frame_z16[y1:y2, x1:x2])
        plane_flag = self.plane_fit_ransac_simplified(frame_pts)
        if plane_flag:
            print("plane_flag is True")
            return True
        # 分布检查
        xyz_min = np.min(frame_pts, axis=0)
        xyz_max = np.max(frame_pts, axis=0)
        diff_xyz = xyz_max - xyz_min
        if diff_xyz[0] < 0.02 and diff_xyz[1] < 0.02 and diff_xyz[2] < 0.02:
            print("diff_xyz is too small")
            return True
        return False

    def plane_fit_ransac_simplified(self, points_3d, dist_thresh=0.005, ratio_thresh=0.8, max_iter=30):
        """
        使用简化的RANSAC算法判断3D点是否平面化
        :param points_3d: 输入的3D点集
        :param dist_thresh: 距离阈值，默认值为0.02
        :param ratio_thresh: 内点比例阈值，默认值为0.8
        :param max_iter: 最大迭代次数，默认值为80
        :return: 如果点集平面化则返回True，否则返回False
        """
        N = len(points_3d)
        if N < 30:
            print("points_3d is too small")
            return True  # 如果点数太少，直接判定为平面
        best_inliers = 0
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
                    print(f"best_inliers is {best_inliers} target_count is {target_count} N is {N}")
                    return True  # 如果内点数量超过阈值，判定为平面
        ratio_val = best_inliers / float(N)
        print(f"ratio_val is {ratio_val} ratio_thresh is {ratio_thresh} N is {N}")
        return ratio_val >= ratio_thresh

    def frame_z16_to_points(self, frame_z16 : np.ndarray, depth_scale=None, intrin=None):
        """
        把 depth_frame_z16
        返回: points_3d  shape=(H*W, 3), dtype=float64
        """
        if self.device_profile is not None:
            depth_scale = self.device_profile.get_device().first_depth_sensor().get_depth_scale()
            intrin = self.device_profile.get_stream(stream.depth).as_video_stream_profile().get_intrinsics()
        else:
            if depth_scale is None or intrin is None:
                return None
        H, W = frame_z16.shape[:2]
        z = frame_z16 * depth_scale  # 米
        # 构建像素坐标网格
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32), indexing='xy')
        # 反投影到相机坐标系
        x = (u - intrin.ppx) * z / intrin.fx
        y = (v - intrin.ppy) * z / intrin.fy
        # 堆叠成 N×3
        points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        # 去掉 0 深度无效点
        valid = points_3d[:, 2] > 0
        return points_3d[valid]

    def draw_face_rectangle(self, frame: np.ndarray) -> None:
        face_info = self._get_smooth_face_info()
        if face_info is not None:
            x1, y1, x2, y2 = face_info.bbox
            if face_info.is_photo:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_body_rect(self,
                       frame_bgr8: np.ndarray,
                       frame_z16: np.ndarray):
        """
        在 1280×720 BGR 图像上绘制人体 3-D AABB 框
        参数:
            frame_bgr8: 8-bit BGR,  shape (H, W, 3)
            frame_z16:  16-bit 深度（mm）, shape (H, W)
        """
        face_info = self._get_smooth_face_info()
        if not face_info or face_info.is_photo or not self.device_profile:
            return

        # 1. 动态读取实际分辨率
        H, W = frame_bgr8.shape[:2]  # 720p -> (720, 1280)
        z_h, z_w = frame_z16.shape[:2]  # 可能是 480p(640,480) 或 720p(1280,720)

        # 2. 把深度图 resize 到与彩色图完全一致（最近邻保边）
        if (z_h, z_w) != (H, W):
            frame_z16 = cv2.resize(frame_z16, (W, H), interpolation=cv2.INTER_NEAREST)
        depth32 = frame_z16.astype(np.float32)

        # 3. 构造与彩色图同分辨率的内参
        intr = intrinsics()
        intr.width, intr.height = W, H
        intr.ppx, intr.ppy = W * 0.5, H * 0.5  # 主点居中
        intr.fx = intr.fy = 424.0 * (W / 640.)  # 按水平分辨率等比放大 fx/fy
        intr.model = distortion.brown_conrady
        intr.coeffs = [0] * 5

        # 4. 人脸中心作为 flood-fill 种子
        x1, y1, x2, y2 = face_info.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        seed_z = depth32[cy, cx]
        if seed_z == 0:
            return

        # 5. floodFill 得 mask
        mask = np.zeros((H + 2, W + 2), np.uint8)
        cv2.floodFill(depth32, mask, (cx, cy), 0,
                      loDiff=50, upDiff=50,
                      flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        mask = mask[1:-1, 1:-1]
        if np.count_nonzero(mask) < 500:
            return

        # 6. 3-D 点云
        ys, xs = np.where(mask)
        zs = frame_z16[ys, xs] / 1000.0
        pts_cam = np.array([rs2_deproject_pixel_to_point(intr, [u, v], z)
                            for u, v, z in zip(xs, ys, zs)], dtype=np.float32)

        # 7. PCA 求主方向 + AABB
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

        # 8. 投影回 2-D 并画框
        pts2d = np.array([rs2_project_point_to_pixel(intr, p)
                          for p in corners_cam]).astype(int)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            cv2.line(frame_bgr8, tuple(pts2d[i]), tuple(pts2d[j]),
                     (0, 255, 0), 2)



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

