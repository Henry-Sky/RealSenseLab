import cv2
import time
import random
import threading
import numpy as np
from typing import Tuple
from collections import deque
from dataclasses import dataclass
from insightface.model_zoo import get_model
from utils.file import BASE_DIR
from pyrealsense2 import rs2_deproject_pixel_to_point, rs2_project_point_to_pixel, distortion, intrinsics

AFTER_FRAMES_CLEAR = 9  # 默认 9 帧未更新 cache 就将其重置
MAX_FACE_CACHES_LENGTH = 9  # 最大检测结果缓冲队列长度


@dataclass(slots=True)
class FaceInfo:
    bbox : Tuple[int, int, int, int]
    is_photo : bool
    body_lines : np.ndarray | None
    timestamp_ms : int


class LabDetector:
    def __init__(self):
        self._device_intr = self._init_device_intr()
        self._face_detector = get_model('models/buffalo_m/det_2.5g.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=BASE_DIR)
        self._face_detector.prepare(ctx_id=0, input_size=(256, 256))
        self._face_caches = deque(maxlen=MAX_FACE_CACHES_LENGTH)
        self._start_time = time.time()  # 单位 s

    def _init_device_intr(self):
        """获取设备内参"""
        intr = intrinsics()
        intr.width, intr.height = 1280, 720
        intr.ppx, intr.ppy = 640.0, 360.0
        intr.fx = intr.fy = 848
        intr.model = distortion.brown_conrady
        intr.coeffs = [0] * 5
        return intr

    def detect_once(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray = None, body_calc = False) -> None:
        """创建并调用检测线程"""
        threading.Thread(target=self._detect_thread(_frame_bgr8, _frame_z16, body_calc), daemon=True).start()

    def _detect_thread(self, _frame_bgr8 : np.ndarray, _frame_z16 : np.ndarray, body_calc) -> None:
        """检测线程执行函数"""
        faces = self._face_detector.detect(_frame_bgr8)
        if faces is None or len(faces) == 0:
            return
        # 检测到人脸，进入后期处理
        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        self._face_caches.clear()
        for face in faces[0]:
            bbox = face[:4]
            _x1, _y1, _x2, _y2 = map(int, bbox)
            photo_flag = self._photo_judge(_frame_z16, (_x1, _y1, _x2, _y2)) if _frame_z16 is not None else True
            body_lines = self._calculate_body_edges(_frame_bgr8, _frame_z16,
                                                    bbox) if body_calc and not photo_flag else None
            # 添加到缓冲队列
            _face_info = FaceInfo(
                bbox=(_x1, _y1, _x2, _y2),
                is_photo=photo_flag,
                body_lines=body_lines,
                timestamp_ms=timestamp_ms,
            )
            self._face_caches.append(_face_info)

    def _roi_to_points(self, frame_z16, roi_bbox):
        """ROI区域深度Z16 -> 3D点云，已剔除 NaN"""
        img_h, img_w = frame_z16.shape[:2]
        _x1, _y1, _x2, _y2 = np.clip(roi_bbox, [0, 0, 0, 0], [img_w - 1, img_h - 1, img_w - 1, img_h - 1])
        roi_z = frame_z16[_y1:_y2, _x1:_x2]
        uy, ux = np.where(roi_z > 0)  # 只取>0
        if uy.size == 0:
            return np.empty((0, 3), np.float32)
        u = (_x1 + ux).astype(np.float32)
        v = (_y1 + uy).astype(np.float32)
        z = roi_z[uy, ux] * 0.001  # mm->m
        pts = np.empty((u.size, 3), np.float32)
        for i in range(u.size):
            pts[i] = rs2_deproject_pixel_to_point(self._device_intr, [u[i], v[i]], z[i])
        # 把 NaN 整行去除
        pts = pts[~np.isnan(pts).any(axis=1)]
        return pts

    def _photo_judge(self, frame_z16, bbox):
        """照片判断，True:photo, False:person"""
        frame_pts = self._roi_to_points(frame_z16, bbox)
        plane_flag = self._plane_fit_ransac_simplified(frame_pts)
        if plane_flag:
            return True
        # 分布检查
        xyz_min = np.min(frame_pts, axis=0)
        xyz_max = np.max(frame_pts, axis=0)
        diff_xyz = xyz_max - xyz_min
        if diff_xyz[0] < 0.02 and diff_xyz[1] < 0.02 and diff_xyz[2] < 0.02:
            return True
        return False

    def _plane_fit_ransac_simplified(self, points_3d, dist_thresh=0.005, ratio_thresh=0.8, max_iter=30):
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
        if N < 50:
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
                    return True  # 如果内点数量超过阈值，判定为平面
        ratio_val = best_inliers / float(N)
        return ratio_val >= ratio_thresh

    def _calculate_body_action(self):
        """ToDo: 细化分析深度信息，捕获人体骨架，判断行为动作 （或直接用黑盒模型）"""
        pass

    def draw_face_rectangle(self, frame_bgr8: np.ndarray) -> None:
        """绘制脸部检测框：绿色真人；红色照片"""
        for face_info in self._face_caches:
            label = 'Photo' if face_info.is_photo else 'Person'
            x1, y1, x2, y2 = face_info.bbox
            color = (0, 0, 255) if face_info.is_photo else (0, 255, 0)
            cv2.rectangle(frame_bgr8, (x1, y1), (x2, y2), color, 2)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = x2 - text_size[0] - 2
            text_y = y1 + text_size[1] + 2
            cv2.putText(frame_bgr8, label,(text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _calculate_body_edges(self, frame_bgr8: np.ndarray, frame_z16 : np.ndarray, bbox : Tuple[int, int, int, int]) -> None | np.ndarray:
        """计算人体3d边线：性能开销巨大！"""
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

    def draw_body_rectangle(self, frame_bgr8: np.ndarray) -> None:
        """绘制人体3D包围框"""
        for face_info in self._face_caches:
            lines = face_info.body_lines
            if lines is not None:
                for (p0, p1) in lines:
                    cv2.line(frame_bgr8, tuple(p0), tuple(p1), (255, 0, 0), 2)


if __name__ == '__main__':
    import timeit
    # 测量函数执行时间
    def foo():
        return sum(range(10_000_000))
    t = timeit.timeit(foo, number=5)
    print(f"平均 {t / 5 * 1000:.3f} ms/次")

