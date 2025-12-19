import cv2
import queue
import random
import numpy as np
import mediapipe as mp
from utils.stream import frame_z16_to_points
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, RunningMode, FaceDetectorOptions

# 配置模型路径（Tasks 官方预训练模型）
MODEL_PATH = './models/blaze_face_short_range.tflite'

class Detector():
    def __init__(self):
        # 创建检测器实例
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,  # 单张图片模式
            min_detection_confidence=0.75  # 可调阈值
        )
        self._detector = FaceDetector.create_from_options(options)
        self._boxes_cache = []

    def judge_person_real_or_photo(self, frame_z16, profile):
        """
        判断是否为真人或照片
        :param inliers_3d: 输入的3D点集
        :return: 'photo' 或 'person'
        """
        res = []
        # 绘制每一个人脸框图
        for box_queue in self._boxes_cache:
            assert isinstance(box_queue, queue.Queue)
            (x1, y1, x2, y2) = self.get_smooth_box(box_queue)
            frame_pts = frame_z16_to_points(frame_z16[y1:y2, x1:x2], profile=profile)
            plane_flag = self.plane_fit_ransac_simplified(frame_pts)
            if plane_flag:
                res.append(False)
                continue
            res.append(True)
        return res

    def plane_fit_ransac_simplified(self, points_3d, dist_thresh=0.02, ratio_thresh=0.8, max_iter=80):
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
                    return True  # 如果内点数量超过阈值，判定为平面
        ratio_val = best_inliers / float(N)
        return ratio_val >= ratio_thresh

    def get_smooth_box(self, q: queue.Queue, alpha: float = 0.7):
        """
        对队列里的所有框做指数平滑，返回平滑后的 (x1,y1,x2,y2)
        如果队列未满直接返回最新框，避免启动阶段没数据
        """
        if q.qsize() == 0:
            return None
        # 转成列表但不弹出
        boxes = list(q.queue)  # queue.Queue -> list
        if not boxes:
            return None
        boxes = np.array(boxes, dtype=np.float32)  # shape=(N,4)
        # 指数平滑：最新帧权重最大
        weight = np.array([alpha ** (len(boxes) - 1 - i) for i in range(len(boxes))])
        weight /= weight.sum()
        smooth = (boxes.T @ weight).astype(int)  # (4,)
        return tuple(smooth)  # (x1,y1,x2,y2)

    def draw_face_rect(self, frame : np.ndarray, type : str):
        """
        传入 frame 单帧图片上画 face 框
        :param frame: 传入图片
        :param type: 图片格式 bgr8, rgb8
        """
        if len(self._boxes_cache) == 0:
            return
        if type == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 绘制每一个人脸框图
        for box_queue in self._boxes_cache:
            assert isinstance(box_queue, queue.Queue)
            (x1, y1, x2, y2) = self.get_smooth_box(box_queue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 保证返回格式不变
        if type == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            assert isinstance(frame, np.ndarray)

    def face_detect(self, frame_bgr8: np.ndarray) -> bool:
        """
        输入:  frame_bgr8: numpy数组 (h, w, 3), frame_z16: 深度图像
        返回:  flag: 只要检测到至少一张人脸就返回 True
        """
        # BGR -> RGB，并包装成 MediaPipe Image 用于模型推理
        rgb = cv2.cvtColor(frame_bgr8, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image( image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # 调用模型推理
        _result = self._detector.detect(mp_image)
        # 解析结果
        for idx, det in enumerate(_result.detections):
            bbox = det.bounding_box
            _x1 = int(bbox.origin_x)
            _y1 = int(bbox.origin_y)
            _x2 = int(bbox.origin_x + bbox.width)
            _y2 = int(bbox.origin_y + bbox.height)
            box = (_x1, _y1, _x2, _y2)
            # 更新人脸检测框
            if len(self._boxes_cache) < idx + 1:
                _box_queue = queue.Queue(maxsize=60)
                _box_queue.put(box)
                self._boxes_cache.append(_box_queue)
            else:
                assert isinstance(self._boxes_cache[idx], queue.Queue)
                if self._boxes_cache[idx].full():
                    self._boxes_cache[idx].get()
                self._boxes_cache[idx].put(box)
        self._boxes_cache = self._boxes_cache[:len(_result.detections)]
        return len(_result.detections) > 0

if __name__ == '__main__':
    detector = Detector()
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # DSHOW 加速 Windows 打开
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头')
    while True:
        ok, frame = cap.read()
        if not ok:
            print('读取失败')
            break
        result = detector.face_detect(frame)
        if result:  # 不为空
            detector.draw_face_rect(frame, "bgr8")

        cv2.imshow('Camera 640x480', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


