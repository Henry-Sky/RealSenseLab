import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetectorOptions, RunningMode, FaceDetector, FaceDetectorResult
from collections import deque
from utils.file import lab_read_path

AFTER_FRAMES_CLEAR = 9  # 默认 9 帧未更新 cache 就将其重置


@dataclass(slots=True)
class FaseCache:
    box : Tuple[int, int, int, int]
    is_photo : bool
    timestamp_ms : int


class LabDetector:
    def __init__(self):
        model_path = lab_read_path("models/blaze_face_short_range.tflite")
        fd_options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path = model_path),
            running_mode=RunningMode.LIVE_STREAM,
            min_detection_confidence=0.6,
            result_callback=self._face_result_callback,
        )
        self._face_detector = FaceDetector.create_from_options(fd_options)
        self._face_caches = deque(maxlen=16)
        self._start_time = time.time()  # 单位 s

    def _face_result_callback(self, result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        if not result.detections:
            return
        # 取可信度最高的一张人脸
        best_detection = max(result.detections, key=lambda d: d.categories[0].score)
        bbox = best_detection.bounding_box
        _x1 = int(bbox.origin_x)
        _y1 = int(bbox.origin_y)
        _x2 = int(bbox.origin_x + bbox.width)
        _y2 = int(bbox.origin_y + bbox.height)
        _face_cache = FaseCache(
            box=(_x1, _y1, _x2, _y2),
            is_photo=False,
            timestamp_ms=timestamp_ms,
        )
        if not self._face_caches:
            self._face_caches.append(_face_cache)
            return
        else:
            last_cache : FaseCache = self._face_caches[-1]
            if _face_cache.timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
                self._face_caches.clear()
            self._face_caches.append(_face_cache)
            return

    def detect_once(self, _frame : np.ndarray) -> None:
        """
        检测人脸
        :param _frame: BGR8格式图片
        :return: None
        """
        rgb = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        self._face_detector.detect_async(mp_image, timestamp_ms)

    def get_face_box(self) -> tuple[int, int, int, int] | None:
        """
        按与最新帧的时间差指数衰减加权，越新框权重越大
        :return: x1, y1, x2, y2
        """
        if not self._face_caches:
            return None
        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        last_cache: FaseCache = self._face_caches[-1]
        if timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
            self._face_caches.clear()
            return None
        # 拆解 face_cache 的 box 和 timestamp_ms
        boxes = np.array([x.box for x in self._face_caches], dtype=np.float32)
        times = np.array([x.timestamp_ms for x in self._face_caches])  # 时间戳列
        dt = times[-1] - times  # 与最新帧的差(ms)
        alpha = 0.7  # 衰减系数，可调
        weight = alpha ** (dt / 33.)  # 33 ms ≈ 30 fps 基准
        weight /= weight.sum()  # 归一化
        _x1, _y1, _x2, _y2 = (boxes[:, :4].T @ weight).astype(int)  # 只平滑 x1,y1,x2,y2
        return _x1, _y1, _x2, _y2  # (x1,y1,x2,y2)

    def draw_face_rectangle(self, frame: np.ndarray) -> None:
        face_box = self.get_face_box()
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



if __name__ == '__main__':
    detector = LabDetector()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError('无法打开摄像头')
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_cnt = 0  # 帧计数器
    fps_delay = int(1000 / 30)  # 33 ms ≈ 30 fps

    while True:
        frame_cnt += 1
        ret, frame = cap.read()
        if not ret: raise RuntimeError("读取摄像头失败")
        if frame_cnt % 3 == 0:
            detector.detect_once(frame)
        box = detector.get_face_box()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

