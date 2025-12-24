import threading
import time
from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque
from utils.file import BASE_DIR

AFTER_FRAMES_CLEAR = 9  # 默认 9 帧未更新 cache 就将其重置
MAX_FACE_CACHES_LENGTH = 6


@dataclass(slots=True)
class FaseCache:
    box : Tuple[int, int, int, int]
    is_photo : bool
    timestamp_ms : int


class LabDetector:
    def __init__(self):
        self._face_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root=BASE_DIR)
        self._face_detector.prepare(ctx_id=0, det_size=(640, 640))
        self._face_caches = deque(maxlen=MAX_FACE_CACHES_LENGTH)
        self._start_time = time.time()  # 单位 s

    def get_face_roi(self, frame_bgr8, width: int, height: int) -> np.ndarray | None:
        _box = self._get_face_box()
        if not _box:
            return None
        tgt_ratio = width / height
        _x1, _y1, _x2, _y2 = _box
        img_h, img_w = frame_bgr8[:2]
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
        # 裁剪并缩放，不改颜色
        roi_bgr8 = frame_bgr8[_y1:_y2, _x1:_x2]
        roi_bgr8 = cv2.resize(roi_bgr8, (width, height), interpolation=cv2.INTER_LINEAR)
        return roi_bgr8

    def detect_once(self, _frame : np.ndarray) -> None:
        """
        检测人脸
        :param _frame: BGR8格式图片
        :return: None
        """
        threading.Thread(target=self._detect_thread(_frame), daemon=True).start()

    def _detect_thread(self, _frame : np.ndarray):
        faces = self._face_detector.get(_frame)
        if faces is None or len(faces) == 0:
            return

        timestamp_ms = int((time.time() - self._start_time) * 1000)  # 单位 ms
        best_face = max(faces, key=lambda f: f['det_score'])
        bbox = best_face['bbox']
        _x1, _y1, _x2, _y2 = bbox
        _face_cache = FaseCache(
            box=(_x1, _y1, _x2, _y2),
            is_photo=False,
            timestamp_ms=timestamp_ms,
        )
        if not self._face_caches:
            self._face_caches.append(_face_cache)
            return
        else:
            last_cache: FaseCache = self._face_caches[-1]
            if _face_cache.timestamp_ms - last_cache.timestamp_ms > 33 * AFTER_FRAMES_CLEAR:
                self._face_caches.clear()
            self._face_caches.append(_face_cache)
            return

    def _get_face_box(self) -> tuple[int, int, int, int] | None:
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
        face_box = self._get_face_box()
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
        ret, frame = cap.read()
        if not ret: raise RuntimeError("读取摄像头失败")
        detector.detect_once(frame)
        detector.draw_face_rectangle(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

