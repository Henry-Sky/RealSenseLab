import queue

import cv2
import numpy as np
from typing import List, Tuple
import mediapipe as mp
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
        self._face_threshold = []

    def _smooth_box(self, q: queue.Queue, alpha: float = 0.7):
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

    def is_photo_face(self, frame_z16: np.ndarray) -> list[bool]:
        """
        用深度方差 + Sobel 梯度判断照片 vs 活体
        返回 True  -> 真人
               False -> 照片/平板
        """
        res = []
        if len(self._boxes_cache) == 0:
            return res

        # 经验阈值，按你的深度单位调
        VAR_TH = 2000  # 深度方差阈值  (mm^2)
        GRAD_TH = 200  # 平均梯度阈值 (mm)

        for box_queue in self._boxes_cache:
            box = self._smooth_box(box_queue)
            if box is None:
                res.append(False)
                continue
            x1, y1, x2, y2 = box

            # 区域裁剪
            h, w = frame_z16.shape
            roi = frame_z16[y1:y2, x1:x2]

            # 剔除无效深度 ——
            valid = roi[(roi > 0) & (roi < 65535)]
            if valid.size < 20:  # 太少点直接判照片
                res.append(False)
                continue

            # 方差特征 ——
            var = valid.var()
            # 梯度特征 ——
            gx = cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3)
            gy = cv2.Sobel(roi, cv2.CV_16S, 0, 1, ksize=3)
            grad = np.hypot(gx, gy)[(roi > 0) & (roi < 65535)].mean()

            # 返回决策
            print(f"{var:.3f} / {grad:.3f}")
            res.append(var > VAR_TH or grad > GRAD_TH)

        return res

    def get_var_and_grad(self):
        pass

    def corp_face_photo(self, frame: np.ndarray, w: int, h: int):
        faces = []
        tgt_ratio = w / h
        if len(self._boxes_cache) == 0:
            return faces
        for box_queue in self._boxes_cache:
            (x1, y1, x2, y2) = self._smooth_box(box_queue)
            # —— 1. 严格边界保护 ——
            img_h, img_w = frame.shape[:2]
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            # 中心裁出同比例最大区域
            fw, fh = x2 - x1, y2 - y1
            src_ratio = fw / fh

            if src_ratio > tgt_ratio:  # 原框更宽 -> 裁宽
                new_fw = int(fh * tgt_ratio)
                dw = fw - new_fw
                x1 += dw // 2
                x2 -= dw - dw // 2
            else:  # 原框更高 -> 裁高
                new_fh = int(fw / tgt_ratio)
                dh = fh - new_fh
                y1 += dh // 2
                y2 -= dh - dh // 2

            # 再次防越界（浮点取舍可能差 1px）
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            # 裁剪并缩放，不改颜色
            roi = frame[y1:y2, x1:x2]
            roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)
            faces.append(roi)

        return faces

    def draw_frame(self, frame: np.ndarray, type : str):
        """
        在传入frame单帧图片上画框
        :param frame: 传入图片
        :param type: 图片格式 bgr8, rgb8
        """
        if len(self._boxes_cache) == 0:
            return
        if type == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for box_queue in self._boxes_cache:
            assert isinstance(box_queue, queue.Queue)
            (x1, y1, x2, y2) = self._smooth_box(box_queue)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if type == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            assert isinstance(frame, np.ndarray)

    def face_detect(self, frame_bgr8: np.ndarray) -> bool:
        """
        输入:  BGR8  numpy 数组 (h, w, 3)
        返回:  list[(x1, y1, x2, y2)]   每张脸一个框，坐标为像素整数
        """
        # BGR -> RGB，并包装成 MediaPipe Image
        rgb = cv2.cvtColor(frame_bgr8, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image( image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        # 调用模型推理
        _result = self._detector.detect(mp_image)

        # 解析结果
        flag = False
        for idx, det in enumerate(_result.detections):
            bbox = det.bounding_box
            _x1 = int(bbox.origin_x)
            _y1 = int(bbox.origin_y)
            _x2 = int(bbox.origin_x + bbox.width)
            _y2 = int(bbox.origin_y + bbox.height)
            flag = True
            if len(self._boxes_cache) < idx + 1:
                _box_queue = queue.Queue()
                _box_queue.put((_x1, _y1, _x2, _y2))

                self._boxes_cache.append(_box_queue)
            else:
                assert isinstance(self._boxes_cache[idx], queue.Queue)
                if self._boxes_cache[idx].full():
                    self._boxes_cache[idx].get()
                self._boxes_cache[idx].put((_x1, _y1, _x2, _y2))
        return flag

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
        result : List[Tuple[int, int, int, int]] = detector.face_detect(frame)
        if result:  # 不为空
            print(len(result))
            for idx, (x1, y1, x2, y2) in enumerate(result, 1):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                print(f'Face {idx}:  top-left=({x1}, {y1})  bottom-right=({x2}, {y2})')
        else:
            print('当前画面未检测到人脸')

        cv2.imshow('Camera 640x480', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


