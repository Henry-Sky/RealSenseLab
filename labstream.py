import time
import numpy as np
import pyrealsense2 as rs
from framebuffer import FrameBuffer, FramePair

class LabStream:
    def __init__(self):
        self._profile = None
        self._frame_buffer = FrameBuffer(maxsize=6)  # 长度为 6 的定长缓冲队列
        self._pipeline = rs.pipeline()
        self._colorizer = rs.colorizer()
        self._align = rs.align(rs.stream.depth)
        # 配置相机捕获帧类型
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self._config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        self._config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    def _call_back(self, _frames):
        _frames = rs.composite_frame(_frames)
        _frames_aligned : rs.composite_frame = self._align.process(_frames)
        _color_frame = _frames_aligned.get_color_frame()
        _depth_frame = _frames_aligned.get_depth_frame()
        _infrared_1_frame = _frames_aligned.get_infrared_frame(1)
        _infrared_2_frame = _frames_aligned.get_infrared_frame(2)
        self._frame_buffer.put(
            FramePair(
                frame_bgr8=np.asarray(_color_frame.get_data()),
                frame_z16=np.asarray(_depth_frame.get_data()),
                frame_ir1_y8=np.asarray(_infrared_1_frame.get_data()),
                frame_ir2_y8=np.asarray(_infrared_2_frame.get_data()),
                ts_color=_color_frame.get_timestamp(),
                ts_depth=_depth_frame.get_timestamp(),
                ts_ir1=_infrared_1_frame.get_timestamp(),
                ts_ir2=_infrared_2_frame.get_timestamp(),
            )
        )

    def start_stream(self):
        self._profile = self._pipeline.start(self._config, self._call_back)
        if not self._profile:
            raise RuntimeError("启动 Stream 失败")
        return self._frame_buffer

    def stop_stream(self):
        self._pipeline.stop()


if __name__ == '__main__':
    ls = LabStream()
    frame_buffer = ls.start_stream()
    start = time.time()
    while True:
        if len(frame_buffer) > 0:
            frame_pair = frame_buffer.peek(-1)
            print(frame_pair.ts_color)
        else:
            print("no buffer")
        if time.time() - start > 10:
            break
        time.sleep(0.5)
    ls.stop_stream()
