import time
import threading
import numpy as np
from typing import List
from collections import deque
from dataclasses import dataclass

@dataclass(slots=True)
class FramePair:
    frame_bgr8 : np.ndarray
    frame_z16 : np.ndarray
    frame_ir1_y8 : np.ndarray
    frame_ir2_y8 : np.ndarray
    # 各帧数据的时间戳
    ts_color: float
    ts_depth: float
    ts_ir1: float
    ts_ir2: float

class FrameBuffer:
    """
    定长帧缓冲队列，线程安全，队满自动丢弃最旧帧，仅限 FramePair 数据
    """
    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError(f"帧缓冲队列不能为: {maxsize} , 需满足 maxsize >= 0")
        self._buf: deque[FramePair] = deque(maxlen=maxsize)  # 天然定长，满时自动 popleft
        self._lock = threading.RLock()  # 支持同线程重入

    # ----------- 核心操作 -----------
    def put(self, item: FramePair) -> None:
        """线程安全：放入一帧（满时自动丢弃最旧帧）"""
        with self._lock:
            self._buf.append(item)

    def get(self) -> FramePair:
        """线程安全：取出最旧帧，空时阻塞直到有数据"""
        with self._lock:
            while not self._buf:
                # 用条件变量可以做成真正阻塞，这里简化成抛异常
                raise IndexError("缓存队列为空，无法取出")
            return self._buf.popleft()

    def peek(self, idx: int = 0) -> FramePair:
        """线程安全：只看，不删除；支持负索引"""
        with self._lock:
            return self._buf[idx]  # deque 原生支持负索引

    def set_item(self, idx: int, new: FramePair) -> None:
        """线程安全：原地替换某帧"""
        with self._lock:
            self._buf[idx] = new

    # ----------- 辅助 -----------
    def peek_all(self) -> List[FramePair]:
        """线程安全：返回全部帧的浅拷贝"""
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __repr__(self) -> str:
        with self._lock:
            return f"FrameBuffer({list(self._buf)})"


if __name__ == "__main__":

    fb = FrameBuffer(maxsize=6)

    def get_frame_pair(i) -> FramePair:
        return FramePair(
            frame_bgr8=np.full((1280,720,3), i, dtype=np.uint8),
            frame_z16=np.full((1280,720,1), i, dtype=np.uint16),
            frame_ir1_y8=np.full((1280,720,1), i, dtype=np.uint8),
            frame_ir2_y8=np.full((1280,720,1), i, dtype=np.uint8),
            ts_color=time.time(),
            ts_depth=time.time(),
            ts_ir1=time.time(),
            ts_ir2=time.time(),
        )

    def producer():
        for i in range(999999):
            frame = get_frame_pair(i)
            fb.put(frame)
            print(f"put {i}")
            time.sleep(0.1)

    def monitor():
        while True:
            time.sleep(0.2)
            if len(fb):
                frm = fb.peek(-1)
                print(f"peek idx=-1: {frm.frame_bgr8[0, 0, 0]}")


    threading.Thread(target=producer, daemon=True).start()
    threading.Thread(target=monitor, daemon=True).start()
    time.sleep(10)