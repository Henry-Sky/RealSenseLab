import queue
import numpy as np
from pyrealsense2 import composite_frame

def frame_callback(_cache):
    """
    工厂函数：把队列包在闭包里，返回一个符合签名的 callback 函数
    :param _cache: 帧缓存队列
    :return: 函数
    """
    def _callback(_frame):
        """后台回调线程，把最新获取的视频帧塞进队列后返回"""
        # 拆解复合帧（pipeline启动后，每一帧都是复合帧，按配置镜头种类各塞一张video_frame）
        _frame = composite_frame(_frame)
        color_frame = _frame.get_color_frame()
        depth_frame = _frame.get_depth_frame()
        if not color_frame or not depth_frame: return
        # 获取裸数据
        color_frame = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())
        # 检查队列缓存是否有空
        if _cache.full():
            try:
                # 尝试扔掉旧帧（get_nowait 非阻塞方法）
                _cache.get_nowait()
            except queue.Empty:
                # 万一队列为空的情况，防止主线程在判full后取空队列
                pass
        # 添加帧缓存
        _cache.put({"color": color_frame, "depth": depth_frame})
    return _callback