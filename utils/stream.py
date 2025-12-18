import queue
import numpy as np
import logging, time
from pyrealsense2 import composite_frame, colorizer, option

# 日志配置
logging.basicConfig(
    filename="realsense.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def frame_callback(_cache):
    """
    工厂函数：把队列包在闭包里，返回一个符合签名的 callback 函数
    :param _cache: 帧缓存队列
    :return: 函数
    """
    _colorizer = colorizer()
    _colorizer.set_option(option.color_scheme, 0)  # Jet
    def _callback(_frame):
        """后台回调线程，把最新获取的视频帧塞进队列后返回"""
        # 拆解复合帧（pipeline启动后，每一帧都是复合帧，按配置镜头种类各塞一张video_frame）
        _frame = composite_frame(_frame)
        depth_f = _frame.get_depth_frame()
        color_f = _frame.get_color_frame()
        if not depth_f or not color_f: return
        # 生成伪彩
        pseudo_f = _colorizer.colorize(depth_f)  # 实例调用
        # 获取裸数据
        depth_z16 = np.asanyarray(depth_f.get_data())
        pseudo_rgb8 = np.asanyarray(pseudo_f.get_data())
        color_bgr8 = np.asanyarray(color_f.get_data())
        # 检查队列缓存是否有空
        if _cache.full():
            try:
                # 尝试扔掉旧帧（get_nowait 非阻塞方法）
                _cache.get_nowait()
            except queue.Empty:
                # 万一队列为空的情况，防止主线程在判full后取空队列
                pass
        # 添加帧缓存
        _cache.put({"color": color_bgr8, "depth": depth_z16, "pseudo": pseudo_rgb8})
    return _callback

if __name__ == "__main__":
    import cv2
    from pyrealsense2 import pipeline, config, stream, format

    # 配置视频流
    pipe = pipeline()  # 视频管线
    cfg = config()  # 配置文件
    cfg.enable_stream(stream.color, 640, 480, format.bgr8, 30)  # 设置BGR-8位彩色视频
    cfg.enable_stream(stream.depth, 640, 480, format.z16, 30)  # 设置深度视频采集
    frame_cache = queue.Queue(maxsize=3)  # 视频缓存
    # 启动 -> 返回一个profile（含实际打开的视频流参数）
    frame_cb = frame_callback(frame_cache)
    try:
        pipe.start(cfg, callback=frame_callback(frame_cache))
        logging.info("管道启动成功")
    except Exception as e:
        logging.error("启动失败: %s", e)
        raise

    try:
        while True:
            if not frame_cache.empty():
                frame = frame_cache.get()
                cv2.imshow("pseudo", frame["pseudo"])
            # 按 ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()