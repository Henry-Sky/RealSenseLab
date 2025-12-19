import queue
import numpy as np
import logging, time
from pyrealsense2 import composite_frame, colorizer, option, stream

# 日志配置
logging.basicConfig(
    filename="realsense.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def frame_z16_to_points(depth_frame, depth_scale=None, intrin=None, profile = None):
    """
    把 depth_frame_z16
    返回: points_3d  shape=(H*W, 3), dtype=float64
    """
    if profile is not None:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrin = profile.get_stream(stream.depth).as_video_stream_profile().get_intrinsics()
    else:
        if depth_scale is None or intrin is None:
            return None
    H, W = depth_frame.shape[:2]
    z = depth_frame * depth_scale                        # 米
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
    try:
        profile = pipe.start(cfg, callback=frame_callback(frame_cache))
        # depth_scale: 0.0010000000474974513, intrin: [ 640x480  p[327.064 240.408]  f[393.544 393.544]  Brown Conrady [0 0 0 0 0] ]
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intrin = profile.get_stream(stream.depth).as_video_stream_profile().get_intrinsics()
        print(f"depth_scale: {depth_scale}, intrin: {intrin}")
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