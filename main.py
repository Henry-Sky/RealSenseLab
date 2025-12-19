import sys
import queue
import pyrealsense2 as rs
from gui import MainWindow
from utils.stream import frame_callback
from PySide6.QtWidgets import QApplication

def main():
    app = QApplication([])

    # 配置视频流
    pipe = rs.pipeline()  # 视频管线
    cfg = rs.config()  # 配置文件
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 设置BGR-8位彩色视频
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 设置深度视频采集
    frame_cache = queue.Queue(maxsize=3)  # 视频缓存
    # 启动 -> 返回一个profile（含实际打开的视频流参数）
    profile = pipe.start(cfg, callback=frame_callback(frame_cache))

    window = MainWindow()
    window.bind_frame_queue(frame_cache)
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()