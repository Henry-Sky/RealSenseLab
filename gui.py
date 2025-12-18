import queue
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QToolBar
from PySide6.QtGui import QPixmap
from utils.file import load_stylesheet
from utils.qutils import bgr8_to_qimage, rgb8_to_qimage, spawn_noise_background

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_queue = None
        self.depth_video_flag = False
        self.photo_flag = False
        self.frame = None

        """---------------------------Widget----------------------------"""
        self.video_label = QLabel("Video")
        self.toolbar = QToolBar()
        self.depth_button = QPushButton("深度伪彩")
        self.photo_label = QLabel("Photo")
        self.result_text = QLabel("无法识别")
        self.result_text.setObjectName("result_text")
        self.run_button = QPushButton("开始")
        self.run_button.setObjectName("run_button")

        """-----------------------Widget Setting------------------------"""
        video_img = spawn_noise_background(640, 480, "无视频信号")
        self.depth_button.setCheckable(True)
        self.depth_button.toggled.connect(self.toggle_depth)
        self.toolbar.addWidget(self.depth_button)
        photo_img = spawn_noise_background(200, 300, "未知人像")
        self.video_label.setPixmap(QPixmap.fromImage(video_img))
        self.photo_label.setPixmap(QPixmap.fromImage(photo_img))
        self.run_button.clicked.connect(self.toggle_running)

        """---------------------------Layout----------------------------"""
        # 创建布局
        self.layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        # 添加子布局到主布局中
        self.left_layout.addWidget(self.toolbar)
        self.left_layout.addWidget(self.video_label)
        self.right_layout.addWidget(self.photo_label)
        self.right_layout.addWidget(self.result_text)
        self.right_layout.addWidget(self.run_button)
        self.layout.addLayout(self.left_layout)
        self.layout.addLayout(self.right_layout)
        self.setLayout(self.layout)

        """---------------------------Timer------------------------------"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)  # 60fps: 1000ms // 60 = 16ms
        self._running = False

        """---------------------------Style------------------------------"""
        qss = load_stylesheet("resources/style.qss")
        self.setStyleSheet(qss)

    def bind_frame_queue(self, fqueue : queue.Queue):
        self.frame_queue = fqueue

    def toggle_running(self):
        if self._running:
            self.run_button.setText("开始")
            self._running = False
        else:
            self.run_button.setText("停止")
            self._running = True

    def toggle_depth(self, checked : bool):
        if checked:
            self.depth_video_flag = True
            print("depth button is checked")
        else:
            self.depth_video_flag = False
            print("depth button is unchecked")

    def update_frame(self):
        if self.frame_queue is None or not self._running:
            video_img = spawn_noise_background(640, 480, "无视频信号")
            photo_img = spawn_noise_background(200, 300, "未知人像")
            self.video_label.setPixmap(QPixmap.fromImage(video_img))
            self.photo_label.setPixmap(QPixmap.fromImage(photo_img))
        else:
            if self.frame_queue.empty():
                return
            else:
                self.frame = self.frame_queue.get()
            # 视频信号处理
            if self.depth_video_flag:
                rgb = rgb8_to_qimage(self.frame["pseudo"])
                self.video_label.setPixmap(QPixmap.fromImage(rgb))
            else:
                bgr = bgr8_to_qimage(self.frame["color"])
                self.video_label.setPixmap(QPixmap.fromImage(bgr))
            # 识别人像处理
            if self.photo_flag:
                pass
            else:
                photo_img = spawn_noise_background(200, 300, "未知人像")
                self.photo_label.setPixmap(QPixmap.fromImage(photo_img))



    def closeEvent(self, event, /):
        self.timer.stop()
        self._running = False
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    frame_cache = queue.Queue(maxsize=3)  # 视频缓存
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())