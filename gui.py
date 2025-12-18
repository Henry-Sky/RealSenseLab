import queue
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPixmap
from utils.file import load_stylesheet
from utils.qutils import bgr8_to_qimage, spawn_noise_background

class MainWindow(QWidget):
    def __init__(self, frame_cache : queue.Queue):
        super().__init__()
        self.frame_queue = frame_cache

        """---------------------------Widget----------------------------"""
        self.video_label = QLabel("Video")
        self.photo_label = QLabel("Photo")
        self.result_text = QLabel("无法识别")
        self.result_text.setObjectName("result_text")
        self.run_button = QPushButton("开始")
        self.run_button.setObjectName("run_button")

        """-----------------------Widget Setting------------------------"""
        video_img = spawn_noise_background(640, 480, "无视频信号")
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

    def toggle_running(self):
        if self._running:
            self.run_button.setText("开始")
            self._running = False
        else:
            self.run_button.setText("停止")
            self._running = True

    def update_frame(self):
        if not self._running:
            video_img = spawn_noise_background(640, 480, "无视频信号")
            photo_img = spawn_noise_background(200, 300, "未知人像")
            self.video_label.setPixmap(QPixmap.fromImage(video_img))
            self.photo_label.setPixmap(QPixmap.fromImage(photo_img))
        else:
            if self.frame_queue.empty():
                pass
            else:
                frame = self.frame_queue.get()
                bgr = bgr8_to_qimage(frame["color"])
                self.video_label.setPixmap(QPixmap.fromImage(bgr))


    def closeEvent(self, event, /):
        self.timer.stop()
        self._running = False
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    frame_cache = queue.Queue(maxsize=3)  # 视频缓存
    app = QApplication([])
    window = MainWindow(frame_cache)
    window.show()
    sys.exit(app.exec())