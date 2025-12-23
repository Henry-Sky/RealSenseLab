import os
import sys
import pathlib
import queue
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QToolBar
from PySide6.QtGui import QPixmap, QIcon
from utils.file import load_stylesheet
from utils.qutils import bgr8_to_qimage, rgb8_to_qimage, spawn_noise_background
from detector import Detector

class MainWindow(QWidget):
    def __init__(self, profile = None):
        super().__init__()
        self.setWindowTitle("RealSenseLab 小组实验课作业：黄瑞、周敬浩")
        # 开发时取项目根目录，打包后取 PyInstaller 临时目录
        BASE_DIR = pathlib.Path(getattr(sys, '_MEIPASS', pathlib.Path(__file__).resolve().parent.parent))
        icon_path = os.path.join(BASE_DIR, "imgs/21.简标朱稿.png")
        self.setWindowIcon(QIcon(icon_path))
        if profile is not None:
            self.profile = profile

        self.frame_queue = None
        self.frame = None
        self.detector = Detector()

        """---------------------------Widget----------------------------"""
        self.toolbar = QToolBar()
        self.depth_button = QPushButton("深度伪彩")
        self.face_button = QPushButton("人脸识别")
        self.body_button = QPushButton("身体框图")
        self.video_label = QLabel("Video")
        self.photo_label = QLabel("Photo")
        self.result_text = QLabel("无法识别")
        self.result_text.setObjectName("result_text")
        self.run_button = QPushButton("开始")
        self.run_button.setObjectName("run_button")

        """-----------------------Widget Setting------------------------"""
        video_img = spawn_noise_background(640, 480, "无视频信号")
        self.depth_button.setCheckable(True)
        self.depth_button.toggled.connect(self.toggle_depth)
        self.face_button.setCheckable(True)
        self.face_button.toggled.connect(self.toggle_face)
        self.body_button.setCheckable(True)
        self.toolbar.addWidget(self.depth_button)
        self.toolbar.addWidget(self.face_button)
        self.toolbar.addWidget(self.body_button)

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

    def toggle_face(self, checked : bool):
        if checked:
            self.face_flag = True
        else:
            self.face_flag = False

    def toggle_depth(self, checked : bool):
        if checked:
            self.depth_video_flag = True
        else:
            self.depth_video_flag = False

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
            # 识别人像处理
            if self.face_button.isChecked():
                photo_img = spawn_noise_background(200, 300, "未知人像")
                if self.detector.face_detect(self.frame["color"]):
                    self.detector.draw_face_rect(self.frame["color"], "bgr8")
                    face_judge = self.detector.judge_person_real_or_photo(self.frame["depth"], profile=self.profile)
                    face_crop = self.detector.corp_face_photo(self.frame["color"], 200, 300)
                    self.photo_label.setPixmap(QPixmap.fromImage(bgr8_to_qimage(face_crop[0])))
                    if face_judge[0]:
                        self.result_text.setText("识别真人")
                        if self.body_button.isChecked():
                            self.detector.draw_body_rect(self.frame["color"], self.frame["depth"], profile=self.profile)
                    else:
                        self.result_text.setText("识别照片")
                else:
                    self.result_text.setText("无法识别")
                    self.photo_label.setPixmap(QPixmap.fromImage(photo_img))
            else:
                self.result_text.setText("无法识别")
                photo_img = spawn_noise_background(200, 300, "未知人像")
                self.photo_label.setPixmap(QPixmap.fromImage(photo_img))
            # 视频信号处理
            if self.depth_button.isChecked():
                rgb = rgb8_to_qimage(self.frame["pseudo"])
                self.video_label.setPixmap(QPixmap.fromImage(rgb))
            else:
                bgr = bgr8_to_qimage(self.frame["color"])
                self.video_label.setPixmap(QPixmap.fromImage(bgr))



    def closeEvent(self, event):
        self.timer.stop()
        self._running = False
        super().closeEvent(event)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())