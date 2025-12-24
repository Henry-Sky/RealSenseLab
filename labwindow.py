from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, QToolBar
from framebuffer import FramePair
from labdetector import LabDetector
from labstream import LabStream
from utils.qutils import spawn_noise_background, bgr8_to_qimage, trans_pseudo_color
from utils.file import load_file_content

DETECTION_DELAY_STAMP_FRAMES = 3  # 每 3 帧调用一次检测器线程
WINDOWS_UPDATE_FPS = 60  # 窗口刷新 FPS

class LabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSenseLab Demo")
        self.stream = LabStream()
        self.detector = LabDetector()
        self.frame_buffer = None
        self.frame_cnt = 0  # 帧计数器
        self._running = False

        # --------------------------创建组件&命名组件--------------------------
        self._view_label = QLabel(self)
        self._view_label.setObjectName("view_label")
        self._photo_label = QLabel(self)
        self._photo_label.setObjectName("photo_label")
        self._result_label = QLabel(self)
        self._result_label.setObjectName("result_label")
        self._run_button = QPushButton(self)
        self._run_button.setObjectName("run_button")
        self._tool_bar = QToolBar(self)
        self._tool_bar.setObjectName("tool_bar")
        self._depth_button = QPushButton("深度伪彩")
        self._face_button = QPushButton("人脸识别")
        self._body_button = QPushButton("身体框图")

        # ------------------------------配置组件------------------------------
        view_img = spawn_noise_background(1280, 720, "无视频信号")
        self._view_label.setPixmap(QPixmap.fromImage(view_img))
        photo_img = spawn_noise_background(200, 300, "未知人像")
        self._photo_label.setPixmap(QPixmap.fromImage(photo_img))
        self._result_label.setText("无法识别")
        self._run_button.setText("开始")
        self._run_button.clicked.connect(self.toggle_running)
        self._depth_button.setCheckable(True)
        self._face_button.setCheckable(True)
        self._body_button.setCheckable(True)
        self._tool_bar.addWidget(self._depth_button)
        self._tool_bar.addWidget(self._face_button)
        self._tool_bar.addWidget(self._body_button)

        # ------------------------------设置布局------------------------------
        main_hbox = QHBoxLayout()
        left_vbox = QVBoxLayout()
        right_vbox = QVBoxLayout()
        left_vbox.addWidget(self._view_label)
        right_vbox.addWidget(self._photo_label)
        right_vbox.addWidget(self._result_label)
        right_vbox.addWidget(self._run_button)
        main_hbox.addLayout(left_vbox)
        main_hbox.addLayout(right_vbox)

        # ------------------------------窗口核心------------------------------
        central_widget = QWidget(self)
        central_widget.setLayout(main_hbox)
        self.setCentralWidget(central_widget)
        self.setStyleSheet(load_file_content("resources/styles/style.qss"))
        self.addToolBar(self._tool_bar)

        # ------------------------------启动定时------------------------------
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.windows_update)
        self.timer.start(1000 // WINDOWS_UPDATE_FPS)  # Default 60 fps

    def toggle_running(self):
        if self._running:
            self.stream.stop_stream()
            self._run_button.setText("开始")
            self._running = False
        else:
            self.frame_buffer = self.stream.start_stream()
            self._run_button.setText("停止")
            self._running = True

    def windows_update(self):
        if not self._running or not self.frame_buffer or len(self.frame_buffer) <= 0:
            view_img = spawn_noise_background(1280, 720, "无视频信号")
            self._view_label.setPixmap(QPixmap.fromImage(view_img))
            photo_img = spawn_noise_background(200, 300, "未知人像")
            self._photo_label.setPixmap(QPixmap.fromImage(photo_img))

        if self._running and self.frame_buffer is not None and len(self.frame_buffer) > 0:
            current_frames : FramePair = self.frame_buffer.peek(-1)
            view_bgr8 = current_frames.frame_bgr8
            if self._depth_button.isChecked():
                view_bgr8 = trans_pseudo_color(current_frames.frame_z16)
            if self._face_button.isChecked():
                if self.frame_cnt % DETECTION_DELAY_STAMP_FRAMES == 0:
                    self.detector.detect_once(current_frames.frame_bgr8)
                if self.frame_cnt > DETECTION_DELAY_STAMP_FRAMES:
                    self.detector.draw_face_rectangle(view_bgr8)
            view_img = bgr8_to_qimage(view_bgr8)
            self._view_label.setPixmap(QPixmap.fromImage(view_img))
        self.frame_cnt += 1


    def closeEvent(self, event: QCloseEvent):
        super().closeEvent(event)
        pass

if __name__ == "__main__":
    app = QApplication([])
    window = LabWindow()
    window.show()
    app.exec()