from PySide6.QtGui import QCloseEvent, Qt, QPixmap
from PySide6.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
from utils.qutils import spawn_noise_background
from utils.file import load_file_content

class LabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSenseLab Demo")

        # --------------------------创建组件&命名组件--------------------------
        self._view_label = QLabel(self)
        self._view_label.setObjectName("view_label")
        self._photo_label = QLabel(self)
        self._photo_label.setObjectName("photo_label")
        self._result_label = QLabel(self)
        self._result_label.setObjectName("result_label")
        self._run_button = QPushButton(self)
        self._run_button.setObjectName("run_button")

        # ------------------------------配置组件------------------------------
        view_img = spawn_noise_background(1280, 720, "无视频信号")
        self._view_label.setPixmap(QPixmap.fromImage(view_img))
        photo_img = spawn_noise_background(200, 300, "未知人像")
        self._photo_label.setPixmap(QPixmap.fromImage(photo_img))
        self._result_label.setText("无法识别")
        self._run_button.setText("开始")

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

    def linkStream(self):
        pass

    def closeEvent(self, event: QCloseEvent):
        super().closeEvent(event)
        pass

if __name__ == "__main__":
    app = QApplication([])
    window = LabWindow()
    window.show()
    app.exec()