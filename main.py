from PySide6.QtWidgets import QApplication
from labwindow import LabWindow


def main():
    app = QApplication([])
    window = LabWindow()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()