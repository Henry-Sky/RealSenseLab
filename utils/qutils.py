import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QFont, QColor

def rgb8_to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    return QImage(rgb.tobytes(), w, h, rgb.strides[0], QImage.Format.Format_RGB888)

def bgr8_to_qimage(bgr: np.ndarray) -> QImage:
    """BGR 8-bit (H,W,3) → QImage RGB888"""
    h, w, _ = bgr.shape
    rgb = bgr[..., ::-1]
    return QImage(rgb.tobytes(), w, h, rgb.strides[0], QImage.Format.Format_RGB888)

def trans_pseudo_color(_depth_frame : np.ndarray) -> np.ndarray:
    depth_cm = cv2.convertScaleAbs(_depth_frame, alpha=255 / 3000.0)
    colorized = cv2.applyColorMap(depth_cm, cv2.COLORMAP_JET)
    return colorized

def spawn_noise_background(w=640, h=480, text="无信号") -> QImage:
    """
    获取固定尺寸噪声背景(用于UI占位)
    :param w: 图像宽度
    :param h: 图像高度
    :param text: 显示提示文字
    :return: 一张QImage噪声图片
    """
    # 生成随机雪花噪声
    noise = np.random.randint(low=0, high=255, size=(h, w, 3), dtype=np.uint8)
    img = QImage(noise.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)

    # 噪声背景上添加提示文字
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Microsoft YaHei", h // 18, QFont.Weight.Bold)  # 字体：白色、加粗、字号随高度适应
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255, 230))
    painter.drawText(img.rect(), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return img