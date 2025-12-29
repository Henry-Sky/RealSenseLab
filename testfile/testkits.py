import cv2
import numpy as np

def generate_layered_depth(frame_bgr):
    """基于颜色分层生成 16-bit 假深度，带调试信息"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # 初始最远
    depth = np.full(frame_bgr.shape[:2], 65535, dtype=np.float32)

    # ---------- 蓝色 ----------
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    depth[blue_mask > 0] = 10000
    # ---------- 绿色 ----------
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    depth[green_mask > 0] = 30000
    # ---------- 红色（HSV 两段） ----------
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2)
    depth[red_mask > 0] = 50000
    # 调试：各 mask 里像素命中
    print('blue pixels:', cv2.countNonZero(blue_mask))
    print('green pixels:', cv2.countNonZero(green_mask))
    print('red pixels:', cv2.countNonZero(red_mask))
    # 平滑
    depth = cv2.GaussianBlur(depth, (5, 5), 0)
    return depth.astype(np.uint16)

if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    depth = generate_layered_depth(img)
    print(depth.shape)