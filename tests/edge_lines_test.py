import timeit
from typing import Tuple

import cv2
import numpy as np
from tests.testkits import generate_layered_depth
from calculate_body_edges import calculate_body_edges  # 导入cpython编译后的算法
from pyrealsense2 import rs2_deproject_pixel_to_point, rs2_project_point_to_pixel, intrinsics, distortion

def test_device_intr(img: np.ndarray):
    """
    根据输入的 np 图片 (H,W,3) 或 (H,W) 生成对应尺寸的 rs.intrinsics 实例
    焦距按常见 RGB 相机经验值：fx=fy=max(w,h)*0.9
    主点设为图像中心
    """
    h, w = img.shape[:2]
    intr = intrinsics()
    intr.width  = w
    intr.height = h
    f = max(w, h) * 0.9
    intr.fx = intr.fy = f
    intr.ppx = (w - 1) * 0.5
    intr.ppy = (h - 1) * 0.5
    intr.model = distortion.none
    intr.coeffs = [0.0] * 5
    return intr

# 原始算法
def _calculate_body_edges(intr : intrinsics, frame_bgr8: np.ndarray, frame_z16: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> None | np.ndarray:
    """计算人体3d边线：性能开销巨大！"""
    img_h, img_w = frame_bgr8.shape[:2]
    depth32 = frame_z16.astype(np.float32)
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) // 2)
    cy = int((y1 + y2) // 2)
    seed_z = depth32[cy, cx]
    if seed_z == 0:
        return None
    # floodFill 泛洪得 mask
    mask = np.zeros((img_h + 2, img_w + 2), np.uint8)
    cv2.floodFill(depth32, mask, (cx, cy), 0,
                  loDiff=50, upDiff=50,
                  flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    mask = mask[1:-1, 1:-1]
    if np.count_nonzero(mask) < 500:
        return None
    # 3-D 点云
    ys, xs = np.where(mask)
    zs = frame_z16[ys, xs] / 1000.0
    pts_cam = np.array([rs2_deproject_pixel_to_point(intr, [u, v], z)
                        for u, v, z in zip(xs, ys, zs)], dtype=np.float32)
    # PCA 求主方向 + AABB包围盒8点坐标
    mean, eig_vec = cv2.PCACompute(pts_cam, mean=None)
    eig_vec = eig_vec[:3]
    pts_local = (pts_cam - mean) @ eig_vec.T
    local_min = pts_local.min(axis=0)
    local_max = pts_local.max(axis=0)
    dx, dy, dz = local_max - local_min
    corners_local = np.array([
        [0, 0, 0], [dx, 0, 0], [dx, dy, 0], [0, dy, 0],
        [0, 0, dz], [dx, 0, dz], [dx, dy, dz], [0, dy, dz]
    ]) + local_min
    corners_cam = (corners_local @ eig_vec) + mean
    # 投影回 2-D 并画框
    pts2d = np.array([rs2_project_point_to_pixel(intr, p)
                      for p in corners_cam]).astype(int)
    edges_idx = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
    lines = np.asarray([[pts2d[i], pts2d[j]] for i, j in edges_idx], dtype=np.int32)
    return lines


if __name__ == "__main__":
    img = cv2.imread('tests/test.jpg')
    depth = generate_layered_depth(img)
    intr = test_device_intr(img)
    bbox = (745, 342, 847, 480)
    lines = calculate_body_edges(intr, img, depth, bbox)
    nums = [50, 100, 150, 200]
    dict = {}
    for n in nums:
        t = timeit.timeit(lambda: calculate_body_edges(intr, img, depth, bbox), number=n)
        t_ms = t / n * 1000
        dict[n] = round(t_ms, 3)
    print(dict)

    # {50: 555.331, 100: 563.534, 150: 554.436, 200: 562.741}
    # {50: 155.193, 100: 155.461, 150: 156.907, 200: 155.659}