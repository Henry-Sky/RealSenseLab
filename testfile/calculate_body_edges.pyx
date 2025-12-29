# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
import cv2
from libc.math cimport sqrt

cnp.import_array()

ctypedef struct rs2_intrinsics_t:
    int width, height
    float ppx, ppy
    float fx, fy
    int model
    float coeffs[5]

cdef inline void deproject(float u, float v, float z,
                           rs2_intrinsics_t *intr,
                           float *x) noexcept nogil:
    x[0] = (u - intr.ppx) / intr.fx * z
    x[1] = (v - intr.ppy) / intr.fy * z
    x[2] = z

cdef inline void project(float *x, rs2_intrinsics_t *intr,
                         float *u, float *v) noexcept nogil:
    if x[2] == 0:
        u[0] = -1
        v[0] = -1
        return
    u[0] = x[0] / x[2] * intr.fx + intr.ppx
    v[0] = x[1] / x[2] * intr.fy + intr.ppy

def calculate_body_edges(object intr,
                         cnp.ndarray[cnp.uint8_t, ndim=3] frame_bgr8,
                         cnp.ndarray[cnp.uint16_t, ndim=2] frame_z16,
                         tuple bbox):
    """
    与 Python 版完全一致的接口，返回 12×2×2 的 int32 边线数组
    """
    cdef:
        int img_h = frame_bgr8.shape[0]
        int img_w = frame_bgr8.shape[1]
        rs2_intrinsics_t intr_c
        float seed_z
        int cx, cy, x1, y1, x2, y2
        cnp.ndarray[cnp.float32_t, ndim=2] depth32
        cnp.ndarray[cnp.uint8_t, ndim=2] mask
        int npts, i, j
        float[:, :] pts_cam
        float mean[3]
        float cov[3][3]
        float local_min[3]
        float local_max[3]
        float corners_local[8][3]
        float corners_cam[8][3]
        float pts2d[8][2]
        cnp.ndarray[cnp.int32_t, ndim=3] lines

    # 初始化 mean
    mean[0] = 0.0
    mean[1] = 0.0
    mean[2] = 0.0

    # 拷贝 intr
    intr_c.width  = intr.width
    intr_c.height = intr.height
    intr_c.ppx    = intr.ppx
    intr_c.ppy    = intr.ppy
    intr_c.fx     = intr.fx
    intr_c.fy     = intr.fy
    intr_c.model  = 0
    for i in range(5):
        intr_c.coeffs[i] = intr.coeffs[i] if len(intr.coeffs) > i else 0.0

    # 深度预处理
    depth32 = (frame_z16 / 1000.0).astype(np.float32)
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    seed_z = depth32[cy, cx]
    if seed_z == 0:
        return None

    # 泛洪
    mask = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
    cv2.floodFill(depth32, mask, (cx, cy), 0,
                  loDiff=50, upDiff=50,
                  flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    mask = mask[1:-1, 1:-1]
    if np.count_nonzero(mask) < 500:
        return None

    # 点云
    ys, xs = np.where(mask)
    zs = frame_z16[ys, xs] / 1000.0
    npts = xs.shape[0]
    pts_cam = np.empty((npts, 3), dtype=np.float32)
    for i in range(npts):
        deproject(float(xs[i]), float(ys[i]), float(zs[i]), &intr_c, &pts_cam[i, 0])

    # PCA
    for i in range(npts):
        mean[0] += pts_cam[i, 0]
        mean[1] += pts_cam[i, 1]
        mean[2] += pts_cam[i, 2]
    mean[0] /= npts
    mean[1] /= npts
    mean[2] /= npts

    for i in range(3):
        for j in range(3):
            cov[i][j] = 0.0

    for i in range(npts):
        cov[0][0] += (pts_cam[i, 0] - mean[0]) * (pts_cam[i, 0] - mean[0])
        cov[0][1] += (pts_cam[i, 0] - mean[0]) * (pts_cam[i, 1] - mean[1])
        cov[0][2] += (pts_cam[i, 0] - mean[0]) * (pts_cam[i, 2] - mean[2])
        cov[1][0] += (pts_cam[i, 1] - mean[1]) * (pts_cam[i, 0] - mean[0])
        cov[1][1] += (pts_cam[i, 1] - mean[1]) * (pts_cam[i, 1] - mean[1])
        cov[1][2] += (pts_cam[i, 1] - mean[1]) * (pts_cam[i, 2] - mean[2])
        cov[2][0] += (pts_cam[i, 2] - mean[2]) * (pts_cam[i, 0] - mean[0])
        cov[2][1] += (pts_cam[i, 2] - mean[2]) * (pts_cam[i, 1] - mean[1])
        cov[2][2] += (pts_cam[i, 2] - mean[2]) * (pts_cam[i, 2] - mean[2])

    for i in range(3):
        for j in range(3):
            cov[i][j] /= npts

    cov_np = np.asarray(cov, dtype=np.float32)
    eig_val, eig_vec = np.linalg.eigh(cov_np)
    eig_vec = eig_vec.T

    # 局部坐标
    mean_arr = np.array([mean[0], mean[1], mean[2]], dtype=np.float32)
    pts_local = np.dot(pts_cam - mean_arr, eig_vec)

    for i in range(3):
        local_min[i] = pts_local[:, i].min()
        local_max[i] = pts_local[:, i].max()

    dx = local_max[0] - local_min[0]
    dy = local_max[1] - local_min[1]
    dz = local_max[2] - local_min[2]

    # 8 个角点
    for i in range(8):
        corners_local[i][0] = (i & 1) * dx + local_min[0]
        corners_local[i][1] = ((i >> 1) & 1) * dy + local_min[1]
        corners_local[i][2] = ((i >> 2) & 1) * dz + local_min[2]

    # 转回相机坐标
    for i in range(8):
        for j in range(3):
            corners_cam[i][j] = mean[j]
            for k in range(3):
                corners_cam[i][j] += corners_local[i][k] * eig_vec[k, j]

    # 投影
    for i in range(8):
        project(&corners_cam[i][0], &intr_c, &pts2d[i][0], &pts2d[i][1])

    # 组装边线
    lines = np.empty((12, 2, 2), dtype=np.int32)
    cdef int edges[12][2]
    edges[0][0] = 0;  edges[0][1] = 1
    edges[1][0] = 1;  edges[1][1] = 2
    edges[2][0] = 2;  edges[2][1] = 3
    edges[3][0] = 3;  edges[3][1] = 0
    edges[4][0] = 4;  edges[4][1] = 5
    edges[5][0] = 5;  edges[5][1] = 6
    edges[6][0] = 6;  edges[6][1] = 7
    edges[7][0] = 7;  edges[7][1] = 4
    edges[8][0] = 0;  edges[8][1] = 4
    edges[9][0] = 1;  edges[9][1] = 5
    edges[10][0] = 2; edges[10][1] = 6
    edges[11][0] = 3; edges[11][1] = 7

    for i in range(12):
        for j in range(2):
            lines[i, j, 0] = <int>pts2d[edges[i][j]][0]
            lines[i, j, 1] = <int>pts2d[edges[i][j]][1]
    return lines