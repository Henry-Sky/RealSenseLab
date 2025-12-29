import numpy as np
# 为旧版numpy添加向后兼容
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import timeit
from insightface.data import get_image as ins_get_image
from insightface.model_zoo import get_model

MODEL_ONNX = r"models\buffalo_s\det_500m.onnx"

# 动态 shape 与实验一致
MIN_SHAPE = (1, 3, 160, 160)
OPT_SHAPE = (4, 3, 640, 640)
MAX_SHAPE = (8, 3, 640, 640)

providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': './trt_cache',
        'trt_max_workspace_size': 2 << 30,
        # 去掉括号，去掉 input:
        'trt_profile_min_shapes': '1,3,160,160',
        'trt_profile_opt_shapes': '4,3,640,640',
        'trt_profile_max_shapes': '8,3,640,640',
    }),
    'CPUExecutionProvider'
]

sizes = [(640,640), (480,480), (256, 256), (160, 160)]
dict = {}
NUM = 50

det = get_model(MODEL_ONNX, providers=providers)

img = ins_get_image('t1')
det.prepare(ctx_id=0, input_size=(640,640))
faces = det.detect(img)

for size in sizes:
    # 初始化模型
    det.prepare(ctx_id=0, input_size=size)

    # 正确方式：使用 lambda 或独立函数
    # 计时人脸检测过程，重复NUM次
    t = timeit.timeit(lambda: det.detect(img), number=NUM)
    t_ms = t / NUM * 1000
    dict[size] = round(t_ms, 3)


print(faces[0][0][:4])
# CPU
# L: {(640, 640): 26.707, (480, 480): 29.846, (256, 256): 24.101, (160, 160): 24.985}
# M: {(640, 640): 10.212, (480, 480): 10.481, (256, 256): 10.676, (160, 160): 10.267}
# S: {(640, 640): 4.185, (480, 480): 6.087, (256, 256): 4.187, (160, 160): 4.011}
# GPU
# L: {(640, 640): 9.282, (480, 480): 5.599, (256, 256): 5.577, (160, 160): 5.82}
# M: {(640, 640): 4.779, (480, 480): 4.732, (256, 256): 4.735, (160, 160): 4.702}
# S: {(640, 640): 4.006, (480, 480): 4.049, (256, 256): 4.001, (160, 160): 3.996}
# TensorRT
# L: {(640, 640): 3.612, (480, 480): 3.619, (256, 256): 3.593, (160, 160): 3.603}
# M: {(640, 640): 3.306, (480, 480): 3.313, (256, 256): 3.29, (160, 160): 3.299}
# S: {(640, 640): 5.518, (480, 480): 4.775, (256, 256): 4.175, (160, 160): 4.277}