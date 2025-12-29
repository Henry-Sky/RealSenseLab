import numpy as np
# 为旧版insightface添加向后兼容
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import timeit
from utils.file import BASE_DIR
from insightface.data import get_image as ins_get_image
from insightface.model_zoo import get_model

import os
os.environ['ORT_CUDA_FP16_ENABLE'] = '1'


sizes = [(640,640), (480,480), (256, 256), (160, 160)]
dict = {}
NUM = 50

det = get_model('models/buffalo_l/det_10g.onnx', providers=[ 'CUDAExecutionProvider'], root=BASE_DIR)

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
    dict[size] = t_ms


print(dict)
print(faces)

