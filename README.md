## 课题A：基于RGB-D的3D人体目标检测
1.桌上放置Realsense传感器  
2.自动检测人体目标（能够区分照片和真人）  
3.得到人体的最小3D包围框  
*识别目标行为，包括：站立、坐下、挥手、行走。

性能优化：
1. 人脸识别模型（InsightFace）

| 后端 \ 输入尺寸      | 640×640 | 480×480 | 256×256 | 160×160 |
| -------------- | ------- | ------- | ------- | ------- |
| **CPU-L**      | 26.707  | 29.846  | 24.101  | 24.985  |
| **CPU-M**      | 10.212  | 10.481  | 10.676  | 10.267  |
| **CPU-S**      | 4.185   | 6.087   | 4.187   | 4.011   |
| **GPU-L**      | 9.282   | 5.599   | 5.577   | 5.820   |
| **GPU-M**      | 4.779   | 4.732   | 4.735   | 4.702   |
| **GPU-S**      | 4.006   | 4.049   | 4.001   | 3.996   |
| **TensorRT-L** | 3.612   | 3.619   | 3.593   | 3.603   |
| **TensorRT-M** | 3.306   | 3.313   | 3.290   | 3.299   |
| **TensorRT-S** | 5.518   | 4.775   | 4.175   | 4.277   |

2. 三维框图

- 人像检测  
![demo_person](https://github.com/Henry-Sky/RealSenseLab/raw/master/resources/imgs/demo_person.png)  

- 照片判别  
![demo_person](https://github.com/Henry-Sky/RealSenseLab/raw/master/resources/imgs/demo_photo.png) 
