<div align="center">

# 🏗️ YOLOv5n 工地 PPE 智能防护检测系统

**基于 YOLOv5n + TensorRT + DeepStream 的实时工地安全装备检测方案**

![](https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220404103032.png?x-oss-process=style/wp)

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-v6.0-green.svg)](https://github.com/ultralytics/yolov5)
[![TensorRT](https://img.shields.io/badge/TensorRT-加速推理-orange.svg)](https://developer.nvidia.com/tensorrt)
[![DeepStream](https://img.shields.io/badge/DeepStream-6.0-76b900.svg)](https://developer.nvidia.com/deepstream-sdk)
[![License](https://img.shields.io/badge/License-学习研究-yellow.svg)](#)

</div>

---

> ⚠️ 本项目是本人授课使用，请仅做个人学习、研究使用。

## 📌 项目简介

本项目基于 **YOLOv5n** 轻量级目标检测模型，实现对工地现场人员 **个人防护装备 (PPE, Personal Protective Equipment)** 的实时智能检测。系统可识别6类目标：**人员 (person)**、**反光背心 (vest)**、**蓝色安全帽**、**红色安全帽**、**白色安全帽**、**黄色安全帽**，并通过 **IOU 关联算法** 将检测到的帽子和背心与人体框进行语义绑定，实现逐人状态判断（是否佩戴安全帽、是否穿着反光背心）。

支持多种推理部署方式：
| 部署方式 | 文件 | 适用平台 | 预期帧率 |
|---------|------|---------|---------|
| **PyTorch 推理** | `demo.py` | PC (Windows/macOS/Linux) | ~10-15 FPS |
| **PyTorch FP16 半精度** | `detect.py --half` | GPU 服务器 | ~10 FPS (97.5ms/帧) |
| **ONNX GPU 加速** | `detect.py --weights *.onnx` | GPU 服务器 (CUDA 11.x) | **~40 FPS (24.9ms/帧)** |
| **TensorRT 加速** | `yolo_trt_demo.py` | Jetson Nano / GPU 服务器 | ~20 FPS |
| **DeepStream 管线** | `DeepStream6.0_Yolov5-6.0/` | Jetson Nano (NVIDIA 平台) | 最优 |

---

## 🔬 创新点与技术亮点

### 1. 💡 轻量化模型选择 — YOLOv5n
- 选用 YOLOv5 **Nano** 版本（仅 **1.9M 参数量**，**4.5 GFLOPs**），在保证检测精度的前提下大幅降低计算需求，使得模型可以在 **Jetson Nano** 等边缘设备上实现实时推理。

### 2. 🧠 基于 IOU 的人-装备语义关联
- 传统检测仅输出独立的目标框。本项目创新性地引入 **IOU（交并比）关联机制**，通过计算人体框与帽子框/背心框的 IOU 值，将装备检测结果绑定到对应人员，实现 **逐人安全状态判断**。
- 该方法避免了复杂的 ReID 或 Tracking 算法，在保证关联准确性的同时具有极低的计算开销。

### 3. 🖼️ 状态可视化浮层渲染
- 采用 **图标浮层 (Overlay Icon)** 方式直观展示每位人员的装备佩戴状态，包括安全帽颜色和背心穿戴情况。
- 当人员未佩戴安全帽或未穿反光背心时，显示对应的 **警告图标**，便于安全监管人员快速识别违规情况。

### 4. 🚀 四级推理部署架构
- **PC 端 PyTorch**：快速验证与调试；
- **ONNX Runtime GPU**：ONNX 模型 + CUDA GPU 加速，实现服务器端高速推理（24.9ms/帧）；
- **Jetson TensorRT**：INT8/FP16 量化加速，实现边缘端实时推理；
- **DeepStream 管线**：端到端 GPU 加速视频分析流水线，适用于多路视频流工业部署场景。

### 5. 📊 多模型对比训练
- 同时训练了 YOLOv5 **n / s / m / n6** 四个规模的模型，提供完整的精度-速度 trade-off 参考，方便用户根据自身硬件条件选择最优模型。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    摄像头输入 (USB/RTSP)                   │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│          YOLOv5n 目标检测 (6类目标)                       │
│   person / vest / blue helmet / red helmet /             │
│   white helmet / yellow helmet                           │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│        IOU 关联引擎 — 人-装备绑定                         │
│   • 计算 person 框与 helmet 框的 IOU                     │
│   • 计算 person 框与 vest 框的 IOU                       │
│   • 生成 person_info_list                                │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│        可视化渲染 & 状态告警                               │
│   • 绘制检测框 + 置信度                                   │
│   • 叠加安全帽/背心状态图标                                │
│   • FPS 与人员计数显示                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 一、硬件要求

| 组件 | 说明 |
|------|------|
| **PC 端** | Windows 10/11（无需 GPU，有最好）或 macOS 均测试可行 |
| **摄像头** | USB RGB 摄像头 |
| **边缘设备** | NVIDIA Jetson Nano（可选，用于 TensorRT / DeepStream 部署） |

## 二、软件依赖

| 软件 | 版本要求 |
|------|---------|
| Python | == 3.8 |
| PyTorch | >= 1.8.0 |
| YOLOv5 | v6.0 |
| OpenCV | >= 4.1.1 |
| NumPy | >= 1.22.2 |
| PyCUDA | （TensorRT 模式需要） |
| TensorRT | （TensorRT 模式需要） |

## 三、快速开始

### 📥 1. 克隆项目

```bash
git clone https://github.com/hexuanJ/YOLOv5-.git
cd YOLOv5-
```

### 📦 2. 准备 YOLOv5 环境

参考 [YOLOv5 官网](https://github.com/ultralytics/yolov5)，将 YOLOv5 clone 到本项目 `yolov5` 目录（当前 YOLOv5 目录为空，替换即可）：

```bash
git clone https://github.com/ultralytics/yolov5.git yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

### ⬇️ 3. 下载权重文件

下载训练好的权重文件（如 `ppe_yolo_n.pt`）放到 `weights` 目录下：

👉 [权重下载地址](https://github.com/enpeizhao/CVprojects/releases/tag/Models)

### ▶️ 4. 运行检测

```bash
# PC 端 PyTorch 推理
python demo.py

# Jetson Nano TensorRT 加速推理 (~20FPS)
python yolo_trt_demo.py
```

### ⚡ 5. ONNX GPU 加速推理部署

> 在 Tesla V100S-PCIE-32GB 上实测，ONNX GPU 推理速度为 **24.9ms/帧**，相比 PyTorch FP16（97.5ms/帧）快约 **4 倍**，相比 ONNX CPU 回退（357.7ms/帧）快约 **14 倍**。

#### 实测性能对比（Tesla V100S-PCIE-32GB）

| 推理方式 | 推理速度 (inference) | 加速比 | 状态 |
|---------|---------------------|--------|------|
| ONNX CPU（回退） | 357.7 ms | 1x（基准） | ❌ GPU 未启用 |
| PyTorch FP16 | 97.5 ms | 3.7x | ✅ 可用 |
| **ONNX GPU** | **24.9 ms** | **14.4x** | ✅ **最佳方案** |

#### 测试环境

| 项目 | 版本 |
|------|------|
| GPU | Tesla V100S-PCIE-32GB |
| Python | 3.10.11 |
| PyTorch | 2.0.1+cu118 |
| ONNX Runtime GPU | 1.16.3 |
| ONNX | 1.20.1 |

#### 方式一：一键配置（推荐）

```bash
# 运行一键配置脚本
chmod +x setup_onnx_gpu.sh
./setup_onnx_gpu.sh
```

#### 方式二：手动配置

**Step 1：安装依赖**

```bash
# 安装 ONNX 相关
pip install onnx
pip uninstall onnxruntime onnxruntime-gpu -y 2>/dev/null
pip install onnxruntime-gpu==1.16.3

# 安装 NVIDIA CUDA 运行时库
pip install nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-curand-cu11 \
            nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cufft-cu11
```

> ⚠️ `onnxruntime-gpu` 必须安装 **1.16.3** 版本，该版本兼容 CUDA 11.8。默认 `pip install onnxruntime-gpu` 会安装最新版（要求 CUDA 12），与 CUDA 11.x 环境不兼容。

**Step 2：创建符号链接**

PyTorch 自带的 `libnvrtc` 文件名带有哈希后缀，cuDNN 加载时找不到，需要创建符号链接：

```bash
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
NVRTC_FILE=$(ls ${TORCH_LIB}/libnvrtc-*.so.* 2>/dev/null | head -1)
ln -sf "$NVRTC_FILE" "${TORCH_LIB}/libnvrtc.so"
```

**Step 3：设置环境变量**

```bash
export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:/usr/local/lib/python3.10/site-packages/nvidia/curand/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
```

如需持久化（重启终端自动生效），将上述 `export` 命令追加到 `~/.bashrc`：

```bash
echo 'export LD_LIBRARY_PATH=...(同上)...' >> ~/.bashrc
```

**Step 4：验证 ONNX GPU**

```bash
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

输出中应包含 `CUDAExecutionProvider`。

**Step 5：导出 ONNX 模型**

```bash
cd yolov5
python export.py --weights yolov5n.pt --include onnx --device 0 --simplify
```

> ⚠️ 推荐使用 **FP32** 导出（不加 `--half`），避免推理时数据类型不匹配。

**Step 6：ONNX GPU 推理**

```bash
# 对图片推理
python detect.py --weights yolov5n.onnx --source data/images/ --device 0 --name result --exist-ok

# 对视频推理
python detect.py --weights yolov5n.onnx --source video.mp4 --device 0 --name result --exist-ok
```

#### 常见问题

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Require cuDNN 9.* and CUDA 12.*` | onnxruntime-gpu 版本过高 | `pip install onnxruntime-gpu==1.16.3` |
| `libcurand.so.10: cannot open` | 缺少 CUDA 库 | `pip install nvidia-curand-cu11` |
| `libcufft.so.10: cannot open` | 缺少 cufft 库 | `pip install nvidia-cufft-cu11` |
| `libnvrtc.so: cannot open` | PyTorch 库文件名带哈希 | 创建 `libnvrtc.so` 符号链接（见 Step 2） |
| `expected: (tensor(float16))` | 模型用 --half 导出 | 重新导出 FP32 模型（去掉 `--half`） |
| `Failed to open 0` | 云服务器无摄像头 | 使用图片/视频作为 `--source` |

---

### 🎯 6. DeepStream 部署

Deepstream 参考 NVIDIA DeepStream SDK 描述运行，对应目录：`DeepStream6.0_Yolov5-6.0`。

具体参考：
- TensorRT Engine 生成: https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5
- DeepStream-YOLO: https://github.com/marcoslucianops/DeepStream-Yolo

---

## 四、模型评估

### Ground Truths vs 预测对比

| Ground Truths | 模型预测 |
|:---:|:---:|
| ![](imgs/val_batch1_labels.jpg) | ![](imgs/val_batch1_pred.jpg) |

### 训练模型一览

共训练了 YOLOv5 **n、m、s、n6** 四个模型：

<img src="https://enpei-md.oss-cn-hangzhou.aliyuncs.com/img20220404104234.png?x-oss-process=style/wp" style="zoom:50%;" />

各个模型评估数据如下：

```shell
# n — 4.3 GFLOPs
Class     Images     Labels     P         R        mAP@.5   mAP@.5:.95
all        121        776     0.783     0.693     0.754      0.41
person     121        198     0.863     0.804     0.859     0.504
vest       121         98     0.769     0.643     0.727     0.424
blue       121         92     0.809     0.717     0.785     0.435
red        121        105     0.788     0.724     0.771     0.413
white      121        189     0.706       0.6     0.647     0.315
yellow     121         94     0.764     0.67      0.736     0.371

# s — 15.8 GFLOPs
Class     Images     Labels     P         R        mAP@.5   mAP@.5:.95
all        121        776     0.832     0.741     0.794     0.461
person     121        198     0.883     0.828     0.876     0.553
vest       121         98     0.816     0.735     0.797     0.499
blue       121         92     0.831     0.761     0.826     0.485
red        121        105     0.849     0.79      0.817     0.471
white      121        189     0.784     0.651     0.688     0.357
yellow     121         94     0.832     0.681     0.762     0.402

# m — 47.9 GFLOPs
Class     Images     Labels     P         R        mAP@.5   mAP@.5:.95
all        121        776     0.865     0.743     0.819     0.487
person     121        198     0.932     0.813     0.893     0.576
vest       121         98     0.836     0.765     0.815     0.508
blue       121         92     0.861     0.761     0.829     0.489
red        121        105     0.876     0.78      0.844     0.503
white      121        189     0.815     0.653     0.725     0.4
yellow     121         94     0.868     0.685     0.805     0.443

# n6 — 5.4 GFLOPs (P6 模型，输入 1280px)
Class     Images     Labels     P         R        mAP@.5   mAP@.5:.95
all        121        776     0.785     0.701     0.762     0.422
person     121        198     0.865     0.798     0.858     0.519
vest       121         98     0.761     0.684     0.737     0.432
blue       121         92     0.805     0.728     0.785     0.436
red        121        105     0.79      0.724     0.781     0.428
white      121        189     0.72      0.597     0.666     0.33
yellow     121         94     0.767     0.676     0.746     0.387
```

---

## 五、核心代码解析

### 📄 `demo.py` — PyTorch 推理入口

| 模块 | 功能 |
|------|------|
| `PPE_detect.__init__()` | 加载 YOLOv5n 模型、设置置信度阈值、初始化摄像头、加载状态图标 |
| `get_iou()` | 计算两个矩形框的 IOU，用于人-装备关联 |
| `get_person_info_list()` | 遍历每个人体框，通过 IOU 与帽子框/背心框进行匹配绑定 |
| `render_frame()` | 在画面上绘制检测框、置信度文本、状态图标浮层 |
| `detect()` | 主循环：读取帧 → 推理 → 关联 → 渲染 → 显示 |

### 📄 `yolo_trt_demo.py` — TensorRT 加速推理

| 模块 | 功能 |
|------|------|
| `YoLov5TRT.__init__()` | 反序列化 TensorRT Engine，分配 CUDA Host/Device 缓存 |
| `preprocess_image()` | BGR→RGB、等比缩放 + Padding、归一化、HWC→NCHW |
| `xywh2xyxy()` | 将模型输出的中心点+宽高格式转换为左上右下角点坐标 |
| `non_max_suppression()` | 手写 NMS（非极大值抑制），过滤冗余框 |
| `infer()` | 主循环：CUDA 推理 → 后处理 → 绘制 → 显示 |

---

## 六、项目结构

```
YOLOv5-/
├── demo.py                          # PyTorch 推理主程序
├── yolo_trt_demo.py                 # TensorRT 加速推理
├── setup_onnx_gpu.sh                # ONNX GPU 一键配置脚本
├── weights/                         # 模型权重文件目录
│   └── ppe_yolo_n.pt               # YOLOv5n PPE 检测权重
├── icons/                           # 状态显示图标
│   ├── person.png                   # 人员图标
│   ├── vest_on.png                  # 穿背心图标
│   ├── vest_off.png                 # 未穿背心警告图标
│   ├── hat_blue.png                 # 蓝色安全帽
│   ├── hat_red.png                  # 红色安全帽
│   ├── hat_white.png                # 白色安全帽
│   ├── hat_yellow.png               # 黄色安全帽
│   └── hat_off.png                  # 未戴帽子警告图标
├── imgs/                            # 验证集可视化结果
├── yolov5/                          # YOLOv5 框架（需自行 clone）
├── DeepStream6.0_Yolov5-6.0/       # DeepStream 部署配置
└── README.md                        # 项目说明文档
```

---

## 七、未来展望 & TODO

- [x] ⚡ ONNX Runtime GPU 加速推理（实测 24.9ms/帧，较 PyTorch FP16 快 4 倍）
- [ ] 🔄 引入目标跟踪（如 ByteTrack / DeepSORT），实现跨帧人员 ID 追踪
- [ ] 📢 增加告警机制：未佩戴 PPE 时触发声音/消息告警
- [ ] 🌐 开发 Web 端可视化界面，支持远程监控
- [ ] 📈 使用更大数据集重新训练，提升白色安全帽等难识别类别的精度
- [ ] 🔧 支持 YOLOv8 / YOLO11 等新一代模型的迁移适配
- [ ] 🎥 支持多路 RTSP 视频流同时检测
- [ ] 📱 移植到 Android / iOS 移动端

---

## 八、参考与致谢

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [CVprojects 权重发布](https://github.com/enpeizhao/CVprojects/releases/tag/Models)

---

## 📜 License

本项目仅供个人学习和研究使用。模型基于 YOLOv5 ([AGPL-3.0 License](https://github.com/ultralytics/yolov5/blob/master/LICENSE))。

---
