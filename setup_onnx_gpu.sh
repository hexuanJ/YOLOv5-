#!/bin/bash
# ============================================================
# YOLOv5 ONNX GPU 加速一键配置脚本
# 适用环境：CUDA 11.x + Python 3.10 + PyTorch 2.0.x
# 测试平台：Tesla V100S-PCIE-32GB
# ============================================================

set -e

echo "=== [1/6] 安装 ONNX ==="
pip install onnx

echo "=== [2/6] 安装 onnxruntime-gpu 1.16.3 (CUDA 11.x 兼容) ==="
pip uninstall onnxruntime onnxruntime-gpu -y 2>/dev/null || true
pip install onnxruntime-gpu==1.16.3

echo "=== [3/6] 安装 NVIDIA CUDA 运行时库 ==="
pip install nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-curand-cu11 \
            nvidia-cusolver-cu11 nvidia-cusparse-cu11 nvidia-cufft-cu11

echo "=== [4/6] 创建 libnvrtc.so 符号链接 ==="
TORCH_LIB=$(python -c "import torch; print(torch.__path__[0])")/lib
NVRTC_FILE=$(ls ${TORCH_LIB}/libnvrtc-*.so.* 2>/dev/null | head -1)
if [ -n "$NVRTC_FILE" ] && [ ! -f "${TORCH_LIB}/libnvrtc.so" ]; then
    ln -s "$NVRTC_FILE" "${TORCH_LIB}/libnvrtc.so"
    echo "Created symlink: libnvrtc.so -> $(basename $NVRTC_FILE)"
else
    echo "Symlink already exists or nvrtc not found, skipping."
fi

echo "=== [5/6] 设置 LD_LIBRARY_PATH ==="
NVIDIA_SITE=/usr/local/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=${TORCH_LIB}:${NVIDIA_SITE}/curand/lib:${NVIDIA_SITE}/cublas/lib:${NVIDIA_SITE}/cuda_runtime/lib:${NVIDIA_SITE}/cusolver/lib:${NVIDIA_SITE}/cusparse/lib:${NVIDIA_SITE}/cufft/lib:$LD_LIBRARY_PATH

echo "=== [6/6] 验证 ONNX GPU ==="
python -c "
import onnxruntime as ort
print('ONNX Runtime Version:', ort.__version__)
providers = ort.get_available_providers()
print('Available Providers:', providers)
if 'CUDAExecutionProvider' in providers:
    print('✅ ONNX GPU 加速可用！')
else:
    print('❌ CUDAExecutionProvider 不可用，请检查环境配置。')
"

echo ""
echo "=============================================="
echo "配置完成！使用以下命令进行推理："
echo ""
echo "  # 导出 ONNX 模型"
echo "  cd yolov5"
echo "  python export.py --weights yolov5n.pt --include onnx --device 0 --simplify"
echo ""
echo "  # ONNX GPU 推理"
echo "  python detect.py --weights yolov5n.onnx --source data/images/ --device 0 --name result --exist-ok"
echo ""
echo "⚠️  注意：每次新开终端需重新设置 LD_LIBRARY_PATH，"
echo "  或将 export 命令追加到 ~/.bashrc 中。"
echo "=============================================="
