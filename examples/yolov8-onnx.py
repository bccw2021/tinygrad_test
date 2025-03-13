#!/usr/bin/env python3
# 指定Python解释器路径

import os  # 导入操作系统模块，用于文件和目录操作
from ultralytics import YOLO  # 导入YOLO类，用于加载YOLOv8模型
import onnx  # 导入ONNX库，用于处理ONNX格式的模型
from pathlib import Path  # 导入Path类，用于文件路径操作
from extra.onnx import OnnxRunner  # 导入tinygrad的ONNX运行器，用于执行ONNX模型
from extra.onnx_helpers import get_example_inputs  # 导入辅助函数，用于生成示例输入数据
from tinygrad.tensor import Tensor  # 导入Tensor类，tinygrad的核心数据结构

os.chdir("/tmp")  # 将当前工作目录更改为/tmp
if not Path("yolov8n-seg.onnx").is_file():  # 检查ONNX模型文件是否已存在
  model = YOLO("yolov8n-seg.pt")  # 如果不存在，加载PyTorch格式的YOLOv8模型
  # ONNX 格式的模型通常需要固定的输入尺寸，因为它们的计算图是静态的。当导出 ONNX 模型时，必须指定这个固定尺寸，以便模型可以在各种推理引擎上正确运行。
  # Ultralytics 的 YOLO 实现会：
  # 创建指定尺寸的示例输入
  # 使用这个示例输入跟踪模型的计算图
  # 将跟踪后的计算图导出为 ONNX 格式
  # 在 ONNX 模型中嵌入输入尺寸信息
  model.export(format="onnx", imgsz=[480,640])  # 将模型导出为ONNX格式，指定输入图像尺寸为[480,640]
onnx_model = onnx.load(open("yolov8n-seg.onnx", "rb"))  # 加载ONNX模型文件
run_onnx = OnnxRunner(onnx_model)  # 创建ONNX运行器实例
run_onnx(get_example_inputs(run_onnx.graph_inputs), debug=True)  # 使用示例输入执行模型，并启用调试模式
