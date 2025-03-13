#!/usr/bin/env python3
# YOLOv8n-seg.onnx 模型运行器
# 基于 tinygrad_compile3_ok.py 的结构

import cv2
import numpy as np
import os, sys, pickle, time, argparse
from pathlib import Path

# 设置环境变量以优化性能
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"  # 使用 FP16 精度
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"      # 图像处理优化
if "GPU" not in os.environ: os.environ["GPU"] = "1"          # 使用 GPU

from tinygrad import Tensor, Device
from tinygrad.helpers import getenv
from extra.onnx import OnnxRunner
import onnx

# 设置默认设备为 GPU
Device.DEFAULT = "GPU"

# 默认模型路径
DEFAULT_MODEL_PATH = "/tmp/yolov8n-seg.onnx"

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="运行 YOLOv8n-seg ONNX 模型")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="ONNX 模型文件路径 (默认: /tmp/yolov8n-seg.onnx)")
    parser.add_argument("--video", type=str, default="0",
                        help="视频源，可以是摄像头索引(例如:0)或视频文件路径")
    parser.add_argument("--image", type=str,
                        help="图像文件路径，如果提供则处理单张图像而非视频")
    parser.add_argument("--conf-thres", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="IoU 阈值 (默认: 0.45)")
    parser.add_argument("--save", action="store_true",
                        help="保存处理结果")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示处理结果")
    return parser.parse_args()

# 获取模型文件路径
def get_model_file(model_path):
    """
    获取模型文件路径，如果不存在则尝试下载或创建
    
    参数:
        model_path: 模型路径
        
    返回:
        本地模型文件路径
    """
    model_file = Path(model_path)
    
    # 如果模型文件不存在，尝试创建
    if not model_file.is_file():
        print(f"模型文件 {model_path} 不存在，尝试创建...")
        try:
            from ultralytics import YOLO
            os.chdir("/tmp")
            model = YOLO("yolov8n-seg.pt")
            model.export(format="onnx", imgsz=[480, 640])
            print(f"成功创建模型文件: {model_path}")
        except Exception as e:
            print(f"创建模型文件失败: {e}")
            sys.exit(1)
    
    return str(model_file)

# 加载 ONNX 模型
def load_model(model_path):
    """
    加载 ONNX 模型
    
    参数:
        model_path: ONNX 模型文件路径
        
    返回:
        加载的模型
    """
    print(f"加载模型: {model_path}")
    try:
        onnx_model = onnx.load(open(model_path, "rb"))
        model = OnnxRunner(onnx_model)
        print("模型加载成功")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

# 预处理图像
def preprocess_image(image):
    """
    预处理图像以适应模型输入
    
    参数:
        image: 输入图像 (BGR 格式)
        
    返回:
        预处理后的图像张量
    """
    # 保存原始图像尺寸用于后处理
    orig_height, orig_width = image.shape[:2]
    
    # 调整图像大小为模型输入尺寸 (480, 640)
    input_height, input_width = 480, 640
    resized = cv2.resize(image, (input_width, input_height))
    
    # 转换为 RGB 并归一化 (0-1)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    
    # 添加批次维度并转换为 NCHW 格式
    # [H, W, C] -> [1, C, H, W]
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    
    return Tensor(tensor), (orig_height, orig_width)

# 后处理模型输出
def postprocess_outputs(outputs, orig_size, conf_thres=0.25, iou_thres=0.45):
    """
    后处理模型输出，包括边界框、置信度和分割掩码
    
    参数:
        outputs: 模型输出
        orig_size: 原始图像尺寸 (height, width)
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        
    返回:
        处理后的检测结果
    """
    # 将输出转换为 numpy 数组
    outputs = [out.numpy() if isinstance(out, Tensor) else out for out in outputs]
    
    # 提取检测输出和分割输出
    # YOLOv8-seg 通常有两个输出:
    # 1. 检测输出: [batch, num_boxes, 85+num_classes] (包括边界框、置信度和类别)
    # 2. 分割输出: [batch, num_boxes, num_masks, mask_height, mask_width]
    
    # 这里简化处理，仅返回原始输出供可视化函数使用
    return outputs

# 可视化检测结果
def visualize_outputs(image, outputs, orig_size, conf_thres=0.25):
    """
    可视化模型输出，包括边界框和分割掩码
    
    参数:
        image: 原始图像
        outputs: 模型输出
        orig_size: 原始图像尺寸
        conf_thres: 置信度阈值
        
    返回:
        可视化后的图像
    """
    # 为简化示例，这里只绘制一个示例文本
    # 实际应用中应该解析输出并绘制边界框和分割掩码
    
    # 创建可视化图像的副本
    vis_image = image.copy()
    
    # 添加示例文本
    cv2.putText(vis_image, "YOLOv8n-seg Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 添加运行设备信息
    device_info = f"Running on: {'GPU' if getenv('GPU', 0) else 'CPU'}"
    cv2.putText(vis_image, device_info, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return vis_image

# 处理单张图像
def process_image(model, image_path, conf_thres=0.25, iou_thres=0.45, save=False, display=True):
    """
    处理单张图像
    
    参数:
        model: 加载的模型
        image_path: 图像文件路径
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        save: 是否保存结果
        display: 是否显示结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 预处理图像
    input_tensor, orig_size = preprocess_image(image)
    
    # 模型推理
    start_time = time.time()
    outputs = model({"images": input_tensor})
    inference_time = time.time() - start_time
    print(f"推理时间: {inference_time:.4f} 秒")
    
    # 后处理输出
    results = postprocess_outputs(outputs, orig_size, conf_thres, iou_thres)
    
    # 可视化结果
    vis_image = visualize_outputs(image, results, orig_size, conf_thres)
    
    # 显示结果
    if display:
        cv2.imshow("YOLOv8n-seg Detection", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果
    if save:
        output_path = f"output_{Path(image_path).name}"
        cv2.imwrite(output_path, vis_image)
        print(f"结果已保存至: {output_path}")

# 处理视频流
def process_video(model, video_source, conf_thres=0.25, iou_thres=0.45, save=False, display=True):
    """
    处理视频流
    
    参数:
        model: 加载的模型
        video_source: 视频源，可以是摄像头索引或视频文件路径
        conf_thres: 置信度阈值
        iou_thres: IoU 阈值
        save: 是否保存结果
        display: 是否显示结果
    """
    # 打开视频源
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        print(f"打开摄像头: {video_source}")
    else:
        cap = cv2.VideoCapture(video_source)
        print(f"打开视频文件: {video_source}")
    
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 创建视频写入器
    video_writer = None
    if save:
        output_path = f"output_{Path(video_source).name if not video_source.isdigit() else 'camera.mp4'}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理帧
    frame_count = 0
    total_time = 0
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 预处理图像
            input_tensor, orig_size = preprocess_image(frame)
            
            # 模型推理
            start_time = time.time()
            outputs = model({"images": input_tensor})
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # 计算 FPS
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            # 后处理输出
            results = postprocess_outputs(outputs, orig_size, conf_thres, iou_thres)
            
            # 可视化结果
            vis_frame = visualize_outputs(frame, results, orig_size, conf_thres)
            
            # 添加 FPS 信息
            cv2.putText(vis_frame, f"FPS: {avg_fps:.2f}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示结果
            if display:
                cv2.imshow("YOLOv8n-seg Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存结果
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # 每 30 帧打印一次状态
            if frame_count % 30 == 0:
                print(f"已处理 {frame_count} 帧, 平均 FPS: {avg_fps:.2f}")
    
    except KeyboardInterrupt:
        print("用户中断处理")
    
    finally:
        # 释放资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print(f"总共处理了 {frame_count} 帧")
        if frame_count > 0:
            print(f"平均 FPS: {frame_count / total_time:.2f}")

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 获取模型文件
    model_path = get_model_file(args.model)
    
    # 加载模型
    model = load_model(model_path)
    
    # 打印设备信息
    print(f"运行设备: {'GPU' if getenv('GPU', 0) else 'CPU'}")
    
    # 处理输入
    if args.image:
        # 处理单张图像
        process_image(model, args.image, args.conf_thres, args.iou_thres,
                     args.save, not args.no_display)
    else:
        # 处理视频流
        process_video(model, args.video, args.conf_thres, args.iou_thres,
                     args.save, not args.no_display)

if __name__ == "__main__":
    main()
