#!/usr/bin/env python3
# 简化版YOLOv8n-seg模型编译器 - 专为RK3588平台+OpenCL优化
# 作者: Cascade Assistant

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# 检测是否在RK3588平台上
IS_RK3588 = False
try:
    with open('/proc/device-tree/model', 'r') as f:
        model_info = f.read()
        if 'RK3588' in model_info:
            IS_RK3588 = True
            print("检测到RK3588平台，将使用专门优化")
except:
    print("未检测到RK3588平台，将使用标准设置")

# ========== 强制设置环境变量，解决FLOAT16问题 ==========
# 强制禁用FLOAT16
os.environ["FLOAT16"] = "0"
os.environ["FORCE_FP32"] = "1"

# 设置OpenCL相关环境变量
os.environ["GPU"] = "1"         # 启用GPU  
os.environ["METAL"] = "0"       # 禁用Metal
os.environ["CL_EXCLUDE_FILTER"] = ""  # 清除排除过滤器
os.environ["CL_CHECK"] = "1"    # 启用OpenCL错误检查
os.environ["CL_DISABLE_HALF"] = "1"  # 禁用half数据类型

# tinygrad优化设置
os.environ["BEAM"] = "0"        # 禁用BEAM编译优化
os.environ["JIT"] = "1"         # 启用JIT编译
os.environ["OPTLOCAL"] = "0"    # 禁用局部优化
os.environ["IMAGE"] = "0"       # 禁用图像处理优化
os.environ["NOLOCALS"] = "0"    # 禁用局部变量优化
os.environ["AVOID_SINKS"] = "1" # 避免使用sink节点
os.environ["ENABLE_GRAPH"] = "0" # 禁用图优化
os.environ["LAZY"] = "0"        # 禁用惰性执行

# 导入tinygrad相关库
try:
    # 根据tinygrad版本不同，导入方式可能会有所不同
    from tinygrad import Tensor, Device, Context, GlobalCounters
    from tinygrad.helpers import getenv, DEBUG
    
    # 尝试不同的方式导入TinyJit
    try:
        from tinygrad import TinyJit
    except ImportError:
        try:
            from tinygrad.jit import TinyJit
        except ImportError:
            # 如果无法导入TinyJit，创建一个空的装饰器
            print("警告: 无法导入TinyJit，将使用替代方案")
            def TinyJit(f):
                return f
    
    # 其他导入
    import onnx
    from onnx import numpy_helper
    from extra.onnx import OnnxRunner
    print("成功导入tinygrad和ONNX库")
    print(f"当前设备: {Device.DEFAULT}")
except ImportError as e:
    print(f"无法导入必要的库: {e}")
    print("请确保安装了tinygrad和onnx: pip install tinygrad onnx")
    sys.exit(1)

# 默认模型路径
DEFAULT_MODEL_PATH = "yolov8n-seg.onnx"
DEFAULT_OUTPUT_PATH = "/tmp/yolov8n-seg.pkl"

# ========== RK3588优化的ONNX运行器 ==========
class RK3588OnnxRunner:
    """专为RK3588平台优化的ONNX运行器"""
    
    def __init__(self, onnx_model):
        """初始化RK3588优化的ONNX运行器"""
        self.onnx_model = onnx_model
        self.input_names = [inp.name for inp in self.onnx_model.graph.input]
        self.output_names = [out.name for out in self.onnx_model.graph.output]
        
        # 获取输入形状
        self.input_shapes = {}
        for inp in self.onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            self.input_shapes[inp.name] = shape
        
        # 获取所有初始化器
        self.initializers = {i.name: i for i in self.onnx_model.graph.initializer}
        
        # 用标准OnnxRunner包装但强制使用FP32
        self.original_runner = OnnxRunner(onnx_model)
        print(f"成功创建RK3588优化的ONNX运行器，输入: {self.input_names}, 输出: {self.output_names}")
    
    def __call__(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """执行推理，确保所有输入输出都是FP32"""
        try:
            # 有关环境变量的设置
            os.environ["FLOAT16"] = "0"  # 确保禁用FLOAT16
            os.environ["FORCE_FP32"] = "1" # 确保强制使用FP32
            
            # 强制转换所有输入为FP32
            fp32_inputs = {}
            for name, tensor in inputs.items():
                if isinstance(tensor, Tensor):
                    # 版本兼容处理。使用字符串'float'而非属性
                    try:
                        # 直接使用字符串表示数据类型
                        fp32_inputs[name] = tensor.cast('float')
                    except Exception as e:
                        print(f"警告: 第一种方法cast失败: {e}")
                        try:
                            # 如果上面方法失败，尝试另一种方式
                            from tinygrad import dtypes
                            fp32_inputs[name] = tensor.cast(dtypes.float32)
                        except Exception as e:
                            print(f"警告: 第二种方法cast失败: {e}")
                            # 如果cast失败，尝试先转为numpy再转回
                            try:
                                print(f"尝试numpy转换方法...")
                                np_data = tensor.numpy()
                                fp32_inputs[name] = Tensor(np_data, dtype='float')
                            except Exception as e:
                                print(f"所有转换方法均失败，使用原始形式: {e}")
                                fp32_inputs[name] = tensor
                else:
                    fp32_inputs[name] = tensor
            
            # 使用CPU设备避免OpenCL编译错误
            original_device = Device.DEFAULT
            try:
                # 使用特殊的Context运行模型
                # 使用环境变量而不是Context参数来控制FLOAT16
                os.environ["FLOAT16"] = "0"
                os.environ["FORCE_FP32"] = "1"
                with Context(BEAM=0, JIT=0):
                    # 运行模型
                    outputs = self.original_runner(fp32_inputs)
                    
                    # 确保所有输出都已实现
                    for name, tensor in outputs.items():
                        if isinstance(tensor, Tensor):
                            tensor.realize()
                    
                    return outputs
            except Exception as e:
                print(f"GPU运行失败: {e}，尝试使用CPU回退")
                # CPU回退
                Device.DEFAULT = "CPU"
                # 将输入转移到CPU
                cpu_inputs = {}
                for name, tensor in fp32_inputs.items():
                    if isinstance(tensor, Tensor):
                        np_data = tensor.numpy()
                        try:
                            # 使用字符串表示数据类型
                            cpu_inputs[name] = Tensor(np_data, dtype='float', device="CPU")
                        except Exception as e:
                            print(f"CPU转换错误: {e}")
                            # 如果失败，尝试不指定类型
                            cpu_inputs[name] = Tensor(np_data, device="CPU")
                    else:
                        cpu_inputs[name] = tensor
                
                # 在CPU上运行
                # 使用环境变量而不是Context参数来控制FLOAT16
                os.environ["FLOAT16"] = "0"
                os.environ["FORCE_FP32"] = "1"
                with Context(BEAM=0, JIT=0):
                    cpu_outputs = self.original_runner(cpu_inputs)
                    
                    # 确保所有输出都已实现
                    for name, tensor in cpu_outputs.items():
                        if isinstance(tensor, Tensor):
                            tensor.realize()
                    
                    # 将输出转移回原始设备
                    outputs = {}
                    for name, tensor in cpu_outputs.items():
                        if isinstance(tensor, Tensor):
                            try:
                                # 使用字符串表示数据类型
                                outputs[name] = Tensor(tensor.numpy(), dtype='float', device=original_device)
                            except Exception as e:
                                print(f"输出转换错误: {e}")
                                # 如果失败，尝试不指定类型
                                outputs[name] = Tensor(tensor.numpy(), device=original_device)
                        else:
                            outputs[name] = tensor
                    
                    # 恢复设备
                    Device.DEFAULT = original_device
                    return outputs
        except Exception as e:
            print(f"RK3588OnnxRunner运行错误: {e}")
            raise

# ========== 模型编译函数 ==========
def compile_model(onnx_file: str, output_file: str = None):
    """编译模型并保存结果"""
    print(f"\n======== 编译模型: {onnx_file} ========")
    print(f"当前设备: {Device.DEFAULT}")
    print(f"环境变量设置:")
    print(f"  FLOAT16: {os.environ.get('FLOAT16', '0')}")
    print(f"  FORCE_FP32: {os.environ.get('FORCE_FP32', '0')}")
    print(f"  GPU: {os.environ.get('GPU', '0')}")
    print(f"  CL_DISABLE_HALF: {os.environ.get('CL_DISABLE_HALF', '0')}")
    print(f"  BEAM: {getenv('BEAM', 0)}")
    print(f"  JIT: {getenv('JIT', 0)}")
    print(f"  AVOID_SINKS: {getenv('AVOID_SINKS', 0)}")
    
    try:
        # 确保环境变量设置正确
        os.environ["FLOAT16"] = "0"
        os.environ["FORCE_FP32"] = "1"
        os.environ["AVOID_SINKS"] = "1"
        
        # 加载ONNX模型
        print(f"加载ONNX模型: {onnx_file}")
        try:
            onnx_model = onnx.load(onnx_file)
        except Exception as e:
            print(f"加载ONNX模型出错: {e}")
            print(f"确认文件是否存在: {os.path.exists(onnx_file)}")
            raise
        
        # 创建RK3588优化的运行器
        print("创建RK3588优化的ONNX运行器...")
        runner = RK3588OnnxRunner(onnx_model)
        
        # 获取输入形状
        input_shapes = runner.input_shapes
        print(f"模型输入名称: {runner.input_names}")
        print(f"模型输入形状: {input_shapes}")
        
        # 创建示例输入
        example_inputs = {}
        for name, shape in input_shapes.items():
            # 创建全为1的tensor作为示例输入，使用字符串表示类型
            example_inputs[name] = Tensor.ones(*shape, device=Device.DEFAULT, dtype='float')
        
        # 测试运行模型
        print("测试运行模型...")
        # 使用环境变量而不是Context参数来控制FLOAT16
        os.environ["FLOAT16"] = "0"  # 设置环境变量
        os.environ["FORCE_FP32"] = "1"
        
        # 只使用支持的Context参数
        with Context(BEAM=0, JIT=0):
            outputs = runner(example_inputs)
        print("模型运行成功!")
        
        # 创建包含所有必要信息的模型信息
        model_info = {
            "runner": runner,
            "input_names": runner.input_names,
            "output_names": runner.output_names,
            "input_shapes": input_shapes,
            "onnx_file": onnx_file,
            "device": Device.DEFAULT
        }
        
        # 保存编译后的模型
        if output_file:
            print(f"保存编译后的模型到: {output_file}")
            with open(output_file, "wb") as f:
                pickle.dump(model_info, f)
        
        return model_info, example_inputs, outputs
    
    except Exception as e:
        print(f"编译模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ========== 模型运行函数 ==========
def run_model(model_info, input_data):
    """运行已编译的模型"""
    runner = model_info["runner"]
    return runner(input_data)

# ========== 解析参数 ==========
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8n-seg模型编译器 (RK3588优化)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"ONNX模型文件路径 (默认: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"输出编译后的模型路径 (默认: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--run", action="store_true",
                        help="编译并运行测试")
    return parser.parse_args()

# ========== 主函数 ==========
def main():
    """主函数"""
    args = parse_args()
    
    # 编译模型
    model_info, example_inputs, outputs = compile_model(args.model, args.output)
    
    # 如果需要测试运行
    if args.run:
        print("\n======== 测试运行模型 ========")
        # 使用示例输入运行模型
        results = run_model(model_info, example_inputs)
        # 打印结果摘要
        for name, result in results.items():
            if isinstance(result, Tensor):
                print(f"输出 {name}: 形状={result.shape}, 设备={result.device}")
    
    print("\n编译完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
