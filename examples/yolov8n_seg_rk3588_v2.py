#!/usr/bin/env python3
# 简化版YOLOv8n-seg模型编译器 - 专为RK3588平台+OpenCL优化
# 版本2：极简设计，避免版本兼容性问题

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

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

try:
    # 导入tinygrad基础组件
    from tinygrad import Tensor, Device
    
    # 导入Context，处理不同版本
    try:
        from tinygrad import Context
    except ImportError:
        class Context:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.old_values = {}
            
            def __enter__(self):
                for k, v in self.kwargs.items():
                    env_var = k
                    self.old_values[env_var] = os.environ.get(env_var)
                    os.environ[env_var] = str(v)
            
            def __exit__(self, *args):
                for k, v in self.old_values.items():
                    if v is None:
                        del os.environ[k]
                    else:
                        os.environ[k] = v
    
    # 导入其他必要的库
    import onnx
    from onnx import numpy_helper
    
    # 尝试导入OnnxRunner
    try:
        from extra.onnx import OnnxRunner
    except ImportError:
        print("无法导入OnnxRunner，尝试替代导入方式")
        try:
            from tinygrad.runtime.onnx import OnnxRunner
        except ImportError:
            print("两种方式均无法导入OnnxRunner，请检查tinygrad安装")
            sys.exit(1)
    
    print("成功导入tinygrad和ONNX库")
    print(f"当前设备: {Device.DEFAULT}")
    
except ImportError as e:
    print(f"无法导入必要的库: {e}")
    print("请确保安装了tinygrad和onnx: pip install tinygrad onnx")
    sys.exit(1)

# 默认模型路径
DEFAULT_MODEL_PATH = "yolov8n-seg.onnx"
DEFAULT_OUTPUT_PATH = "/tmp/yolov8n-seg.pkl"

# ========== 超简化ONNX运行器 ==========
class SimpleOnnxRunner:
    """超简化版ONNX运行器，完全避免FLOAT16相关问题"""
    
    def __init__(self, onnx_model):
        """初始化简化版ONNX运行器"""
        self.onnx_model = onnx_model
        self.input_names = [inp.name for inp in self.onnx_model.graph.input]
        self.output_names = [out.name for out in self.onnx_model.graph.output]
        
        # 获取输入形状
        self.input_shapes = {}
        for inp in self.onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            self.input_shapes[inp.name] = shape
            
        # 尝试创建原始的OnnxRunner
        try:
            self.original_runner = OnnxRunner(onnx_model)
            print(f"成功创建ONNX运行器，输入: {self.input_names}, 输出: {self.output_names}")
        except Exception as e:
            print(f"创建原始OnnxRunner失败: {e}")
            self.original_runner = None
            raise
    
    def __call__(self, inputs):
        """运行推理，简化版"""
        # 确保环境变量设置正确
        os.environ["FLOAT16"] = "0"
        os.environ["FORCE_FP32"] = "1"
        
        try:
            # 记录原始设备
            original_device = Device.DEFAULT
            
            # 先尝试直接运行
            with Context(BEAM=0, JIT=0):
                outputs = self.original_runner(inputs)
                # 确保输出已实现
                for name, tensor in outputs.items():
                    if hasattr(tensor, 'realize'):
                        tensor.realize()
                return outputs
        except Exception as e:
            print(f"GPU运行出错: {e}")
            print("尝试使用CPU回退...")
            
            try:
                # 切换到CPU
                Device.DEFAULT = "CPU"
                
                # 创建CPU上的输入
                cpu_inputs = {}
                for name, tensor in inputs.items():
                    if hasattr(tensor, 'numpy'):
                        # 如果是Tensor，转到CPU
                        np_data = tensor.numpy()
                        # 避免使用dtype参数
                        cpu_inputs[name] = Tensor(np_data, device="CPU")
                    else:
                        cpu_inputs[name] = tensor
                
                # 在CPU上运行
                with Context(BEAM=0, JIT=0):
                    outputs = self.original_runner(cpu_inputs)
                    # 确保输出已实现
                    for name, tensor in outputs.items():
                        if hasattr(tensor, 'realize'):
                            tensor.realize()
                
                # 恢复原始设备
                Device.DEFAULT = original_device
                return outputs
            
            except Exception as inner_e:
                print(f"CPU回退也失败: {inner_e}")
                # 恢复原始设备
                Device.DEFAULT = original_device
                raise

# ========== 模型编译函数 ==========
def compile_model(onnx_file, output_file=None):
    """编译模型并保存结果"""
    print(f"\n======== 编译模型: {onnx_file} ========")
    print(f"当前设备: {Device.DEFAULT}")
    print(f"环境变量设置:")
    for var in ["FLOAT16", "FORCE_FP32", "GPU", "CL_DISABLE_HALF", "BEAM", "JIT", "AVOID_SINKS"]:
        print(f"  {var}: {os.environ.get(var, 'not set')}")
    
    try:
        # 加载ONNX模型
        print(f"加载ONNX模型: {onnx_file}")
        if not os.path.exists(onnx_file):
            print(f"错误: 模型文件 {onnx_file} 不存在")
            sys.exit(1)
            
        onnx_model = onnx.load(onnx_file)
        
        # 创建简化的运行器
        print("创建简化的ONNX运行器...")
        runner = SimpleOnnxRunner(onnx_model)
        
        # 获取输入形状
        input_shapes = runner.input_shapes
        print(f"模型输入名称: {runner.input_names}")
        print(f"模型输入形状: {input_shapes}")
        
        # 创建示例输入
        example_inputs = {}
        for name, shape in input_shapes.items():
            # 不指定dtype，让tinygrad自动推断
            example_inputs[name] = Tensor.ones(*shape, device=Device.DEFAULT)
        
        # 测试运行模型
        print("测试运行模型...")
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
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
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
    parser = argparse.ArgumentParser(description="YOLOv8n-seg简化模型编译器 (RK3588优化-V2)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"ONNX模型文件路径 (默认: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"输出编译后的模型路径 (默认: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--run", action="store_true",
                        help="编译并运行测试")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用CPU运行")
    return parser.parse_args()

# ========== 主函数 ==========
def main():
    """主函数"""
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return 1
    
    # 强制CPU模式
    if args.cpu:
        print("启用强制CPU模式")
        Device.DEFAULT = "CPU"
    
    # 编译模型
    model_info, example_inputs, outputs = compile_model(args.model, args.output)
    
    # 如果需要测试运行
    if args.run:
        print("\n======== 测试运行模型 ========")
        # 使用示例输入运行模型
        results = run_model(model_info, example_inputs)
        # 打印结果摘要
        for name, result in results.items():
            if hasattr(result, 'shape'):
                print(f"输出 {name}: 形状={result.shape}, 设备={result.device if hasattr(result, 'device') else 'unknown'}")
            else:
                print(f"输出 {name}: 类型={type(result)}")
    
    print("\n编译完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
