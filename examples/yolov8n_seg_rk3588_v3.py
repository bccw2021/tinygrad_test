#!/usr/bin/env python3
# YOLOv8n-seg模型运行器 - 专为RK3588平台优化 (V3)
# 内存安全版本，解决段错误问题

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
import time
import gc
from typing import Dict, List, Tuple, Any

# ========== 环境变量配置 ==========
# 基本配置
os.environ["FLOAT16"] = "0"          # 强制禁用FLOAT16
os.environ["FORCE_FP32"] = "1"       # 强制使用FP32
os.environ["AVOID_SINKS"] = "1"      # 避免sink节点
os.environ["GPU"] = "1"              # 启用GPU

# OpenCL特定配置
os.environ["CL_DISABLE_HALF"] = "1"  # 禁用OpenCL half数据类型
os.environ["CL_CHECK"] = "1"         # 启用OpenCL错误检查
os.environ["CL_MEM_GROWTH"] = "256"  # 内存增长限制
os.environ["CL_EXCLUSIVE"] = "1"     # 独占模式
os.environ["CL_PRINT_COMPILE"] = "1" # 打印编译信息

# 针对RK3588的优化
os.environ["WGSIZE"] = "64"          # 工作组大小
os.environ["NOLOCALS"] = "1"         # 禁用局部变量
os.environ["CL_DISABLE_KERNEL_DEBUG"] = "1"  # 禁用内核调试
os.environ["CL_PRESERVE_PROGRAM_BINARIES"] = "1"  # 保存程序二进制

# 解释器和优化配置
os.environ["JIT"] = "0"              # 禁用JIT编译
os.environ["OPTLOCAL"] = "0"         # 禁用局部优化
os.environ["BEAM"] = "0"             # 禁用BEAM
os.environ["PYTHONUNBUFFERED"] = "1" # 无缓冲输出
os.environ["PYTHON"] = "1"           # 使用Python解释器模式

# 使用线程锁避免并发问题
os.environ["CLBUFFERCMD"] = "0"      # 禁用缓冲区命令
os.environ["PYOPENCL_NO_CACHE"] = "1"  # 禁用PyOpenCL缓存

# 调试输出控制
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"

def debug_print(*args, **kwargs):
    """调试打印函数"""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)

# 导入必要的库
try:
    import onnx
    import sys
    from tinygrad import Tensor, Device
    from tinygrad.helpers import getenv
    
    # 导入TinyJit和Context（处理不同版本）
    try:
        from tinygrad import Context
    except ImportError:
        # 如果无法导入Context，创建一个简单的替代版本
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
                        if env_var in os.environ:
                            del os.environ[k]
                    else:
                        os.environ[k] = v
    
    # 尝试导入ONNX运行器
    try:
        from extra.onnx import OnnxRunner
    except ImportError:
        try:
            from tinygrad.runtime.onnx import OnnxRunner
        except ImportError:
            print("警告: 无法直接导入OnnxRunner")
            OnnxRunner = None
    
    print("成功导入tinygrad和ONNX库")
    print(f"当前设备: {Device.DEFAULT}")
    
except ImportError as e:
    print(f"无法导入必要的库: {e}")
    print("请确保安装了tinygrad和onnx: pip install tinygrad onnx")
    sys.exit(1)

# ========== 安全的ONNX模型加载器 ==========
def safe_load_onnx(onnx_file_path):
    """安全地加载ONNX模型"""
    if not os.path.exists(onnx_file_path):
        print(f"错误: 模型文件 {onnx_file_path} 不存在")
        sys.exit(1)
        
    try:
        print(f"加载ONNX模型: {onnx_file_path}")
        onnx_model = onnx.load(onnx_file_path)
        return onnx_model
    except Exception as e:
        print(f"加载ONNX模型失败: {e}")
        sys.exit(1)

# ========== 内存安全的ONNX运行器 ==========
class SafeOnnxRunner:
    """内存安全的ONNX运行器，专为RK3588设计"""
    
    def __init__(self, onnx_model, memory_efficient=True):
        """初始化内存安全的ONNX运行器"""
        if OnnxRunner is None:
            raise ImportError("无法导入OnnxRunner，请确保tinygrad安装正确")
            
        self.onnx_model = onnx_model
        self.memory_efficient = memory_efficient
        self.input_names = [inp.name for inp in self.onnx_model.graph.input]
        self.output_names = [out.name for out in self.onnx_model.graph.output]
        
        # 获取输入形状
        self.input_shapes = {}
        for inp in self.onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            self.input_shapes[inp.name] = shape
        
        # 延迟初始化runner
        self.runner = None
        self.initialized = False
        print(f"创建内存安全的ONNX运行器，输入: {self.input_names}, 输出: {self.output_names}")
    
    def _ensure_initialized(self):
        """确保运行器已初始化"""
        if not self.initialized:
            try:
                # 按需初始化以节省内存
                print("初始化ONNX运行器...")
                self.runner = OnnxRunner(self.onnx_model)
                self.initialized = True
            except Exception as e:
                print(f"初始化ONNX运行器失败: {e}")
                raise
    
    def __call__(self, inputs):
        """安全执行推理"""
        self._ensure_initialized()
        
        # 保存原始设备
        original_device = Device.DEFAULT
        
        try:
            # 清理内存
            if self.memory_efficient:
                gc.collect()
            
            # 设置关键环境变量
            os.environ["FLOAT16"] = "0"
            os.environ["FORCE_FP32"] = "1"
            
            # 使用特殊上下文运行
            with Context(BEAM=0, JIT=0):
                # 分步处理以减少内存使用
                debug_print("使用GPU执行推理...")
                
                # 深拷贝输入以避免修改原始输入
                safe_inputs = {}
                for name, tensor in inputs.items():
                    if hasattr(tensor, 'numpy'):
                        # 确保用float32类型
                        np_data = tensor.numpy().astype(np.float32)
                        safe_inputs[name] = Tensor(np_data, device=Device.DEFAULT)
                    else:
                        safe_inputs[name] = tensor
                
                # 执行推理
                outputs = self.runner(safe_inputs)
                
                # 确保输出已实现
                for name, tensor in outputs.items():
                    if hasattr(tensor, 'realize'):
                        tensor.realize()
                
                # 清理内存
                if self.memory_efficient:
                    gc.collect()
                
                return outputs
                
        except Exception as e:
            print(f"GPU推理失败: {e}")
            try:
                if self.memory_efficient:
                    gc.collect()
                
                # 如果出错，尝试使用CPU
                print("尝试使用CPU回退...")
                # 切换到CPU
                Device.DEFAULT = "CPU"
                
                # 设置解释器模式以避免需要编译器
                os.environ["PYTHON"] = "1"
                os.environ["JIT"] = "0"
                
                # 创建CPU输入
                cpu_inputs = {}
                for name, tensor in inputs.items():
                    if hasattr(tensor, 'numpy'):
                        np_data = tensor.numpy().astype(np.float32)
                        cpu_inputs[name] = Tensor(np_data, device="CPU")
                    else:
                        cpu_inputs[name] = tensor
                
                # 在CPU上运行
                with Context(BEAM=0, JIT=0):
                    debug_print("使用CPU执行推理...")
                    cpu_outputs = self.runner(cpu_inputs)
                    
                    # 确保输出已实现
                    for name, tensor in cpu_outputs.items():
                        if hasattr(tensor, 'realize'):
                            tensor.realize()
                    
                    # 恢复设备
                    Device.DEFAULT = original_device
                    return cpu_outputs
                    
            except Exception as inner_e:
                print(f"CPU回退也失败: {inner_e}")
                Device.DEFAULT = original_device
                raise
        finally:
            # 确保设备恢复
            Device.DEFAULT = original_device
            if self.memory_efficient:
                gc.collect()

# ========== YOLO模型编译函数 ==========
def compile_yolo_model(onnx_file, output_file=None, memory_efficient=True):
    """编译YOLO模型"""
    print(f"\n======== 编译YOLO模型: {onnx_file} ========")
    print(f"当前设备: {Device.DEFAULT}")
    print(f"关键环境变量:")
    for key in ["FLOAT16", "FORCE_FP32", "GPU", "CL_DISABLE_HALF", "JIT", "PYTHON"]:
        print(f"  {key}: {os.environ.get(key, 'not set')}")
    
    try:
        # 加载ONNX模型
        onnx_model = safe_load_onnx(onnx_file)
        
        # 创建安全的运行器
        print(f"创建安全的ONNX运行器...")
        runner = SafeOnnxRunner(onnx_model, memory_efficient=memory_efficient)
        
        # 获取输入形状
        input_shapes = runner.input_shapes
        print(f"模型输入名称: {runner.input_names}")
        print(f"模型输入形状: {input_shapes}")
        
        # 创建示例输入
        print("创建示例输入...")
        example_inputs = {}
        for name, shape in input_shapes.items():
            # 创建全1的Tensor，明确使用float32
            data = np.ones(shape, dtype=np.float32)
            example_inputs[name] = Tensor(data, device=Device.DEFAULT)
        
        # 测试运行模型
        print("测试运行模型...")
        start_time = time.time()
        outputs = runner(example_inputs)
        end_time = time.time()
        print(f"模型运行成功! 耗时: {end_time - start_time:.3f}秒")
        
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

# ========== 运行YOLO模型 ==========
def run_model(model_info, input_data):
    """运行已编译的模型"""
    try:
        runner = model_info["runner"]
        return runner(input_data)
    except Exception as e:
        print(f"运行模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========== 命令行参数 ==========
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8n-seg优化运行器 (RK3588专用-V3)")
    parser.add_argument("--model", type=str, default="yolov8n-seg.onnx",
                        help="ONNX模型文件路径 (默认: yolov8n-seg.onnx)")
    parser.add_argument("--output", type=str, default="/tmp/yolov8n-seg.pkl",
                        help="输出编译后的模型路径 (默认: /tmp/yolov8n-seg.pkl)")
    parser.add_argument("--run", action="store_true", 
                        help="编译并运行测试")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用CPU运行")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式")
    parser.add_argument("--memory-efficient", action="store_true", default=True,
                        help="使用内存高效模式")
    return parser.parse_args()

# ========== 主函数 ==========
def main():
    """主函数"""
    args = parse_args()
    
    # 设置调试模式
    if args.debug:
        global DEBUG_MODE
        DEBUG_MODE = True
        os.environ["DEBUG_MODE"] = "1"
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return 1
    
    # 设置设备
    if args.cpu:
        print("启用强制CPU模式")
        Device.DEFAULT = "CPU"
        os.environ["GPU"] = "0"
        os.environ["PYTHON"] = "1"  # 使用Python解释器
    else:
        print("使用默认设备 (GPU)")
        os.environ["GPU"] = "1"
    
    # 编译模型
    model_info, example_inputs, outputs = compile_yolo_model(
        args.model, args.output, memory_efficient=args.memory_efficient
    )
    
    # 如果需要测试运行
    if args.run:
        print("\n======== 测试运行模型 ========")
        # 使用示例输入运行模型
        results = run_model(model_info, example_inputs)
        if results:
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
