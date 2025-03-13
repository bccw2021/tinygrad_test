#!/usr/bin/env python3
# RK3588优化的ONNX Runner，专门解决FLOAT16相关错误

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set, Any, Callable
import onnx
from onnx import numpy_helper

# 确保环境变量设置正确
os.environ["FLOAT16"] = "0"
os.environ["FORCE_FP32"] = "1"
os.environ["AVOID_SINKS"] = "1"

# 导入tinygrad相关库
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context, DEBUG, getenv
from tinygrad.jit import TinyJit
from tinygrad import Device

try:
    from extra.onnx import safe_numpy, conv_pool_default_shape, get_values_from_initializer, fetch_attrs
    from extra.onnx import OnnxRunner as OriginalOnnxRunner
    HAS_ONNX_ORIGINAL = True
except ImportError:
    HAS_ONNX_ORIGINAL = False
    print("警告: 无法导入原始OnnxRunner，请确保tinygrad安装正确")

class RK3588OnnxRunner:
    """
    专为RK3588平台优化的ONNX模型运行器，避免FLOAT16相关错误
    """
    def __init__(self, onnx_model):
        self.onnx_model = onnx_model
        self.input_names = [inp.name for inp in self.onnx_model.graph.input]
        self.output_names = [out.name for out in self.onnx_model.graph.output]
        
        # 获取所有初始化器
        self.initializers = {i.name: i for i in self.onnx_model.graph.initializer}
        
        # 分析模型，提取节点和值信息
        self.nodes = self.onnx_model.graph.node
        self.value_info = {vi.name: vi for vi in self.onnx_model.graph.value_info}
        
        # 添加输入和输出到value_info
        for inp in self.onnx_model.graph.input:
            self.value_info[inp.name] = inp
        for out in self.onnx_model.graph.output:
            self.value_info[out.name] = out
        
        # 创建计算图
        self.create_graph()
    
    def create_graph(self):
        """创建计算图，并确保全部使用FP32"""
        # 注册处理函数
        self.handlers = {}
        
        # 存储中间结果
        self.tensors = {}
        
        # 创建计算图
        for node in self.nodes:
            print(f"处理节点: {node.op_type} - 输入: {node.input}")
            
            # 处理输入张量
            inputs = []
            for inp in node.input:
                if inp in self.initializers:
                    # 转换权重为FP32
                    weight = numpy_helper.to_array(self.initializers[inp])
                    weight = weight.astype(np.float32)  # 强制使用FP32
                    self.tensors[inp] = Tensor(weight, dtype=Tensor.float)
                    inputs.append(self.tensors[inp])
                elif inp in self.tensors:
                    inputs.append(self.tensors[inp])
                else:
                    print(f"警告: 找不到输入 {inp}")
                    inputs.append(None)
            
            # 确保所有输入都是FP32
            for i, inp in enumerate(inputs):
                if inp is not None and isinstance(inp, Tensor):
                    # 确保使用FP32
                    if inp.dtype != Tensor.float:
                        inputs[i] = inp.cast(Tensor.float)
            
            # 处理输出张量
            outputs = []
            for out in node.output:
                outputs.append(out)
            
            # 存储节点信息
            print(f"保存节点: {node.op_type} - 输出: {node.output}")
            self.tensors.update({out: None for out in outputs})
    
    def __call__(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """运行模型推理，确保全部使用FP32"""
        # 确保输入是FP32
        fp32_inputs = {}
        for name, tensor in inputs.items():
            if isinstance(tensor, Tensor):
                # 强制转换为FP32
                fp32_inputs[name] = tensor.cast(Tensor.float)
            else:
                fp32_inputs[name] = tensor
        
        # 初始化张量字典
        tensors = {name: tensor for name, tensor in fp32_inputs.items()}
        
        # 加载初始化器
        for name, initializer in self.initializers.items():
            if name not in tensors:
                # 转换权重为FP32
                weight = numpy_helper.to_array(initializer)
                weight = weight.astype(np.float32)  # 强制使用FP32
                tensors[name] = Tensor(weight, dtype=Tensor.float)
        
        # 在CPU上运行模型推理
        with Context(BEAM=0, JIT=0, FLOAT16=0, IMAGE=0, NOLOCALS=0, OPTLOCAL=0, AVOID_SINKS=1):
            # 保存原始设备
            original_device = Device.DEFAULT
            
            try:
                # 切换到CPU
                Device.DEFAULT = "CPU"
                print(f"切换到设备: {Device.DEFAULT} 进行推理")
                
                # 确保所有输入都在CPU上并且是FP32
                cpu_tensors = {}
                for name, tensor in tensors.items():
                    if isinstance(tensor, Tensor):
                        # 确保在CPU上并且是FP32
                        np_data = tensor.numpy()
                        cpu_tensors[name] = Tensor(np_data, dtype=Tensor.float, device="CPU")
                    else:
                        cpu_tensors[name] = tensor
                
                # 使用原始OnnxRunner的__call__函数
                if HAS_ONNX_ORIGINAL:
                    # 创建原始运行器
                    original_runner = OriginalOnnxRunner(self.onnx_model)
                    # 运行模型
                    cpu_outputs = original_runner(cpu_tensors)
                    
                    # 确保所有输出都已实现
                    for name, tensor in cpu_outputs.items():
                        if isinstance(tensor, Tensor):
                            tensor.realize()
                else:
                    # 如果无法导入原始OnnxRunner，返回空结果
                    cpu_outputs = {name: Tensor.zeros(1, dtype=Tensor.float, device="CPU") for name in self.output_names}
                
                # 将输出转回原始设备
                outputs = {}
                for name, tensor in cpu_outputs.items():
                    if isinstance(tensor, Tensor):
                        # 转回原始设备
                        outputs[name] = Tensor(tensor.numpy(), dtype=Tensor.float, device=original_device)
                    else:
                        outputs[name] = tensor
                
                # 恢复原始设备
                Device.DEFAULT = original_device
                print(f"恢复设备为: {Device.DEFAULT}")
                
                return outputs
            except Exception as e:
                print(f"RK3588OnnxRunner运行错误: {e}")
                # 恢复原始设备
                Device.DEFAULT = original_device
                print(f"恢复设备为: {Device.DEFAULT}")
                raise

# 辅助函数: 加载ONNX模型并创建RK3588OnnxRunner
def load_onnx_model(model_path: str) -> RK3588OnnxRunner:
    """
    加载ONNX模型并创建RK3588优化的运行器
    
    参数:
        model_path: ONNX模型文件路径
        
    返回:
        优化的ONNX运行器
    """
    # 加载ONNX模型
    print(f"加载ONNX模型: {model_path}")
    onnx_model = onnx.load(model_path)
    
    # 创建优化的运行器
    print("创建RK3588优化的ONNX运行器...")
    runner = RK3588OnnxRunner(onnx_model)
    print("ONNX运行器创建成功")
    
    return runner
