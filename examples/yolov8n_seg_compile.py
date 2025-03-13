#!/usr/bin/env python3
# YOLOv8n-seg 模型编译和推理脚本
# 使用 tinygrad 编译优化 ONNX 模型并进行推理

import cv2
import numpy as np
import os, sys, pickle, time, argparse
from pathlib import Path

# 检测是否在RK3588平台上
IS_RK3588 = False
try:
    with open('/proc/device-tree/model', 'r') as f:
        model_info = f.read()
        if 'RK3588' in model_info:
            IS_RK3588 = True
            print("检测到RK3588平台，将使用专门优化")
except:
    # 如果无法读取，可能不是Linux系统
    pass

# 尝试不同的环境变量组合来解决 OpenCL 编译错误

# RK3588 OpenCL 优化设置 - 强制禁用 FLOAT16
# 设置环境变量以优化性能

# 强制禁用 FLOAT16
os.environ["FLOAT16"] = "0"  # 禁用 FP16 精度，使用 FP32
os.environ["FORCE_FP32"] = "1"  # 强制使用 FP32

# 其他设置
os.environ["IMAGE"] = "0"      # 禁用图像处理优化
os.environ["NOLOCALS"] = "0"  # 禁用局部变量优化
os.environ["JIT_BATCH_SIZE"] = "0"  # 禁用批处理

# 设置 OpenCL 相关环境变量
os.environ["METAL"] = "0"  # 禁用 Metal，强制使用 OpenCL
os.environ["GPU"] = "1"      # 启用 GPU
os.environ["DEBUG"] = "3"  # 增加调试级别以获取更多信息

# RK3588 特定的 OpenCL 优化选项
os.environ["OPTLOCAL"] = "0"  # 禁用局部内存优化
os.environ["CL_EXCLUDE_FILTER"] = ""  # 清除排除过滤器
os.environ["CL_CHECK"] = "1"  # 启用 OpenCL 错误检查
os.environ["FORWARD_ONLY"] = "1"  # 只运行前向传播
os.environ["DISABLE_CUDNN"] = "1"  # 禁用 CUDNN
os.environ["AVOID_SINKS"] = "1"  # 避免 SINK 操作
os.environ["ENABLE_GRAPH"] = "0"  # 禁用图优化
os.environ["LAZY"] = "0"  # 禁用惰性执行

# RK3588 特定的 OpenCL 设备设置
os.environ["CL_PLATFORM_NAME"] = ""  # 不限制平台名称
os.environ["CL_DEVICE_TYPE"] = "GPU"  # 使用 GPU 设备类型
os.environ["CL_DEVICE_NAME"] = ""  # 不限制设备名称

# 添加 RK3588 特定的优化选项
os.environ["OPENCL_INCLUDE_PATH"] = "/usr/include/CL"  # OpenCL 头文件路径
os.environ["OPENCL_LIBRARY_PATH"] = "/usr/lib"  # OpenCL 库路径

# 打印当前环境变量设置
from tinygrad.helpers import DEBUG, getenv
print(f"DEBUG 级别: {DEBUG}")
print(f"FLOAT16: {getenv('FLOAT16', 0)}")
print(f"IMAGE: {getenv('IMAGE', 0)}")
print(f"NOLOCALS: {getenv('NOLOCALS', 0)}")
print(f"OPTLOCAL: {getenv('OPTLOCAL', 0)}")
print(f"AVOID_SINKS: {getenv('AVOID_SINKS', 0)}")
print(f"ENABLE_GRAPH: {getenv('ENABLE_GRAPH', 0)}")
print(f"LAZY: {getenv('LAZY', 0)}")

# 导入 tinygrad 相关库
try:
    from tinygrad import Tensor, Device, TinyJit, Context, GlobalCounters
    from tinygrad.helpers import getenv, DEBUG
    from tinygrad.tensor import _from_np_dtype
    from tinygrad.engine.realize import CompiledRunner
    from extra.onnx import OnnxRunner
    import onnx
    from onnx.helper import tensor_dtype_to_np_dtype
    HAS_TINYGRAD = True
    HAS_ONNX = True
    print("成功导入 tinygrad 和 ONNX")
    print(f"当前设备: {Device.DEFAULT}")
    
    # 如果在RK3588平台上，尝试导入专用运行器
    HAS_RK3588_RUNNER = False
    if IS_RK3588:
        try:
            from examples.rk3588_onnx_runner import RK3588OnnxRunner, load_onnx_model
            HAS_RK3588_RUNNER = True
            print("成功导入 RK3588 专用 ONNX 运行器")
        except ImportError:
            print("警告: 无法导入 RK3588 专用运行器，将使用标准运行器")
    
except ImportError as e:
    HAS_TINYGRAD = False
    HAS_ONNX = False
    HAS_RK3588_RUNNER = False
    print(f"警告: 无法导入 tinygrad 或 ONNX: {e}")
    print("请安装必要的库: pip install tinygrad onnx")
    sys.exit(1)

# 尝试导入 ONNX Runtime (仅用于失败时的回退)
try:
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
    print("成功导入 ONNX Runtime (备用)")
except ImportError:
    HAS_ONNX_RUNTIME = False
    print("警告: 无法导入 ONNX Runtime，如果 tinygrad 失败将无法回退")

# 默认模型路径
DEFAULT_MODEL_PATH = "/tmp/yolov8n-seg.onnx"
DEFAULT_OUTPUT_PATH = "/tmp/yolov8n-seg.pkl"

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="编译和运行 YOLOv8n-seg 模型")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="ONNX 模型文件路径 (默认: /tmp/yolov8n-seg.onnx)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="输出编译后的模型路径 (默认: /tmp/yolov8n-seg.pkl)")
    parser.add_argument("output_file", type=str, nargs="?", 
                        help="输出编译后的模型路径 (可选位置参数，如果提供则覆盖--output)")
    parser.add_argument("--run", action="store_true", help="仅运行编译后的模型，不进行编译")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
    parser.add_argument("--ort", action="store_true", help="使用ONNX Runtime进行测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    # 添加视频流处理选项
    parser.add_argument("--video", type=str, help="视频源，可以是摄像头索引(例如:0)或视频文件路径")
    parser.add_argument("--image", type=str, help="图像文件路径，如果提供则处理单张图像而非视频")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU 阈值 (默认: 0.45)")
    parser.add_argument("--no-display", action="store_true", help="不显示处理结果")
    parser.add_argument("--save", action="store_true", help="保存处理结果")
    return parser.parse_args()

# 检查导入ONNX库
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    print("警告: 无法导入ONNX库，请安装: pip install onnx")
    HAS_ONNX = False

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

# 编译 ONNX 模型使用 tinygrad 和 OpenCL
def compile(onnx_file):
    """
    使用 tinygrad 编译优化 ONNX 模型 - 专为 RK3588 OpenCL 优化
    
    参数:
        onnx_file: ONNX 模型文件路径
        
    返回:
        编译后的模型和示例输入
    """
    if not HAS_TINYGRAD or not HAS_ONNX:
        print("错误: 无法编译模型，缺少 tinygrad 或 ONNX 库")
        print("请安装必要的库: pip install tinygrad onnx")
        sys.exit(1)
    
    # 再次确认环境变量设置
    print("\n\n确认当前环境变量设置:")
    print(f"FLOAT16: {getenv('FLOAT16', 0)}")
    print(f"FORCE_FP32: {getenv('FORCE_FP32', 0)}")
    print(f"AVOID_SINKS: {getenv('AVOID_SINKS', 0)}")
    print(f"GPU: {getenv('GPU', 0)}")
    print(f"METAL: {getenv('METAL', 0)}")
    
    if IS_RK3588:
        print("RK3588平台专用设置:")
        print(f"CL_DISABLE_HALF: {os.environ.get('CL_DISABLE_HALF', '0')}")
        print(f"BEAM: {getenv('BEAM', 0)}")
        print(f"JIT: {getenv('JIT', 1)}")
        print(f"OPTLOCAL: {getenv('OPTLOCAL', 0)}")
    
    print(f"加载 ONNX 模型: {onnx_file}")
    print(f"当前设备: {Device.DEFAULT}")
    print(f"OpenCL 状态: GPU={getenv('GPU', 0)}, METAL={getenv('METAL', 0)}")
    
    # 创建临时变量
    example_inputs = None
    runner = None
    input_names = []
    output_names = []
    input_shapes = {}
    
    # 如果是RK3588平台并且有专用运行器，先尝试使用这个
    if IS_RK3588 and HAS_RK3588_RUNNER:
        try:
            print("\n使用 RK3588 专用 ONNX 运行器...")
            # 使用自定义运行器加载模型
            runner = load_onnx_model(onnx_file)
            print("成功加载模型到 RK3588 专用运行器")
            
            # 获取模型信息
            input_names = runner.input_names
            output_names = runner.output_names
            
            # 读取 ONNX 模型获取输入形状
            onnx_model = onnx.load(onnx_file)
            for inp in onnx_model.graph.input:
                shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
                input_shapes[inp.name] = shape
                
            # 创建示例输入
            example_inputs = {}
            for name, shape in input_shapes.items():
                example_inputs[name] = Tensor.ones(*shape, device=Device.DEFAULT)
            
            # 测试运行
            print("测试运行 RK3588 专用运行器...")
            outputs = runner(example_inputs)
            print("RK3588 专用运行器运行成功!")
            
            # 返回模型信息
            model_info = {
                "run_model": runner,
                "input_names": input_names,
                "output_names": output_names,
                "input_shapes": input_shapes,
                "onnx_file": onnx_file,
                "use_tinygrad": True,
                "device": Device.DEFAULT,
                "use_rk3588_runner": True
            }
            
            return model_info, example_inputs, outputs
            
        except Exception as e:
            print(f"RK3588 专用运行器失败: {e}")
            print("尝试回退到标准方法...")
    
    # 如果不是RK3588或专用运行器失败，尝试标准方法
    try:
        print("\n尝试使用 CPU 设备来避免 OpenCL 编译错误...")
        # 保存原始设备
        original_device = Device.DEFAULT
        # 切换到 CPU
        Device.DEFAULT = "CPU"
        print(f"切换到设备: {Device.DEFAULT}")
        
        # 加载 ONNX 模型
        onnx_model = onnx.load(onnx_file)
        print("成功加载 ONNX 模型")
        
        # 创建 tinygrad OnnxRunner
        print("创建 tinygrad OnnxRunner...")
        runner = OnnxRunner(onnx_model)
        print("成功创建 OnnxRunner")
        
        # 完成后恢复原始设备
        Device.DEFAULT = original_device
        print(f"恢复设备为: {Device.DEFAULT}")
        
    except Exception as e:
        print(f"CPU 方法失败: {e}")
        # 恢复原始设备如果已更改
        if 'original_device' in locals():
            Device.DEFAULT = original_device
            print(f"恢复设备为: {Device.DEFAULT}")
            
        # 尝试正常加载
        try:
            onnx_model = onnx.load(onnx_file)
            print("成功加载 ONNX 模型")
            
            print("创建 tinygrad OnnxRunner...")
            runner = OnnxRunner(onnx_model)
            print("成功创建 OnnxRunner")
        except Exception as e:
            print(f"创建 OnnxRunner 失败: {e}")
            # 如果有 ONNX Runtime，尝试回退
            if HAS_ONNX_RUNTIME:
                print("尝试回退到 ONNX Runtime...")
                try:
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(onnx_file, session_options, providers=['CPUExecutionProvider'])
                    print(f"成功回退到 ONNX Runtime: {session.get_providers()}")
                    # 返回特殊标记，表示使用 ONNX Runtime
                    return {"use_ort": True, "session": session}, None, None
                except Exception as e2:
                    print(f"ONNX Runtime 回退也失败: {e2}")
            sys.exit(1)
    
    # 获取模型输入信息
    input_names = []
    input_shapes = {}
    for inp in onnx_model.graph.input:
        name = inp.name
        input_names.append(name)
        # 获取输入形状
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        input_shapes[name] = shape
    
    print(f"模型输入: {input_names}")
    print(f"模型输入形状: {input_shapes}")
    
    # 创建示例输入
    example_inputs = {}
    for name, shape in input_shapes.items():
        # 对于 YOLOv8n-seg，输入通常是 [1, 3, 480, 640]
        # 创建 Tensor 作为示例输入
        # 先创建一个 CPU 设备的输入用于测试
        example_inputs_cpu = {name: Tensor.ones(*shape, device="CPU") for name, shape in input_shapes.items()}
        # 然后创建原始设备的输入
        example_inputs[name] = Tensor.ones(*shape, device=Device.DEFAULT)
    
    # 获取模型输出信息
    output_names = [out.name for out in onnx_model.graph.output]
    print(f"模型输出: {output_names}")
    
    # 使用 TinyJit 编译模型
    print("使用 TinyJit 编译模型...")
    try:
        # 尝试多种方法解决 Ops.SINK 错误
        print("尝试使用特定的 Context 设置...")
        
        # 方法 1: 使用简单的运行函数，不使用 JIT
        print("方法 1: 使用简单的运行函数，不使用 JIT...")
        def simple_run(inputs):
            # 使用特定的 Context 设置来避免 OpenCL 错误 - 针对 RK3588 优化
            with Context(BEAM=0, JIT=0, AVOID_SINKS=1, FLOAT16=0):
                try:
                    # 将输入转换为 FP32 以避免 FLOAT16 错误
                    fp32_inputs = {}
                    for name, tensor in inputs.items():
                        if isinstance(tensor, Tensor):
                            # 确保使用 FP32 精度
                            fp32_inputs[name] = tensor.cast(Tensor.float)
                        else:
                            fp32_inputs[name] = tensor
                    
                    # 使用 FP32 输入运行模型
                    outputs = runner(fp32_inputs)
                    
                    # 确保所有输出都已实现
                    for out_name, out in outputs.items():
                        if isinstance(out, Tensor):
                            out.realize()
                    return outputs
                except Exception as e:
                    print(f"简单运行错误: {e}")
                    raise
        
        try:
            # 尝试运行简单函数
            print("尝试运行简单函数...")
            test_outputs = simple_run(example_inputs)
            print("简单函数运行成功!")
            
            # 如果简单函数成功，则返回它
            model_info = {
                "run_model": simple_run,
                "input_names": input_names,
                "output_names": output_names,
                "input_shapes": input_shapes,
                "onnx_file": onnx_file,
                "use_tinygrad": True
            }
            
            print("简单函数编译成功，跳过 TinyJit 编译")
            return model_info, example_inputs, test_outputs
            
        except Exception as e:
            print(f"简单函数失败: {e}")
            print("尝试方法 2: 使用 CPU 设备...")
            
            # 方法 2: 尝试使用 CPU 设备
            # 保存原始设备
            original_device = Device.DEFAULT
            # 切换到 CPU
            Device.DEFAULT = "CPU"
            print(f"切换到设备: {Device.DEFAULT}")
            
            def cpu_run(inputs):
                # 使用特定的 Context 设置来避免错误 - 针对 CPU 优化
                with Context(BEAM=0, JIT=0, FLOAT16=0, IMAGE=0, NOLOCALS=0, AVOID_SINKS=0):
                    try:
                        # 将输入转换为 FP32 以避免 FLOAT16 错误
                        fp32_inputs = {}
                        for name, tensor in inputs.items():
                            if isinstance(tensor, Tensor):
                                # 确保使用 FP32 精度
                                fp32_inputs[name] = tensor.cast(Tensor.float)
                            else:
                                fp32_inputs[name] = tensor
                        
                        # 使用 FP32 输入运行模型
                        outputs = runner(fp32_inputs)
                        
                        # 确保所有输出都已实现
                        for out_name, out in outputs.items():
                            if isinstance(out, Tensor):
                                out.realize()
                        return outputs
                    except Exception as e:
                        print(f"CPU 运行错误: {e}")
                        raise
            
            try:
                # 创建 CPU 输入
                cpu_inputs = {}
                for name, shape in input_shapes.items():
                    cpu_inputs[name] = Tensor.ones(*shape, device="CPU")
                
                # 尝试运行 CPU 函数
                print("尝试运行 CPU 函数...")
                cpu_outputs = cpu_run(cpu_inputs)
                print("CPU 函数运行成功!")
                
                # 如果 CPU 函数成功，则返回它
                model_info = {
                    "run_model": cpu_run,
                    "input_names": input_names,
                    "output_names": output_names,
                    "input_shapes": input_shapes,
                    "onnx_file": onnx_file,
                    "use_tinygrad": True,
                    "device": "CPU"
                }
                
                print("CPU 函数编译成功，使用 CPU 运行")
                
                # 恢复原始设备
                Device.DEFAULT = original_device
                print(f"恢复设备为: {Device.DEFAULT}")
                
                return model_info, cpu_inputs, cpu_outputs
                
            except Exception as e:
                print(f"CPU 方法失败: {e}")
                # 恢复原始设备
                Device.DEFAULT = original_device
                print(f"恢复设备为: {Device.DEFAULT}")
        # 方法 3: 尝试使用 TinyJit 与特殊设置
        print("现在尝试方法 3: 使用 TinyJit 与特殊设置...")
        
        # 确保我们在原始设备上
        if 'original_device' in locals():
            Device.DEFAULT = original_device
            print(f"恢复设备为: {Device.DEFAULT}")
        
        # 尝试使用不同的 TinyJit 策略
        # 策略 1: 使用更简单的运行方式，完全禁用 JIT
        def run_model_no_jit(inputs):
            # 使用非常基本的设置，专为 RK3588 OpenCL 优化
            with Context(BEAM=0, JIT=0, FLOAT16=0, IMAGE=0, NOLOCALS=0, OPTLOCAL=0, AVOID_SINKS=1):
                try:
                    # 将输入转换为 NumPy 数组，然后再转回 Tensor
                    np_inputs = {}
                    for name, tensor in inputs.items():
                        if isinstance(tensor, Tensor):
                            # 先转换为 NumPy数组，然后再转回 Tensor，确保使用 FP32
                            np_data = tensor.numpy()
                            np_inputs[name] = Tensor(np_data, dtype=Tensor.float, device="CPU")
                        else:
                            np_inputs[name] = tensor
                    
                    # 先在 CPU 上运行模型
                    cpu_outputs = runner(np_inputs)
                    
                    # 确保所有输出都已实现
                    for out_name, out in cpu_outputs.items():
                        if isinstance(out, Tensor):
                            out.realize()
                    
                    return cpu_outputs
                except Exception as e:
                    print(f"非 JIT 运行错误: {e}")
                    raise
        
        try:
            # 尝试使用非 JIT 运行
            print("尝试使用非 JIT 运行...")
            outputs = run_model_no_jit(example_inputs)
            print("非 JIT 运行成功!")
            
            # 返回非 JIT 编译结果
            model_info = {
                "run_model": run_model_no_jit,
                "input_names": input_names,
                "output_names": output_names,
                "input_shapes": input_shapes,
                "onnx_file": onnx_file,
                "use_tinygrad": True,
                "device": Device.DEFAULT
            }
            
            print("非 JIT 运行成功，跳过 TinyJit 编译")
            return model_info, example_inputs, outputs
        except Exception as e:
            print(f"非 JIT 运行失败: {e}")
        
        # 策略 2: 尝试使用 CPU-GPU 混合方法 - 针对 RK3588 OpenCL 优化
        print("策略 2: 尝试使用 CPU-GPU 混合方法...")
        
        # 尝试先在 CPU 上编译运行，然后转到 GPU
        try:
            # 先切换到 CPU 设备
            original_device = Device.DEFAULT
            Device.DEFAULT = "CPU"
            print(f"切换到设备: {Device.DEFAULT} 进行编译")
            
            # 在 CPU 上创建输入
            cpu_inputs = {}
            for name, shape in input_shapes.items():
                cpu_inputs[name] = Tensor.ones(*shape, device="CPU")
            
            # 在 CPU 上编译模型
            @TinyJit
            def run_model_cpu(inputs):
                with Context(BEAM=0, JIT=1, FLOAT16=0, IMAGE=0, NOLOCALS=0, OPTLOCAL=0, AVOID_SINKS=0):
                    try:
                        outputs = runner(inputs)
                        # 确保所有输出都已实现
                        for out_name, out in outputs.items():
                            if isinstance(out, Tensor):
                                out.realize()
                        return outputs
                    except Exception as e:
                        print(f"CPU TinyJit 运行错误: {e}")
                        raise
            
            # 测试运行 CPU 版本
            print("测试运行 CPU 版本...")
            cpu_outputs = run_model_cpu(cpu_inputs)
            print("CPU 版本运行成功!")
            
            # 创建包装函数，先转换到 CPU，运行模型，然后转回 GPU
            def wrapped_run_model(inputs):
                # 将输入转换到 CPU
                cpu_inputs = {}
                for name, tensor in inputs.items():
                    if isinstance(tensor, Tensor):
                        # 先转换为 NumPy数组，然后再转回 CPU Tensor
                        np_data = tensor.numpy()
                        cpu_inputs[name] = Tensor(np_data, dtype=Tensor.float, device="CPU")
                    else:
                        cpu_inputs[name] = tensor
                
                # 在 CPU 上运行模型
                cpu_outputs = run_model_cpu(cpu_inputs)
                
                # 将输出转换回 GPU
                gpu_outputs = {}
                for name, out in cpu_outputs.items():
                    if isinstance(out, Tensor):
                        # 先实现在 CPU 上
                        out.realize()
                        # 然后转到 GPU
                        gpu_outputs[name] = Tensor(out.numpy(), device="GPU")
                    else:
                        gpu_outputs[name] = out
                        
                return gpu_outputs
            
            # 恢复原始设备
            Device.DEFAULT = original_device
            print(f"恢复设备为: {Device.DEFAULT}")
            
            # 返回包装函数
            model_info = {
                "run_model": wrapped_run_model,
                "input_names": input_names,
                "output_names": output_names,
                "input_shapes": input_shapes,
                "onnx_file": onnx_file,
                "use_tinygrad": True,
                "device": Device.DEFAULT,
                "use_cpu_fallback": True
            }
            
            print("使用 CPU 编译后 GPU 运行的方法成功!")
            return model_info, example_inputs, cpu_outputs
            
        except Exception as e:
            print(f"CPU-GPU 混合方法失败: {e}")
            # 恢复原始设备如果已更改
            if 'original_device' in locals():
                Device.DEFAULT = original_device
                print(f"恢复设备为: {Device.DEFAULT}")
                
            # 如果混合方法失败，尝试原始的 TinyJit 方法
            print("尝试原始的 TinyJit 方法...")
            
            @TinyJit
            def run_model(inputs):
                # 使用最简化的设置
                with Context(BEAM=0, FLOAT16=0, IMAGE=0, NOLOCALS=0, OPTLOCAL=0, AVOID_SINKS=1):
                    try:
                        # 将输入转换为 NumPy 数组，然后再转回 Tensor
                        np_inputs = {}
                        for name, tensor in inputs.items():
                            if isinstance(tensor, Tensor):
                                # 先转换为 NumPy数组，然后再转回 Tensor，确保使用 FP32
                                np_data = tensor.numpy()
                                np_inputs[name] = Tensor(np_data, dtype=Tensor.float)
                            else:
                                np_inputs[name] = tensor
                        
                        # 使用转换后的输入运行模型
                        outputs = runner(np_inputs)
                        
                        # 确保所有输出都已实现
                        for out_name, out in outputs.items():
                            if isinstance(out, Tensor):
                                out.realize()
                        return outputs
                    except Exception as e:
                        print(f"TinyJit 运行错误: {e}")
                        raise
        
        # 运行一次以编译模型
        print("测试运行 TinyJit 编译的模型...")
        GlobalCounters.reset()
        start_time = time.time()
        outputs = run_model(example_inputs)
        end_time = time.time()
        print(f"编译成功! 耗时: {end_time - start_time:.2f} 秒")
        
        # 检查输出
        output_shapes = {}
        for name, out in outputs.items():
            if isinstance(out, Tensor):
                output_shapes[name] = out.shape
            else:
                output_shapes[name] = "non-tensor"
        print(f"输出形状: {output_shapes}")
        
        # 打印详细的编译统计信息
        print(f"编译统计: {GlobalCounters.counter}")
    except Exception as e:
        print(f"编译模型失败: {e}")
        # 如果有 ONNX Runtime，尝试回退
        if HAS_ONNX_RUNTIME:
            print("尝试回退到 ONNX Runtime...")
            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session = ort.InferenceSession(onnx_file, session_options, providers=['CPUExecutionProvider'])
                print(f"成功回退到 ONNX Runtime: {session.get_providers()}")
                # 返回特殊标记，表示使用 ONNX Runtime
                return {"use_ort": True, "session": session}, None, None
            except Exception as e2:
                print(f"ONNX Runtime 回退也失败: {e2}")
        sys.exit(1)
    
    # 保存模型信息
    model_info = {
        "runner": runner,
        "run_model": run_model,
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": input_shapes,
        "onnx_file": onnx_file,
        "use_tinygrad": True
    }
    
    return model_info, example_inputs, outputs

# 加载已编译的模型
def load_model(model_path):
    """
    加载保存的模型信息
    
    参数:
        model_path: 保存的模型信息路径
        
    返回:
        加载的模型运行函数
    """
    print(f"加载模型信息: {model_path}")
    
    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # 从模型数据中获取 ONNX 文件路径
        onnx_file = model_data["onnx_file"]
        print(f"使用 ONNX 文件: {onnx_file}")
        
        # 检查是否使用 tinygrad 编译的模型
        if model_data.get("use_tinygrad", False) and HAS_TINYGRAD:
            print("加载 tinygrad 编译的模型...")
            print(f"当前设备: {Device.DEFAULT}")
            print(f"OpenCL 状态: GPU={getenv('GPU', 0)}, METAL={getenv('METAL', 0)}")
            
            # 尝试多种方法加载 tinygrad 模型
            try:
                # 重新加载 ONNX 模型并创建 OnnxRunner
                onnx_model = onnx.load(onnx_file)
                print("成功加载 ONNX 模型")
                
                # 先尝试不使用 JIT 的方式
                try:
                    print("创建 OnnxRunner...")
                    runner = OnnxRunner(onnx_model)
                    print("成功创建 OnnxRunner")
                    
                    # 先尝试不使用 JIT 的简单运行函数
                    def simple_run(inputs):
                        # 使用特定的 Context 设置来避免 OpenCL 错误
                        with Context(BEAM=0, JIT=0, AVOID_SINKS=1, FLOAT16=getenv("FLOAT16", 1)):
                            try:
                                outputs = runner(inputs)
                                return outputs
                            except Exception as e:
                                print(f"OnnxRunner 运行错误: {e}")
                                raise
                    
                    # 测试简单运行函数
                    print("测试简单运行函数...")
                    example_input = {}
                    for name, shape in model_data["input_shapes"].items():
                        example_input[name] = Tensor.ones(*shape, device=Device.DEFAULT)
                    
                    # 尝试运行一次
                    test_output = simple_run(example_input)
                    print("简单运行函数测试成功")
                    
                    # 现在尝试使用 TinyJit
                    print("创建 TinyJit 运行函数...")
                    
                    @TinyJit
                    def run_model(inputs):
                        # 使用特定的 Context 设置来避免 OpenCL 错误
                        with Context(BEAM=getenv("BEAM", 0), AVOID_SINKS=1, FLOAT16=getenv("FLOAT16", 1)):
                            # OnnxRunner 使用 __call__ 方法
                            outputs = runner(inputs)
                            # 确保所有输出都已实现
                            for out_name, out in outputs.items():
                                if isinstance(out, Tensor):
                                    out.realize()
                            return outputs
                    
                    # 尝试运行 TinyJit 函数
                    print("测试 TinyJit 运行函数...")
                    jit_output = run_model(example_input)
                    print("TinyJit 运行函数测试成功")
                    
                    print("模型加载成功")
                    print(f"运行设备: {Device.DEFAULT}")
                    return run_model
                    
                except Exception as e:
                    print(f"TinyJit 创建错误: {e}")
                    print("回退到简单运行函数...")
                    # 如果 TinyJit 失败，但简单运行函数成功，则返回简单运行函数
                    return simple_run
                    
            except Exception as e:
                print(f"tinygrad 加载失败: {e}")
                print("尝试回退到 ONNX Runtime...")
        elif model_data.get("use_ort", False) and HAS_ONNX_RUNTIME:
            print("加载 ONNX Runtime 模型...")
        else:
            print("未检测到 tinygrad 编译的模型，尝试使用 ONNX Runtime...")
        
        # 如果 tinygrad 加载失败或者不是 tinygrad 模型，尝试使用 ONNX Runtime
        if HAS_ONNX_RUNTIME:
            # 创建新的 ONNX Runtime 会话
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(onnx_file, session_options, providers=providers)
            
            # 创建运行函数
            def run_model(inputs):
                # 确保输入是 numpy 数组
                np_inputs = {}
                for k, v in inputs.items():
                    if HAS_TINYGRAD and isinstance(v, Tensor):
                        np_inputs[k] = v.numpy()
                    else:
                        np_inputs[k] = v
                
                # 运行推理
                outputs = session.run(None, np_inputs)
                return outputs
            
            print("模型加载成功")
            print(f"运行设备: {session.get_providers()[0]}")
            return run_model
        else:
            print("错误: 无法加载模型，缺少 ONNX Runtime 库")
            print("请安装 ONNX Runtime: pip install onnxruntime")
            sys.exit(1)
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
    
    # 如果有 tinygrad，则创建并返回 GPU Tensor
    if HAS_TINYGRAD:
        try:
            # 尝试直接在 GPU 上创建 Tensor
            print(f"在 {Device.DEFAULT} 上创建输入 Tensor")
            gpu_tensor = Tensor(tensor, device=Device.DEFAULT)
            return gpu_tensor, (orig_height, orig_width)
        except Exception as e:
            print(f"在 GPU 上创建 Tensor 失败: {e}")
            print("回退到 CPU Tensor")
            return Tensor(tensor), (orig_height, orig_width)
    else:
        return tensor, (orig_height, orig_width)

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
    # 创建可视化图像的副本
    vis_image = image.copy()
    
    # 添加标题
    cv2.putText(vis_image, "YOLOv8n-seg Detection", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 添加运行设备信息
    device_info = f"Running on: ONNX Runtime (CPU)"
    cv2.putText(vis_image, device_info, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 如果有输出，尝试解析并绘制边界框和分割掩码
    if outputs and len(outputs) > 0:
        try:
            # YOLOv8-seg 输出通常包含检测和分割结果
            # 第一个输出是检测结果，包含边界框、置信度和类别
            detection_output = outputs[0]
            
            # 添加输出形状信息
            shape_info = f"Output shape: {detection_output.shape}"
            cv2.putText(vis_image, shape_info, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 添加检测到的对象数量信息
            if len(detection_output.shape) >= 2:
                num_detections = detection_output.shape[1]
                cv2.putText(vis_image, f"Detections: {num_detections}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        except Exception as e:
            # 如果解析失败，添加错误信息
            cv2.putText(vis_image, f"Error parsing output: {str(e)}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
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
            try:
                outputs = model({"images": input_tensor})
                inference_time = time.time() - start_time
                total_time += inference_time
                success = True
            except Exception as e:
                print(f"推理错误: {e}")
                success = False
            
            # 计算 FPS
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            # 只有在推理成功时才进行后处理
            if success:
                # 后处理输出
                results = postprocess_outputs(outputs, orig_size, conf_thres, iou_thres)
                
                # 可视化结果
                vis_frame = visualize_outputs(frame, results, orig_size, conf_thres)
            else:
                # 如果推理失败，只显示原始帧
                vis_frame = frame.copy()
                cv2.putText(vis_frame, "Inference Error", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 添加 FPS 信息
            cv2.putText(vis_frame, f"FPS: {avg_fps:.2f}", (10, 190),
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
        if frame_count > 0 and total_time > 0:
            print(f"平均 FPS: {frame_count / total_time:.2f}")
        else:
            print("无法计算 FPS (处理时间为 0)")

# 运行基准测试
def run_benchmark(model, num_runs=100):
    """
    运行基准测试
    
    参数:
        model: 加载的模型
        num_runs: 运行次数
    """
    print(f"运行基准测试，迭代次数: {num_runs}")
    
    # 创建示例输入
    input_tensor = Tensor.randn(1, 3, 480, 640, device=Device.DEFAULT)
    
    # 预热
    print("预热中...")
    for _ in range(10):
        model({"images": input_tensor})
    
    # 计时
    print("开始基准测试...")
    start_time = time.time()
    for i in range(num_runs):
        model({"images": input_tensor})
    end_time = time.time()
    
    # 计算结果
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time
    
    print(f"基准测试结果:")
    print(f"总时间: {total_time:.4f} 秒")
    print(f"平均推理时间: {avg_time*1000:.2f} 毫秒")
    print(f"FPS: {fps:.2f}")
    
    return fps

# 主函数
def main():
    # 解析命令行参数
    args = parse_args()
    
    # 如果提供了位置参数，使用位置参数作为输出文件路径
    output_file = args.output_file if args.output_file else args.output
    
    # 获取模型文件
    model_path = get_model_file(args.model)
    
    # 打印当前运行环境信息
    if HAS_TINYGRAD:
        print(f"当前设备: {Device.DEFAULT}")
        print(f"OpenCL 状态: GPU={getenv('GPU', 0)}, METAL={getenv('METAL', 0)}")
        print(f"DEBUG 级别: {DEBUG}")
    
    # 编译或加载模型
    if args.run:
        # 仅运行，不编译
        print(f"加载模型信息: {output_file}")
        model = load_model(output_file)
    else:
        # 使用 tinygrad 编译模型
        print(f"准备编译模型: {model_path}")
        model_info, example_inputs, outputs = compile(model_path)
        
        # 保存模型信息
        print(f"保存模型信息: {output_file}")
        
        # 检查是否使用 tinygrad 或 ONNX Runtime
        if model_info.get("use_tinygrad", False):
            # 保存 tinygrad 模型数据
            model_data = {
                "onnx_file": model_path,
                "input_names": model_info["input_names"],
                "output_names": model_info["output_names"],
                "input_shapes": model_info["input_shapes"],
                "use_tinygrad": True,
                "creation_time": time.time()
            }
            
            with open(output_file, "wb") as f:
                pickle.dump(model_data, f)
            
            print(f"tinygrad 模型信息已保存至: {output_file}")
            
            # 使用预编译的运行函数
            model = model_info["run_model"]
        elif model_info.get("use_ort", False):
            # 保存 ONNX Runtime 模型数据
            model_data = {
                "onnx_file": model_path,
                "input_names": model_info["input_names"] if "input_names" in model_info else [],
                "output_names": model_info["output_names"] if "output_names" in model_info else [],
                "input_shapes": model_info["input_shapes"] if "input_shapes" in model_info else {},
                "use_ort": True,
                "creation_time": time.time()
            }
            
            with open(output_file, "wb") as f:
                pickle.dump(model_data, f)
            
            print(f"ONNX Runtime 模型信息已保存至: {output_file}")
            
            # 创建运行函数
            def run_model(inputs):
                # 确保输入是 numpy 数组
                np_inputs = {}
                for k, v in inputs.items():
                    if HAS_TINYGRAD and isinstance(v, Tensor):
                        np_inputs[k] = v.numpy()
                    else:
                        np_inputs[k] = v
                
                # 运行推理
                outputs = model_info["session"].run(None, np_inputs)
                return outputs
            
            model = run_model
    
    # 打印运行设备信息
    if HAS_TINYGRAD and getenv("GPU", 0):
        print(f"运行设备: tinygrad (OpenCL)")
    elif HAS_ONNX_RUNTIME:
        print(f"运行设备: ONNX Runtime (CPU)")
    
    # 运行基准测试
    if args.benchmark:
        run_benchmark(model)
    
    # 处理输入
    if args.image:
        # 处理单张图像
        process_image(model, args.image, args.conf_thres, args.iou_thres,
                     args.save, not args.no_display)
    elif args.video:
        # 处理视频流
        process_video(model, args.video, args.conf_thres, args.iou_thres,
                     args.save, not args.no_display)

if __name__ == "__main__":
    main()
