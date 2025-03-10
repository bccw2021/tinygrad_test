import os, sys, pickle, time
import numpy as np
import argparse

# 设置环境变量
if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

# 明确禁用Metal后端，强制使用OpenCL
os.environ["METAL"] = "0"
os.environ["OPENCL"] = "1"

# 尝试导入OpenCV，用于视频捕获
try:
    import cv2
    HAVE_OPENCV = True
except ImportError:
    print("警告: 未找到OpenCV。要使用实时输入流，请安装OpenCV: pip install opencv-python")
    HAVE_OPENCV = False

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import _from_np_dtype
from tinygrad.engine.realize import CompiledRunner
from tinygrad.ops import PatternMatcher, UPat, Ops

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import OnnxRunner   # TODO: port to main tinygrad


"""
编译模型并使用视频文件进行推理：
python -m examples.openpilot.compile3 --video path/to/video.mp4

使用摄像头进行实时推理：
python -m examples.openpilot.compile3 --video 摄像头 --camera 0

仅编译模型，不运行推理：
python -m examples.openpilot.compile3 --compile-only

使用已编译的模型进行推理（跳过编译步骤）：
# 确保模型已经编译过
python -m examples.openpilot.compile3 --video path/to/video.mp4
"""



# 强制使用GPU（OpenCL）
Device.DEFAULT = "GPU"

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='SuperCombo模型编译和实时推理')
    parser.add_argument('--model', type=str, default="https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx",
                        help='ONNX模型文件路径或URL')
    parser.add_argument('--output', type=str, default="/tmp/openpilot.pkl",
                        help='输出编译后的模型文件路径')
    parser.add_argument('--video', type=str, default=None,
                        help='输入视频文件路径，如果不提供则使用随机生成的数据')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头索引，当--video为“摄像头”时使用')
    parser.add_argument('--compile-only', action='store_true',
                        help='仅编译模型，不运行实时推理')
    return parser.parse_args()

# 全局参数
args = parse_args()
OPENPILOT_MODEL = args.model
OUTPUT = args.output


# 检查和修复OpenCLRenderer的code_for_workitem字典
def debug_and_fix_opencl_renderer():
  from tinygrad.renderer.cstyle import OpenCLRenderer
  import sys
  
  # 打印调试信息
  print("调试信息：")
  print(f"OpenCLRenderer.code_for_workitem = {OpenCLRenderer.code_for_workitem}")
  print(f"'i' in OpenCLRenderer.code_for_workitem: {'i' in OpenCLRenderer.code_for_workitem}")
  
  # 确保'i'键存在于OpenCLRenderer的code_for_workitem字典中
  if 'i' not in OpenCLRenderer.code_for_workitem:
    print("添加缺失的'i'键到OpenCLRenderer.code_for_workitem")
    OpenCLRenderer.code_for_workitem['i'] = lambda x: f"get_global_id({x})"

# 处理视频输入并准备模型输入数据
def process_video_frame(frame, input_shapes, input_types):
  """
  处理视频帧并准备SuperCombo模型的输入数据
  
  参数:
      frame: 原始视频帧（OpenCV格式，BGR）
      input_shapes: 模型输入形状字典
      input_types: 模型输入类型字典
  
  返回:
      inputs: 包含所有模型输入的字典
  """
  if not HAVE_OPENCV:
    raise ImportError("需要OpenCV处理视频输入。请安装: pip install opencv-python")
  
  # 初始化输入字典
  inputs = {}
  
  # 处理图像输入 - input_imgs 和 big_input_imgs
  if 'input_imgs' in input_shapes:
    # 获取目标尺寸
    _, channels, height, width = input_shapes['input_imgs']
    
    # 根据SuperCombo文档，模型期望的是YUV420格式的输入
    # 输入图像应该是256x512，但在YUV420格式中存储为6个通道的128x256
    # 我们需要将BGR图像转换为YUV420格式，并按照特定方式处理
    
    # 首先将图像调整为256x512
    model_height, model_width = 256, 512
    resized_frame = cv2.resize(frame, (model_width, model_height))
    
    # 将BGR转换为YUV
    yuv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2YUV)
    
    # 创建输入张量，形状为(batch_size, channels, height, width)
    # SuperCombo期望两个连续帧，每帧6个通道
    input_img = np.zeros((1, channels, height, width), dtype=np.float32)
    
    # 提取Y通道并进行子采样
    y_channel = yuv_frame[:, :, 0]
    
    # 提取U和V通道
    u_channel = yuv_frame[:, :, 1]
    v_channel = yuv_frame[:, :, 2]
    
    # 对U和V通道进行下采样（在YUV420中，U和V通道的分辨率是Y通道的一半）
    u_downsampled = cv2.resize(u_channel, (model_width//2, model_height//2))
    v_downsampled = cv2.resize(v_channel, (model_width//2, model_height//2))
    
    # 对于两帧输入（当前只有一帧，所以复制它）
    for frame_idx in range(2):  # SuperCombo需要2个连续帧
      base_idx = frame_idx * 6  # 每帧6个通道
      
      # Y通道的四个子采样 - 按照文档中描述的方式
      # Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], Y[1::2, 1::2]
      input_img[0, base_idx + 0] = cv2.resize(y_channel[::2, ::2], (width, height)) / 127.5 - 1.0
      input_img[0, base_idx + 1] = cv2.resize(y_channel[::2, 1::2], (width, height)) / 127.5 - 1.0
      input_img[0, base_idx + 2] = cv2.resize(y_channel[1::2, ::2], (width, height)) / 127.5 - 1.0
      input_img[0, base_idx + 3] = cv2.resize(y_channel[1::2, 1::2], (width, height)) / 127.5 - 1.0
      
      # U和V通道（已经是半分辨率）
      input_img[0, base_idx + 4] = cv2.resize(u_downsampled, (width, height)) / 127.5 - 1.0
      input_img[0, base_idx + 5] = cv2.resize(v_downsampled, (width, height)) / 127.5 - 1.0
    
    # 创建tinygrad张量并明确指定为GPU设备
    # 确保所有张量都在GPU上，并立即realize()以确保分配到GPU
    inputs['input_imgs'] = Tensor(input_img, device="GPU").realize()
    
    # 如果模型需要big_input_imgs，使用相同的处理方式
    if 'big_input_imgs' in input_shapes:
      inputs['big_input_imgs'] = Tensor(input_img, device="GPU").realize()
  
  # 处理其他必要的输入 - 确保所有张量都在GPU上
  if 'desire' in input_shapes and 'desire' not in inputs:
    # desire: 100x8的one-hot编码缓冲区，表示过去5秒的命令（20FPS下的100帧）
    desire_shape = input_shapes['desire']
    # 明确指定device="GPU"并立即realize()
    inputs['desire'] = Tensor(np.zeros(desire_shape, dtype=np.float32), device="GPU").realize()
  
  if 'traffic_convention' in input_shapes and 'traffic_convention' not in inputs:
    # traffic_convention: 2元素的one-hot向量，表示交通是右侧还是左侧
    tc_shape = input_shapes['traffic_convention']
    tc = np.zeros(tc_shape, dtype=np.float32)
    tc[0, 0] = 1.0  # 默认右侧驾驶 [1,0]，左侧驾驶为[0,1]
    # 明确指定device="GPU"并立即realize()
    inputs['traffic_convention'] = Tensor(tc, device="GPU").realize()
  
  if 'lateral_control_params' in input_shapes and 'lateral_control_params' not in inputs:
    # lateral_control_params: 横向控制参数
    lcp_shape = input_shapes['lateral_control_params']
    lcp = np.zeros(lcp_shape, dtype=np.float32)
    lcp[0, 0] = 0.0  # 车速 (m/s)
    lcp[0, 1] = 0.2  # 转向延迟 (s)
    # 明确指定device="GPU"并立即realize()
    inputs['lateral_control_params'] = Tensor(lcp, device="GPU").realize()
  
  if 'prev_desired_curv' in input_shapes and 'prev_desired_curv' not in inputs:
    # prev_desired_curv: 前一个期望曲率
    pdc_shape = input_shapes['prev_desired_curv']
    # 明确指定device="GPU"并立即realize()
    inputs['prev_desired_curv'] = Tensor(np.zeros(pdc_shape, dtype=np.float32), device="GPU").realize()
  
  if 'features_buffer' in input_shapes and 'features_buffer' not in inputs:
    # features_buffer: 99x512的特征缓冲区，用于提供5秒的时间上下文
    fb_shape = input_shapes['features_buffer']
    # 明确指定device="GPU"并立即realize()
    inputs['features_buffer'] = Tensor(np.zeros(fb_shape, dtype=np.float32), device="GPU").realize()
  
  # 确保所有必需的输入都已创建，并且都在GPU上
  for k, shp in sorted(input_shapes.items()):
    if k not in inputs:
      print(f"警告: 创建默认输入 '{k}' 形状为 {shp}")
      # 明确指定device="GPU"并立即realize()
      inputs[k] = Tensor(np.zeros(shp, dtype=_from_np_dtype(input_types[k])), device="GPU").realize()
  
  # 最后再次确保所有输入都在GPU上
  for k in inputs:
    if inputs[k].device != "GPU":
      print(f"警告: 将输入 '{k}' 从 {inputs[k].device} 移动到 GPU")
      inputs[k] = inputs[k].to("GPU").realize()
  
  return inputs

# 解析和打印SuperCombo模型的输出
def parse_supercombo_output(output_tensor):
  """
  解析SuperCombo模型的输出，并打印车道线检测、车辆检测、路径规划和驾驶状态信息
  
  SuperCombo模型输出是一个形状为(1, 6504)的张量，包含多种信息
  根据OpenPilot项目的结构，我们可以大致将输出分为以下几部分：
  - 车道线检测结果 (约前2000个元素)
  - 车辆检测结果 (约2000-4000元素)
  - 路径规划信息 (约4000-5500元素)
  - 驾驶状态信息 (约5500-6504元素)
  
  参数:
      output_tensor: 模型输出的NumPy数组，形状为(1, 6504)
  """
  # 确保输出是一维数组
  if output_tensor.ndim > 1:
    output_tensor = output_tensor.flatten()
  
  # 定义各部分的索引范围（这些是估计值，实际范围可能需要根据OpenPilot文档调整）
  lane_detection_range = (0, 2000)
  vehicle_detection_range = (2000, 4000)
  path_planning_range = (4000, 5500)
  driving_state_range = (5500, min(6504, len(output_tensor)))
  
  # 提取各部分数据
  lane_detection = output_tensor[lane_detection_range[0]:lane_detection_range[1]]
  vehicle_detection = output_tensor[vehicle_detection_range[0]:vehicle_detection_range[1]]
  path_planning = output_tensor[path_planning_range[0]:path_planning_range[1]]
  driving_state = output_tensor[driving_state_range[0]:driving_state_range[1]]
  
  # 打印车道线检测结果
  print("\n===== 车道线检测结果 =====")
  print(f"数据范围: 索引 {lane_detection_range[0]}-{lane_detection_range[1]}")
  print(f"数据形状: {lane_detection.shape}")
  print(f"数据统计: 最小值={lane_detection.min():.4f}, 最大值={lane_detection.max():.4f}, 均值={lane_detection.mean():.4f}")
  print("前10个值:")
  for i in range(min(10, len(lane_detection))):
    print(f"  [{i}]: {lane_detection[i]:.4f}")
  
  # 打印车辆检测结果
  print("\n===== 车辆检测结果 =====")
  print(f"数据范围: 索引 {vehicle_detection_range[0]}-{vehicle_detection_range[1]}")
  print(f"数据形状: {vehicle_detection.shape}")
  print(f"数据统计: 最小值={vehicle_detection.min():.4f}, 最大值={vehicle_detection.max():.4f}, 均值={vehicle_detection.mean():.4f}")
  print("前10个值:")
  for i in range(min(10, len(vehicle_detection))):
    print(f"  [{i+vehicle_detection_range[0]}]: {vehicle_detection[i]:.4f}")
  
  # 打印路径规划信息
  print("\n===== 路径规划信息 =====")
  print(f"数据范围: 索引 {path_planning_range[0]}-{path_planning_range[1]}")
  print(f"数据形状: {path_planning.shape}")
  print(f"数据统计: 最小值={path_planning.min():.4f}, 最大值={path_planning.max():.4f}, 均值={path_planning.mean():.4f}")
  print("前10个值:")
  for i in range(min(10, len(path_planning))):
    print(f"  [{i+path_planning_range[0]}]: {path_planning[i]:.4f}")
  
  # 打印驾驶状态信息
  print("\n===== 驾驶状态信息 =====")
  print(f"数据范围: 索引 {driving_state_range[0]}-{driving_state_range[1]}")
  print(f"数据形状: {driving_state.shape}")
  print(f"数据统计: 最小值={driving_state.min():.4f}, 最大值={driving_state.max():.4f}, 均值={driving_state.mean():.4f}")
  print("前10个值:")
  for i in range(min(10, len(driving_state))):
    print(f"  [{i+driving_state_range[0]}]: {driving_state[i]:.4f}")

def compile(onnx_file):
  # 调试并修复OpenCLRenderer，确保'i'键存在于code_for_workitem字典中
  debug_and_fix_opencl_renderer()
  
  # 加载ONNX模型文件
  onnx_model = onnx.load(onnx_file)
  # 创建ONNX运行器实例
  run_onnx = OnnxRunner(onnx_model)
  print("loaded model")

  # 提取模型输入的形状信息
  # SuperCombo模型的输入包括：
  # 1. 'input_imgs'：图像输入，形状通常为(1, 12, 128, 256)，表示批次大小、通道数、高度和宽度
  # 2. 'desire'：驾驶意图，形状为(1, 8)，表示不同驾驶意图的概率
  # 3. 'traffic_convention'：交通规则，形状为(1, 2)，表示左/右侧驾驶等信息
  # 4. 'initial_state'：初始状态，形状为(1, 512)，表示模型的内部状态
  # 等其他输入...
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  
  # 提取模型输入的数据类型
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  # 确保浮点输入和输出都是float32类型（OpenPilot要求）
  input_types = {k:(np.float32 if v==np.float16 else v) for k,v in input_types.items()}
  
  # 设置随机种子以确保结果可重现
  Tensor.manual_seed(100)
  
  # 为每个输入创建随机张量，并乘以8以增加数值范围
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
  
  # 将张量转换为NumPy数组，用于后续处理
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
  print("created tensors")

  # 创建TinyJit对象来优化模型执行
  # 这个函数将输入张量转换到指定设备，执行模型，并将结果转换为float32类型
  run_onnx_jit = TinyJit(lambda **kwargs:
                         next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), prune=True)
  
  # 运行模型三次：第一次初始化，第二次JIT编译，第三次调试输出
  for i in range(3):
    # 重置计数器
    GlobalCounters.reset()
    print(f"run {i}")
    
    # 准备输入数据：确保所有输入张量都在GPU上
    inputs = {k:v.clone().to("GPU").realize() for k,v in new_inputs.items()}
    
    # 设置调试级别并执行模型
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      # 执行模型并获取NumPy格式的输出
      # SuperCombo模型的输出是一个形状为(1, 6504)的张量，包含以下信息：
      # - 车道线检测结果：包括车道线的位置、曲率、宽度等
      # - 车辆检测结果：周围车辆的位置、速度、加速度等
      # - 路径规划信息：建议的驾驶路径、转向角等
      # - 驾驶状态信息：当前车速、加速度等
      ret = run_onnx_jit(**inputs).numpy()
    # 使用第二次运行的结果作为测试基准
    
    # 打印模型输出信息
    if i == 2:  # 在最后一次运行后打印详细信息
      print("\n模型输出信息:")
      print(f"输出形状: {ret.shape}")
      print(f"输出类型: {ret.dtype}")
      print(f"输出范围: 最小值={ret.min()}, 最大值={ret.max()}")
      print(f"输出均值: {ret.mean()}")
      print(f"输出标准差: {ret.std()}")
      
      # 打印输出的前几个值
      print("\n输出的前10个值:")
      flat_output = ret.flatten()
      for j in range(min(10, len(flat_output))):
        print(f"  [{j}]: {flat_output[j]}")
      
      # 解析和打印车道线检测、车辆检测、路径规划和驾驶状态信息
      print("\n解析SuperCombo模型输出的各部分数据:")
      parse_supercombo_output(ret)
    # 如果是第二次运行，保存结果作为测试基准
    if i == 1: test_val = np.copy(ret)
    
  # 输出捕获的内核数量
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  
  # 验证JIT运行结果与测试基准一致
  np.testing.assert_equal(test_val, ret, "JIT run failed")
  print("jit run validated")

  # 从编译结果中收集统计信息
  kernel_count = 0          # 内核数量
  read_image_count = 0      # 读取图像的次数
  gated_read_image_count = 0 # 条件读取图像的次数
  
  # 遍历所有缓存的JIT内核
  for ei in run_onnx_jit.captured.jit_cache:
    if isinstance(ei.prg, CompiledRunner):
      kernel_count += 1
      # 统计代码中读取图像的次数
      read_image_count += ei.prg.p.src.count("read_image")
      gated_read_image_count += ei.prg.p.src.count("?read_image")
  
  # 输出统计信息
  print(f"{kernel_count=},  {read_image_count=}, {gated_read_image_count=}")
  
  # 根据环境变量检查内核数量是否超过限制
  if (allowed_kernel_count:=getenv("ALLOWED_KERNEL_COUNT", -1)) != -1:
    assert kernel_count <= allowed_kernel_count, f"too many kernels! {kernel_count=}, {allowed_kernel_count=}"
  
  # 检查读取图像的次数是否符合预期
  if (allowed_read_image:=getenv("ALLOWED_READ_IMAGE", -1)) != -1:
    assert read_image_count == allowed_read_image, f"different read_image! {read_image_count=}, {allowed_read_image=}"
  
  # 检查条件读取图像的次数是否超过限制
  if (allowed_gated_read_image:=getenv("ALLOWED_GATED_READ_IMAGE", -1)) != -1:
    assert gated_read_image_count <= allowed_gated_read_image, f"too many gated read_image! {gated_read_image_count=}, {allowed_gated_read_image=}"

  # 将编译后的模型保存为pickle文件
  with open(OUTPUT, "wb") as f:
    pickle.dump(run_onnx_jit, f)
  
  # 计算并输出模型大小信息
  mdl_sz = os.path.getsize(onnx_file)  # 原始ONNX模型大小
  pkl_sz = os.path.getsize(OUTPUT)     # 序列化后的模型大小
  print(f"mdl size is {mdl_sz/1e6:.2f}M")  # 输出原始模型大小（单位：MB）
  print(f"pkl size is {pkl_sz/1e6:.2f}M") # 输出序列化后的模型大小（单位：MB）
  print("**** compile done ****")
  
  # 返回测试基准值供后续验证使用
  return test_val

def test_vs_compile(run, new_inputs, test_val=None):
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}

  # create fake "from_blob" tensors for the inputs, and wrapped NPY tensors for the numpy inputs (these have the same underlying memory)
  inputs = {**{k:v for k,v in new_inputs.items() if 'img' in k},
            **{k:Tensor(v, device="NPY").realize() for k,v in new_inputs_numpy.items() if 'img' not in k}}

  # run 20 times
  for _ in range(20):
    st = time.perf_counter()
    out = run(**inputs)
    mt = time.perf_counter()
    val = out.numpy()
    et = time.perf_counter()
    print(f"enqueue {(mt-st)*1e3:6.2f} ms -- total run {(et-st)*1e3:6.2f} ms")
  print(out, val.shape, val.dtype)
  if test_val is not None: np.testing.assert_equal(test_val, val)
  print("**** test done ****")

  # test that changing the numpy changes the model outputs
  if any([x.device == 'NPY' for x in inputs.values()]):
    for v in new_inputs_numpy.values(): v *= 2
    out = run(**inputs)
    changed_val = out.numpy()
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, val, changed_val)
  return val

def test_vs_onnx(new_inputs, test_val, onnx_file, ort=False):
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
  onnx_model = onnx.load(onnx_file)

  timings = []
  if ort:
    # test with onnxruntime
    import onnxruntime as ort
    onnx_session = ort.InferenceSession(onnx_file)
    for _ in range(1 if test_val is not None else 5):
      st = time.perf_counter()
      onnx_output = onnx_session.run([onnx_model.graph.output[0].name], {k:v.astype(np.float16) for k,v in new_inputs_numpy.items()})
      timings.append(time.perf_counter() - st)
    new_torch_out = onnx_output[0]
  else:
    # test with torch
    import torch
    from onnx2torch import convert
    inputs = {k.name:new_inputs_numpy[k.name] for k in onnx_model.graph.input}
    torch_model = convert(onnx_model).float()
    with torch.no_grad():
      for _ in range(1 if test_val is not None else 5):
        st = time.perf_counter()
        torch_out = torch_model(*[torch.tensor(x) for x in inputs.values()])
        timings.append(time.perf_counter() - st)
      new_torch_out = torch_out.numpy()

  if test_val is not None:
    np.testing.assert_allclose(new_torch_out.reshape(test_val.shape), test_val, atol=1e-4, rtol=1e-2)
    print("test vs onnx passed")
  return timings

# 实时推理函数
def run_realtime_inference(model_file, video_source):
  """
  使用编译好的模型进行实时推理
  
  参数:
      model_file: 编译后的模型文件路径
      video_source: 视频源（可以是视频文件路径或摄像头索引）
  """
  if not HAVE_OPENCV:
    print("错误: 需要OpenCV进行实时推理。请安装: pip install opencv-python")
    return
  
  print(f"加载编译后的模型: {model_file}")
  with open(model_file, "rb") as f:
    run = pickle.load(f)
  
  # 加载原始ONNX模型以获取输入形状信息
  onnx_file = fetch(OPENPILOT_MODEL)
  onnx_model = onnx.load(onnx_file)
  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  input_types = {k:(np.float32 if v==np.float16 else v) for k,v in input_types.items()}
  
  # 打印模型输入信息
  print("模型输入信息:")
  for name, shape in input_shapes.items():
    print(f"  {name}: {shape} ({input_types[name]})")
  
  # 创建窗口
  cv2.namedWindow('SuperCombo 实时推理', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('SuperCombo 实时推理', 800, 600)
  
  # 初始化性能计数器
  frame_count = 0
  start_time = time.time()
  processing_times = []
  
  # 上一帧的模型状态（用于GRU/LSTM等循环网络）
  prev_state = None
  
  # 确定视频源
  use_test_frame = False
  if video_source == "摄像头":
    try:
      cap = cv2.VideoCapture(args.camera)
      print(f"打开摄像头 #{args.camera}")
      if not cap.isOpened():
        print("警告: 无法打开摄像头，将使用生成的测试帧")
        use_test_frame = True
    except Exception as e:
      print(f"打开摄像头时出错: {e}")
      print("将使用生成的测试帧")
      use_test_frame = True
  elif video_source == "测试":
    print("使用生成的测试帧进行推理")
    use_test_frame = True
  else:
    try:
      cap = cv2.VideoCapture(video_source)
      print(f"打开视频文件: {video_source}")
      if not cap.isOpened():
        print("警告: 无法打开视频文件，将使用生成的测试帧")
        use_test_frame = True
    except Exception as e:
      print(f"打开视频文件时出错: {e}")
      print("将使用生成的测试帧")
      use_test_frame = True
  
  # 如果需要使用测试帧，创建一个彩色渐变测试图像
  if use_test_frame:
    # 获取模型期望的输入尺寸
    if 'input_imgs' in input_shapes:
      _, channels, height, width = input_shapes['input_imgs']
    else:
      # 默认尺寸
      height, width, channels = 128, 256, 3
    
    # 创建一个测试帧 - 彩色渐变
    def create_test_frame(frame_num):
      # 创建一个随时间变化的彩色渐变图像
      test_frame = np.zeros((height, width, 3), dtype=np.uint8)
      
      # 水平渐变 - 红色
      for x in range(width):
        red_value = int(255 * x / width)
        test_frame[:, x, 0] = red_value
      
      # 垂直渐变 - 绿色
      for y in range(height):
        green_value = int(255 * y / height)
        test_frame[y, :, 1] = green_value
      
      # 时间渐变 - 蓝色（随帧数变化）
      blue_value = int(127 + 127 * np.sin(frame_num / 10))
      test_frame[:, :, 2] = blue_value
      
      # 添加一些移动的圆形
      circle_x = int(width/2 + width/4 * np.sin(frame_num / 20))
      circle_y = int(height/2 + height/4 * np.cos(frame_num / 15))
      cv2.circle(test_frame, (circle_x, circle_y), 20, (255, 255, 255), -1)
      
      # 添加帧计数文本
      cv2.putText(test_frame, f"Frame: {frame_num}", (10, 20), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      
      return test_frame
  else:
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频尺寸: {frame_width}x{frame_height}, FPS: {fps}")
  
  try:
    while True:
      if use_test_frame:
        # 使用生成的测试帧
        frame = create_test_frame(frame_count)
        ret = True
      else:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
          # 如果是视频文件，可以循环播放
          if video_source != "摄像头":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
          else:
            break
      
      # 处理帧并准备模型输入
      frame_start = time.time()
      inputs = process_video_frame(frame, input_shapes, input_types)
      
      # 如果有上一帧的状态，可以将其作为initial_state输入
      if prev_state is not None and 'initial_state' in inputs:
        inputs['initial_state'] = prev_state
      
      try:
        # 运行模型推理
        with Context(DEBUG=1):
          output = run(**inputs)
          output_np = output.numpy()
        
        # 保存当前状态用于下一帧（如果模型有状态输出）
        # 注意：这里需要根据实际模型输出调整
        # prev_state = output_np[...]
        
        # 计算处理时间
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)
        
        # 在帧上显示信息
        fps_text = f"FPS: {1.0/frame_time:.1f}"
        cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 每10帧打印一次详细输出
        if frame_count % 10 == 0:
          print(f"\n帧 #{frame_count}")
          print(f"处理时间: {frame_time*1000:.1f} ms (FPS: {1.0/frame_time:.1f})")
          print(f"输出形状: {output_np.shape}")
          
          # 可选：解析并打印模型输出
          # parse_supercombo_output(output_np)
      except Exception as e:
        print(f"推理过程中出错: {e}")
        # 在帧上显示错误信息
        cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
      
      # 显示帧
      cv2.imshow('SuperCombo 实时推理', frame)
      
      # 按ESC键退出
      if cv2.waitKey(1) == 27:
        break
      
      # 限制帧率（测试帧模式）
      if use_test_frame:
        time.sleep(0.033)  # 约30fps
      
      frame_count += 1
  
  finally:
    # 释放资源
    if not use_test_frame:
      cap.release()
    cv2.destroyAllWindows()
    
    # 打印性能统计
    if processing_times:
      avg_time = sum(processing_times) / len(processing_times)
      avg_fps = 1.0 / avg_time
      print(f"\n性能统计:")
      print(f"处理帧数: {frame_count}")
      print(f"平均处理时间: {avg_time*1000:.1f} ms")
      print(f"平均FPS: {avg_fps:.1f}")
      print(f"最快帧: {min(processing_times)*1000:.1f} ms")
      print(f"最慢帧: {max(processing_times)*1000:.1f} ms")

if __name__ == "__main__":
  onnx_file = fetch(OPENPILOT_MODEL)
  
  # 根据命令行参数决定是否编译模型
  if not os.path.exists(OUTPUT) or not args.compile_only:
    print("开始编译模型...")
    test_val = compile(onnx_file)
    print(f"模型已编译并保存到 {OUTPUT}")
  else:
    test_val = None
    print(f"跳过编译，使用已存在的模型文件: {OUTPUT}")
  
  # 如果指定了视频源，运行实时推理
  if args.video and not args.compile_only:
    print(f"使用视频文件进行实时推理: {args.video}")
    run_realtime_inference(OUTPUT, args.video)
  # 如果指定了摄像头，运行实时推理
  elif args.video == "摄像头" and not args.compile_only:
    print(f"使用摄像头 #{args.camera} 进行实时推理")
    run_realtime_inference(OUTPUT, "摄像头")
  # 否则，运行标准测试
  elif not args.compile_only:
    print("使用随机生成的数据进行测试...")
    with open(OUTPUT, "rb") as f: 
      pickle_loaded = pickle.load(f)
    
    # same randomness as compile
    Tensor.manual_seed(100)
    new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                  sorted(zip(pickle_loaded.captured.expected_names, pickle_loaded.captured.expected_st_vars_dtype_device))}
    
    test_val = test_vs_compile(pickle_loaded, new_inputs, test_val)
    if getenv("BENCHMARK"):
      for be in ["torch", "ort"]:
        try:
          timings = test_vs_onnx(new_inputs, None, onnx_file, be=="ort")
          print(f"timing {be}: {min(timings)*1000:.2f} ms")
        except Exception as e:
          print(f"{be} fail with {e}")
    if not getenv("FLOAT16"): test_vs_onnx(new_inputs, test_val, onnx_file, getenv("ORT"))

