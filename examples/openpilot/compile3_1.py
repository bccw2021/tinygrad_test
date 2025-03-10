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
#try:
#    import cv2
#    HAVE_OPENCV = True
#except ImportError:
#    print("警告: 未找到OpenCV。要使用实时输入流，请安装OpenCV: pip install opencv-python")
#    HAVE_OPENCV = False

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import _from_np_dtype
from tinygrad.engine.realize import CompiledRunner
from tinygrad.ops import PatternMatcher, UPat, Ops

import onnx
from onnx.helper import tensor_dtype_to_np_dtype
from extra.onnx import OnnxRunner   # TODO: port to main tinygrad

OPENPILOT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/openpilot.pkl"
# 强制使用GPU（OpenCL）
Device.DEFAULT = "GPU"


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
    
    # 准备输入数据：图像数据保持在GPU上，非图像数据使用NumPy数组
    inputs = {**{k:v.clone() for k,v in new_inputs.items() if 'img' in k},
              **{k:Tensor(v, device="NPY").realize() for k,v in new_inputs_numpy.items() if 'img' not in k}}
    
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

if __name__ == "__main__":
  onnx_file = fetch(OPENPILOT_MODEL)
  test_val = compile(onnx_file) if not getenv("RUN") else None

  with open(OUTPUT, "rb") as f: pickle_loaded = pickle.load(f)

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

