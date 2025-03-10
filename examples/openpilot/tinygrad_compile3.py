#!/usr/bin/env python3
import cv2
import numpy as np
import os, sys, pickle, time, argparse
from pathlib import Path

if "FLOAT16" not in os.environ: os.environ["FLOAT16"] = "1"
if "IMAGE" not in os.environ: os.environ["IMAGE"] = "2"
if "NOLOCALS" not in os.environ: os.environ["NOLOCALS"] = "1"
if "JIT_BATCH_SIZE" not in os.environ: os.environ["JIT_BATCH_SIZE"] = "0"

from tinygrad import fetch, Tensor, TinyJit, Context, GlobalCounters, Device
from tinygrad.helpers import DEBUG, getenv
from tinygrad.tensor import _from_np_dtype
from tinygrad.engine.realize import CompiledRunner

Device.DEFAULT = "GPU"

DEFAULT_REMOTE_MODEL = "https://github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx"

'''
使用摄像头作为输入：
python3 examples/openpilot/tinygrad_compile3.py --model driving_vision.onnx driving_vision.pkl --run --video 0

使用视频文件作为输入：
python3 examples/openpilot/tinygrad_compile3.py --model driving_vision.onnx driving_vision.pkl --run --video 行车视频.mp4

指定期望动作和交通规则：
python3 examples/openpilot/tinygrad_compile3.py --model driving_vision.onnx driving_vision.pkl --run --video 行车视频.mp4 --desire 1 --traffic 0
'''

## 解析命令行参数
#def parse_args():
#    parser = argparse.ArgumentParser(description="编译和测试OpenPilot模型")
#    parser.add_argument("--model", type=str, default=DEFAULT_REMOTE_MODEL,
#                        help="ONNX模型文件路径或URL (默认: OpenPilot远程模型)")
#    parser.add_argument("--output", type=str, default="/tmp/openpilot.pkl",
#                        help="输出编译后的模型路径 (默认: /tmp/openpilot.pkl)")
#    parser.add_argument("output_file", type=str, nargs="?", 
#                        help="输出编译后的模型路径 (可选位置参数，如果提供则覆盖--output)")
#    parser.add_argument("--run", action="store_true", help="仅运行编译后的模型，不进行编译")
#    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
#    parser.add_argument("--ort", action="store_true", help="使用ONNX Runtime进行测试")
#    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
#    return parser.parse_args()

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="编译和测试OpenPilot模型")
    parser.add_argument("--model", type=str, default=DEFAULT_REMOTE_MODEL,
                        help="ONNX模型文件路径或URL (默认: OpenPilot远程模型)")
    parser.add_argument("--output", type=str, default="/tmp/openpilot.pkl",
                        help="输出编译后的模型路径 (默认: /tmp/openpilot.pkl)")
    parser.add_argument("output_file", type=str, nargs="?", 
                        help="输出编译后的模型路径 (可选位置参数，如果提供则覆盖--output)")
    parser.add_argument("--run", action="store_true", help="仅运行编译后的模型，不进行编译")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
    parser.add_argument("--ort", action="store_true", help="使用ONNX Runtime进行测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    # 添加视频流处理选项
    parser.add_argument("--video", type=str, help="视频源，可以是摄像头索引(例如:0)或视频文件路径")
    parser.add_argument("--no-display", action="store_true", help="不显示处理结果")
    parser.add_argument("--desire", type=int, default=0, choices=range(8),
                       help="期望动作 (0-7)，默认为0")
    parser.add_argument("--traffic", type=int, default=0, choices=[0, 1],
                       help="交通规则 (0=右侧通行, 1=左侧通行)，默认为0")
    return parser.parse_args()

# 检查导入ONNX库
try:
    import onnx
    from onnx.helper import tensor_dtype_to_np_dtype
    from extra.onnx import OnnxRunner   # TODO: port to main tinygrad
    HAS_ONNX = True
except ImportError:
    print("警告: 无法导入ONNX库，请确保安装了onnx和extra.onnx")
    HAS_ONNX = False




def yuv420_from_rgb(rgb_image):
    """
    将RGB图像转换为OpenPilot所需的YUV420格式（6通道格式）
    """
    # 确保图像尺寸为 256x512
    if rgb_image.shape[:2] != (256, 512):
        rgb_image = cv2.resize(rgb_image, (512, 256))
        
    # 将RGB转换为YUV
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    
    # 提取Y通道并分割为4个子采样
    y_channel = yuv_image[:, :, 0]
    y_0 = y_channel[::2, ::2]    # 偶数行偶数列
    y_1 = y_channel[::2, 1::2]   # 偶数行奇数列
    y_2 = y_channel[1::2, ::2]   # 奇数行偶数列
    y_3 = y_channel[1::2, 1::2]  # 奇数行奇数列
    
    # 提取并下采样U和V通道
    u_channel = yuv_image[:, :, 1][::2, ::2]  # 半分辨率
    v_channel = yuv_image[:, :, 2][::2, ::2]  # 半分辨率
    
    # 创建6通道YUV420格式
    yuv_channels = [y_0, y_1, y_2, y_3, u_channel, v_channel]
    return yuv_channels

def prepare_vision_inputs(prev_frame, curr_frame, is_wide=False):
    """
    准备视觉输入（标准或宽视角）
    
    参数:
        prev_frame: 前一帧RGB图像
        curr_frame: 当前帧RGB图像
        is_wide: 是否为宽视角图像
    
    返回:
        处理后的视觉输入张量
    """
    # 转换为YUV420格式
    prev_yuv = yuv420_from_rgb(prev_frame)
    curr_yuv = yuv420_from_rgb(curr_frame)
    
    # 组合两帧
    combined_yuv = []
    
    # 添加前一帧的通道
    for channel in prev_yuv:
        combined_yuv.append(channel)
    
    # 添加当前帧的通道
    for channel in curr_yuv:
        combined_yuv.append(channel)
    
    # 堆叠所有通道
    stacked_yuv = np.stack(combined_yuv, axis=0)
    
    # 添加批次维度（模型期望形状为 (1, 12, 128, 256)）
    stacked_yuv = np.expand_dims(stacked_yuv, axis=0)
    
    # 创建Tensor
    input_name = "big_input_imgs" if is_wide else "input_imgs"
    return input_name, Tensor(stacked_yuv, device=Device.DEFAULT)

def prepare_policy_inputs(desire=0, traffic_convention=0):
    """
    准备策略输入 - 由于模型不期望这些输入，此函数暂时不使用
    
    参数:
        desire: 期望的动作（0-7）
        traffic_convention: 交通规则（0=右侧通行, 1=左侧通行）
    
    返回:
        策略输入字典
    """
    inputs = {}
    
    # 注：当前模型不使用这些政策输入，仅保留函数以备将来使用
    
    return inputs

def process_video_stream(model, video_source=0, display=True, desire=0, traffic_convention=0):
    """
    处理视频流并将每一帧输入模型
    
    参数:
        model: 已加载的编译模型
        video_source: 视频源，可以是摄像头索引或视频文件路径
        display: 是否显示处理结果
        desire: 期望的动作（0-7）
        traffic_convention: 交通规则（0=右侧通行, 1=左侧通行）
    """
    # 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"错误: 无法打开视频源 {video_source}")
        return
    
    print(f"成功打开视频源: {video_source}")
    
    # 存储前一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取初始帧")
        cap.release()
        return
    
    # 调整大小并转换为RGB
    prev_frame = cv2.resize(prev_frame, (512, 256))
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    
    # 注：当前模型不使用策略输入
    # policy_inputs = prepare_policy_inputs(desire, traffic_convention)
    
    try:
        frame_count = 0
        while True:
            # 读取当前帧
            ret, curr_frame = cap.read()
            if not ret:
                print("视频流结束或读取错误")
                break
            
            # 调整大小并转换为RGB
            curr_frame = cv2.resize(curr_frame, (512, 256))
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            
            # 准备视觉输入（标准视图和宽视图）
            # 注意: 由于我们只有一个视频源，这里使用同一帧作为标准和宽视图输入
            # 在实际应用中，应该使用两个不同的摄像头
            image_input_name, image_tensor = prepare_vision_inputs(prev_frame, curr_frame, is_wide=False)
            wide_input_name, wide_image_tensor = prepare_vision_inputs(prev_frame, curr_frame, is_wide=True)
            
            # 准备完整的输入字典 - 仅使用模型期望的输入名称
            inputs = {
                image_input_name: image_tensor,
                wide_input_name: wide_image_tensor
                # 不包含策略输入，因为模型不期望这些输入
            }
            
            # 模型推理
            start_time = time.time()
            outputs = model(**inputs)
            inference_time = (time.time() - start_time) * 1000
            
            # 当前帧变为前一帧
            prev_frame = curr_frame.copy()
            
            # 后处理输出
            result_frame = visualize_outputs(curr_frame, outputs.numpy())
            
            # 显示结果
            if display:
                # 调整显示大小，使其更容易查看
                display_frame = cv2.resize(result_frame, (1024, 512))
                cv2.putText(display_frame, f"inference time: {inference_time:.1f}ms", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv2.imshow("OpenPilot 模型输出", display_frame)
                
                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            # 每处理20帧打印一次状态
            if frame_count % 20 == 0:
                print(f"已处理 {frame_count} 帧, 最新推理时间: {inference_time:.1f}ms")
    
    finally:
        # 释放资源
        cap.release()
        if display:
            cv2.destroyAllWindows()

def visualize_outputs(frame, outputs):
    """
    可视化模型输出
    
    参数:
        frame: 原始帧
        outputs: 模型输出
    
    返回:
        可视化后的帧
    """
    result_frame = frame.copy()
    
    try:
        # 打印输出形状信息，以便调试
        print(f"\n模型输出形状: {outputs.shape}")
        
        # 这里是一个简化示例，实际处理需要根据模型的具体输出格式调整
        height, width = frame.shape[:2]
        
        # 引导路径和车道线的颜色
        colors = [
            (0, 0, 255),   # 红色 - 左边线
            (0, 255, 0),   # 绿色 - 路径
            (255, 0, 0),   # 蓝色 - 右边线
            (255, 255, 0)  # 黄色 - 其他
        ]
        
        # 假设输出是一个大向量，我们尝试绘制前100个点（如果有的话）
        if outputs.shape[1] >= 100:
            # 将输出重塑为更可视化的格式
            # 我们假设前100个值是50个点（x, y坐标）
            points = []
            for i in range(0, min(100, outputs.shape[1]), 2):
                if i+1 < outputs.shape[1]:
                    # 将输出值缩放到图像大小
                    x = int((outputs[0, i] + 1) * width / 2)
                    y = int((outputs[0, i+1] + 1) * height / 2)
                    if 0 <= x < width and 0 <= y < height:
                        points.append((x, y))
            
            # 绘制点
            for point in points:
                cv2.circle(result_frame, point, 3, (0, 255, 0), -1)
            
            # 绘制线段
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(result_frame, points[i], points[i+1], (0, 255, 255), 2)
                
        # 在结果帧上显示一些信息
        cv2.putText(result_frame, "Vision Model", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(result_frame, f"Output shape: {outputs.shape}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
    except Exception as e:
        print(f"可视化输出时出错: {e}")
    
    return result_frame




def compile(onnx_file):
  """编译ONNX模型
  
  参数:
    onnx_file: ONNX模型文件路径
    
  返回:
    测试值用于验证
  """
  # 验证模型文件是否存在
  if not os.path.exists(onnx_file):
    print(f"错误: 模型文件不存在: {onnx_file}")
    return None
    
  # 检查文件大小
  file_size = os.path.getsize(onnx_file)
  if file_size == 0:
    print(f"错误: 模型文件为空: {onnx_file}")
    return None
    
  print(f"模型文件大小: {file_size/1024/1024:.2f} MB")
  
  try:
    # 加载ONNX模型
    print(f"正在加载模型: {onnx_file}")
    onnx_model = onnx.load(onnx_file)
    run_onnx = OnnxRunner(onnx_model)
    print("成功加载模型")
  except Exception as e:
    print(f"加载模型时出错: {e}")
    return None

  input_shapes = {inp.name:tuple(x.dim_value for x in inp.type.tensor_type.shape.dim) for inp in onnx_model.graph.input}
  input_types = {inp.name: tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type) for inp in onnx_model.graph.input}
  # Float inputs and outputs to tinyjits for openpilot are always float32
  input_types = {k:(np.float32 if v==np.float16 else v) for k,v in input_types.items()}
  Tensor.manual_seed(100)
  new_inputs = {k:Tensor.randn(*shp, dtype=_from_np_dtype(input_types[k])).mul(8).realize() for k,shp in sorted(input_shapes.items())}
  new_inputs_numpy = {k:v.numpy() for k,v in new_inputs.items()}
  print("created tensors")

  run_onnx_jit = TinyJit(lambda **kwargs:
                         next(iter(run_onnx({k:v.to(Device.DEFAULT) for k,v in kwargs.items()}).values())).cast('float32'), prune=True)
  for i in range(3):
    GlobalCounters.reset()
    print(f"run {i}")
    inputs = {**{k:v.clone() for k,v in new_inputs.items() if 'img' in k},
              **{k:Tensor(v, device="NPY").realize() for k,v in new_inputs_numpy.items() if 'img' not in k}}
    with Context(DEBUG=max(DEBUG.value, 2 if i == 2 else 1)):
      ret = run_onnx_jit(**inputs).numpy()
    # copy i == 1 so use of JITBEAM is okay
    if i == 1: test_val = np.copy(ret)
  print(f"captured {len(run_onnx_jit.captured.jit_cache)} kernels")
  np.testing.assert_equal(test_val, ret, "JIT run failed")
  print("jit run validated")

  # checks from compile2
  kernel_count = 0
  read_image_count = 0
  gated_read_image_count = 0
  for ei in run_onnx_jit.captured.jit_cache:
    if isinstance(ei.prg, CompiledRunner):
      kernel_count += 1
      read_image_count += ei.prg.p.src.count("read_image")
      gated_read_image_count += ei.prg.p.src.count("?read_image")
  print(f"{kernel_count=},  {read_image_count=}, {gated_read_image_count=}")
  if (allowed_kernel_count:=getenv("ALLOWED_KERNEL_COUNT", -1)) != -1:
    assert kernel_count <= allowed_kernel_count, f"too many kernels! {kernel_count=}, {allowed_kernel_count=}"
  if (allowed_read_image:=getenv("ALLOWED_READ_IMAGE", -1)) != -1:
    assert read_image_count == allowed_read_image, f"different read_image! {read_image_count=}, {allowed_read_image=}"
  if (allowed_gated_read_image:=getenv("ALLOWED_GATED_READ_IMAGE", -1)) != -1:
    assert gated_read_image_count <= allowed_gated_read_image, f"too many gated read_image! {gated_read_image_count=}, {allowed_gated_read_image=}"

  # 保存编译后的模型
  output_dir = os.path.dirname(OUTPUT)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
      
  try:
    with open(OUTPUT, "wb") as f:
      pickle.dump(run_onnx_jit, f)
      
    mdl_sz = os.path.getsize(onnx_file)
    pkl_sz = os.path.getsize(OUTPUT)
    print(f"原始模型大小: {mdl_sz/1e6:.2f} MB")
    print(f"编译后模型大小: {pkl_sz/1e6:.2f} MB")
    print(f"编译后模型已保存至: {OUTPUT}")
    print("**** 编译完成 ****")
    return test_val
  except Exception as e:
    print(f"保存编译后模型时出错: {e}")
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

def load_model(model_path):
  """加载已编译的模型
  
  参数:
    model_path: 编译后的模型路径
    
  返回:
    加载的模型
  """
  if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    return None
    
  try:
    with open(model_path, "rb") as f:
      model = pickle.load(f)
    print(f"成功加载模型: {model_path}")
    return model
  except Exception as e:
    print(f"加载模型时出错: {e}")
    return None

def get_model_file(model_path):
  """获取模型文件路径（支持URL或本地路径）
  
  参数:
    model_path: 模型路径或URL
    
  返回:
    本地模型文件路径
  """
  # 如果是URL，下载模型
  if model_path.startswith(('http://', 'https://')):
    try:
      print(f"正在从URL下载模型: {model_path}")
      return fetch(model_path)
    except Exception as e:
      print(f"下载模型时出错: {e}")
      return None
  
  # 如果是本地路径，直接返回
  if os.path.exists(model_path):
    # 如果是相对路径，转换为绝对路径
    return os.path.abspath(model_path)
  else:
    print(f"错误: 无法找到模型文件: {model_path}")
    return None

if __name__ == "__main__":
  # 解析命令行参数
  args = parse_args()
  # 如果提供了位置参数，使用位置参数作为输出文件路径
  OUTPUT = args.output_file if args.output_file else args.output
  
  # 获取模型文件
  onnx_file = get_model_file(args.model)
  if not onnx_file:
    sys.exit(1)
    
  # 编译或加载模型
  test_val = None
  if not args.run:
    print(f"正在编译模型: {onnx_file}")
    test_val = compile(onnx_file)
    if test_val is None:
      print("编译失败，退出程序。")
      sys.exit(1)
  
  # 加载编译后的模型
  try:
    with open(OUTPUT, "rb") as f: 
      pickle_loaded = pickle.load(f)
  except Exception as e:
    print(f"加载编译后的模型时出错: {e}")
    sys.exit(1)
  
  # 创建测试输入
  print("创建测试输入数据...")
  Tensor.manual_seed(100)
  try:
    new_inputs = {nm:Tensor.randn(*st.shape, dtype=dtype).mul(8).realize() for nm, (st, _, dtype, _) in
                  sorted(zip(pickle_loaded.captured.expected_names, pickle_loaded.captured.expected_st_vars_dtype_device))}
  except Exception as e:
    print(f"创建测试输入时出错: {e}")
    sys.exit(1)
  
  # 运行测试
  print("运行模型测试...")
  test_val = test_vs_compile(pickle_loaded, new_inputs, test_val)
  
  # 如果指定了基准测试，运行基准测试
  if args.benchmark:
    print("运行基准测试...")
    for be in ["torch", "ort"]:
      try:
        timings = test_vs_onnx(new_inputs, None, onnx_file, be=="ort")
        print(f"{be}基准测试: {min(timings)*1000:.2f} ms")
      except Exception as e:
        print(f"{be}基准测试失败: {e}")
  
  # 如果不是FP16模式，与ONNX运行结果对比
  if not getenv("FLOAT16"): 
    print("与ONNX原始模型比较结果...")
    test_vs_onnx(new_inputs, test_val, onnx_file, args.ort)
    
  print("所有测试完成!")

  # 处理视频流（如果指定）
  if args.video:
      print("\n开始处理视频流...")
      video_source = int(args.video) if args.video.isdigit() else args.video
      process_video_stream(
          pickle_loaded, 
          video_source, 
          not args.no_display,
          args.desire, 
          args.traffic
      )