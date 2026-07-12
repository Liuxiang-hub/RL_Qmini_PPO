# -*- coding: utf-8 -*-
"""
export_pt2onnx.py - PyTorch模型转ONNX格式导出脚本

功能:
  将训练好的PyTorch策略模型转换为ONNX格式,用于部署到真实机器人
  相当于"格式转换器",让电脑上的模型能在机器人上运行

在机器人走路中的作用:
  train.py训练的模型是PyTorch格式(.pt),只能在Python环境运行
  真实Qmini机器人需要ONNX格式才能高效推理
  
  转换流程:
  policy.pt (PyTorch) ──→ export_pt2onnx.py ──→ policy.onnx (通用格式)
  
  ONNX优势:
  1. 跨平台:可在C++、嵌入式设备上运行
  2. 高效:推理速度快,适合实时控制
  3. 通用:被多家厂商支持

使用方式:
  python export_pt2onnx.py --name q2
  
输出文件:
  experiments/q2/deploy/policy.onnx

代码结构:
  ┌───────────────────────────────────────────────────────────────────────┐
  │                         转换流程                                      │
  ├───────────────────────────────────────────────────────────────────────┤
  │                                                                       │
  │   1. 加载PyTorch模型                                                 │
  │      ├─ 读取cfg.yaml获取网络结构                                      │
  │      ├─ 创建Actor网络(deploy=True)                                   │
  │      └─ 加载policy.pt权重                                            │
  │                              ↓                                        │
  │   2. 转换为ONNX格式                                                  │
  │      └─ torch.onnx.export()                                         │
  │                              ↓                                        │
  │   3. 验证转换正确性                                                  │
  │      ├─ PyTorch推理输出                                              │
  │      ├─ ONNX推理输出                                                 │
  │      └─ 对比两者差异(Gap应接近0)                                     │
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘

与其他文件的关系:
  - train.py: 生成policy.pt,本脚本读取并转换
  - 真实Qmini: 加载policy.onnx进行推理控制
"""
import argparse
import isaacgym
import numpy as np
import os
from os.path import exists, join
import torch.nn as nn
from env.utils.helpers import class_to_dict

from model import load_actor
from env.utils import get_args
import importlib
from utils.yaml import ParamsProcess
import onnxruntime as ort
import torch

args = get_args()
exp_dir = join('experiments', args.name)
model_dir = join(exp_dir, 'model')
deploy_dir = join(exp_dir, 'deploy')
os.makedirs(deploy_dir, exist_ok=True)

paramsProcess = ParamsProcess()
params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
cfg = getattr(importlib.import_module('.'.join(['config', params['task']['cfg']])), params['task']['cfg'])
cfg = paramsProcess.dict2class(cfg, params)


def convert(name: str, model: nn.Module, input: np.ndarray):
    """
    将PyTorch模型转换为ONNX格式,并验证转换正确性
    
    参数:
      name: 模型名称(如'policy')
      model: PyTorch模型
      input: 输入数据的示例(用于确定输入维度)
    
    流程:
      1. 导出模型为ONNX格式
      2. 用PyTorch推理,记录输出
      3. 用ONNX推理,记录输出
      4. 对比两者差异,确保转换正确
    """
    print(f'\n******************************** {name} ********************************************\n')
    deploy_path = join(deploy_dir, f'{name}.onnx')
    
    # 核心: torch.onnx.export() 执行格式转换
    # 输入: model, 示例输入, 输出路径, opset版本, 输入输出名称
    torch.onnx.export(
        model,                           # 要导出的模型
        torch.from_numpy(input),         # 示例输入(确定维度)
        deploy_path,                     # 输出ONNX文件路径
        verbose=False,                   # 是否打印详细信息
        opset_version=12,               # ONNX算子集版本
        input_names=['input'],          # 输入节点名称
        output_names=['output']          # 输出节点名称
    )
    
    # 验证PyTorch输出
    print('Pytorch')
    print(model(torch.from_numpy(input)).detach().cpu().numpy())
    
    # 验证ONNX输出
    ort_session = ort.InferenceSession(deploy_path)
    print('Onnx')
    print(ort_session.run(None, {'input': input})[0])
    
    # 对比两者差异,确保转换正确
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
    print('Gap')
    print(gap)


# ========== 主程序 ==========
# 加载策略网络并导出为ONNX格式

# 加载配置
paramsProcess = ParamsProcess()
params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
cfg = getattr(importlib.import_module('.'.join(['config', params['task']['cfg']])), params['task']['cfg'])
cfg = paramsProcess.dict2class(cfg, params)

# 创建策略网络(Actor),deploy=True表示用于部署(输出确定性动作)
policy = load_actor(class_to_dict(cfg.policy), deploy=True).eval()

# 加载训练好的模型权重
policy_path = join(model_dir, 'policy.pt')
assert exists(policy_path), policy_path
policy.load_state_dict(torch.load(policy_path, map_location='cpu')['actor'], strict=False)

# 多次测试,确保导出稳定
for i in range(3):
    # 生成随机输入(观测维度要与配置一致)
    input = torch.rand([1, cfg.policy.num_observations]).cpu().numpy()
    convert('policy', policy, input)