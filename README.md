# RoboTamer4Qmini_v1.0

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)
[![Powered by Isaac Gym](https://img.shields.io/badge/Powered%20by-Isaac%20Gym-blue.svg)](https://developer.nvidia.com/isaac-gym)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-green.svg)](https://arxiv.org/abs/1707.06347)
![Version](https://img.shields.io/badge/Version-1.0-blue.svg)

**当前版本为 1.0。**
✅ 初始版本已发布。
🚀 后续计划持续更新，新增功能并优化性能。
敬请关注更新日志！

本仓库提供了一个基于 NVIDIA Isaac Gym 深度强化学习的双足机器人运动控制开源框架。它支持对 Unitree Qmini 等机器人在崎岖地形上的行走进行训练，并在训练中引入了关键的领域随机化（domain randomization）与随机推扰（random push），以实现从仿真到真机（sim-to-real）的迁移。仓库包含训练与部署双足机器人的完整代码。

**维护者**：陈延云（Yanyun Chen）、方体宇（Tiyu Fang）、谭文浩（Wenhao Tan）、方兴（Xing Fang）、李凯文（Kaiwen Li）、张坤奇（Kunqi Zhang）、张伟（Wei Zhang）
**所属单位**：山东大学控制科学与工程学院，视觉感知与智能系统实验室（VSISLab）
**网站**：www.vsislab.com
**联系邮箱**：info@vsislab.com

## 特性

- **先进的控制算法**：基于 Isaac Gym 实现 PPO 算法，实现稳定高效的步态控制。
- **模块化设计**：提供易用的接口，便于自定义机器人模型、环境与奖励函数。
- **真机部署**：提供将训练好的策略从仿真迁移到实体机器人的工具。
- **完善文档**：提供详细的教程与文档，便于快速上手与二次开发。

## 代码结构

   ```
RoboTamer4Qmini/
   ├── assets/                 # 机器人的 URDF 模型
   ├── config/                 # 配置文件
   ├── env/                    # 仿真环境
   ├── experiments/            # 预训练模型与评测结果
   ├── model/                  # 神经网络结构
   ├── utils/                  # 工具函数
   ├── export_pt2onnx.py       # 将 *.pt 预训练模型导出为 *.onnx
   ├── play.py                 # 评测预训练模型
   ├── train.py                # 训练模型
   ├── tune_pid.py             # 优化 PID 参数，以缩小仿真与真机之间的差异
   ├── tune_urdf.py            # 加载并查看机器人的 URDF 模型
   ├── requirements.txt        # 额外依赖
   └── README.md
   ```

### 注意事项

* 在 _play.py_、_train.py_、_Base.py_、_tune_urdf.py_ 和 _tune_pid.py_ 中存在一些硬编码路径，请注意修改。
* 本仓库不再维护，如有问题请发送邮件至 info@vsislab.com。
* 项目需安装成功后才能运行。

## 安装

### 环境要求

- Ubuntu 18.04 或 20.04
- NVIDIA 驱动版本 470+
- 硬件：NVIDIA Pascal 或更新的 GPU，显存至少 8 GB
- CUDA 11.4+
- Python 3.8+
- PyTorch 2.0.0+
- Isaac Gym 1.0rc3+（用于仿真环境）
- 其他依赖（见 `requirements.txt` 与"安装依赖"）

### 步骤

1. 创建新的 conda 环境：

```bash
$ conda create -n isaac python==3.8 && conda activate isaac
```

2. 安装依赖：

```bash
    pip3 install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.0
    tar -zxvf IsaacGym_Preview_3_Package.tar.gz && cd ./isaacgym/python && pip install -e .
    pip3 install -r requirements.txt
    pip3 install matplotlib pandas tensorboard opencv-python numpy==1.23.5 openpyxl onnxruntime onnx
```

## 使用说明

### 训练（默认名称为 test）

```bash
$ python train.py --config BIRL --name <name>
```

  - --name <str> # 实验名称（默认 'test'），覆盖配置文件中的设置
  - --config <str> # 实验配置文件（默认 'config.Base'）
  - --resume <str> # 从检查点恢复训练（默认 'test'），需要指定检查点路径
  - --render # 布尔标志（默认 False），强制关闭显示
  - --fix_cam # 布尔标志（默认 False），将相机固定在 0 号环境中的机器人上
  - --horovod # 布尔标志（默认 False），启用 Horovod 多卡训练
  - --rl_device <str> # 强化学习设备（默认 'cuda:0'），支持 'cpu' / 'cuda:0' 等格式
  - --num_envs <int> # 环境数量（默认 None），覆盖配置文件设置
  - --seed <int> # 随机种子（默认 None），覆盖配置文件设置
  - --max_iterations <int> # 最大迭代次数（默认 None），覆盖配置文件设置

### 在浏览器中可视化训练日志

```bash
$ tensorboard --logdir experiments/
```

### 评测（默认名称为 test）

```bash
$ python play.py --render --name <name>
```

#### 训练或评测时打开查看器（默认关闭）

```bash
$ python play.py --name <name> --render
```

#### 修改评测时长（默认 4s）

```bash
$ python play.py --name <name> --render --time 10
```

#### 评测时录制视频（默认关闭）

```bash
$ python play.py --name <name> --render --time 10 --video
```

#### 评测时保存数据到 Excel（默认关闭）

```bash
$ python play.py --name <name> --render --time 10 --video --debug
```

  - --name <str> # 实验名称（默认 'test'），覆盖配置文件中的设置
  - --render # 布尔标志（默认 False），强制关闭显示
  - --fix_cam # 布尔标志（默认 False），将相机固定在 0 号环境中的机器人上
  - --cmp_real # 布尔标志（默认 False），与真机数据对比绘图
  - --plt_sim # 布尔标志（默认 False），绘制仿真曲线
  - --num_envs <int> # 环境数量（默认 None），覆盖配置文件设置
  - --video # 布尔标志（默认 False），录制视频
  - --time <float> # 评测时长（秒，默认 10s）
  - --iter <int> # 按训练迭代次数指定策略（默认 None，加载当前目录下最新的策略）
  - --epochs <int> # 评测轮数（默认 1）
  - --debug # 布尔标志（默认 False），保存数据到 Excel

### 导出预训练模型为 ONNX（默认名称为 test）

```bash
$ python export_pt2onnx.py --name <name>
```

  - --name <str> # 实验名称（默认 'test'），policy.onnx 保存到目录 'name/deploy'

### 加载 URDF 模型

```bash
$ python tune_urdf.py
```

### 优化 PID 参数以缩小仿真与真机之间的差异

```bash
$ python tune_pid.py
```

  - --mode <str> # 测试模式：{'sin', 'real', 'reset'}，选择测试模式（仿真、真机或重置）

## 参与贡献

欢迎社区贡献！提交 Pull Request 前请先通过 info@vsislab.com 联系我们。

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/your-feature`）
3. 提交更改（`git commit -am 'Add some feature'`）
4. 推送到分支（`git push origin feature/your-feature`）
5. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证——详见 [LICENSE] 文件。

## 引用

如果您在研究中使用了本代码，请引用我们的工作：

```
@article{Chen2025GALA,
  author={Yanyun Chen, Ran Song, Jiapeng Sheng, Xing Fang, Wenhao Tan, Wei Zhang and Yibin Li},
  journal={IEEE Transactions on Automation Science and Engineering},
  title={A Generalist Agent Learning Architecture for Versatile Quadruped Locomotion},
  year={2025},
  keywords={Quadruped Robots, Versatile Locomotion, Deep Reinforcement Learning, A Single Policy Network, Multiple Critic Networks}
}

@article{Sheng2022BioInspiredRL,
  title={Bio-Inspired Rhythmic Locomotion for Quadruped Robots},
  author={Jiapeng Sheng and Yanyun Chen and Xing Fang and Wei Zhang and Ran Song and Yuan-hua Zheng and Yibin Li},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  volume={7},
  pages={6782-6789}
}

@article{Liu2024MCLER,
  author={Liu, Maoqi and Chen, Yanyun and Song, Ran and Qian, Longyue and Fang, Xing and Tan, Wenhao and Li, Yibin and Zhang, Wei},
  journal={IEEE Robotics and Automation Letters},
  title={MCLER: Multi-Critic Continual Learning With Experience Replay for Quadruped Gait Generation},
  year={2024},
  volume={9},
  number={9},
  pages={8138-8145},
  keywords={Quadrupedal robots;Task analysis;Continuing education;Optimization;Legged locomotion;Training;Motors;Continual learning;legged robots},
  doi={10.1109/LRA.2024.3418310}
}
```
