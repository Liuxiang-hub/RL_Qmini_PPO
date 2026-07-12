import cv2
import datetime
import numpy as np
import os
from os.path import exists
import shutil
import pandas as pd
import torch

import torch


def get_unique_num():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + str(np.random.randint(10, 100))


def get_print_time(t):
    h = int(t // 3600)
    m = int((t // 60) % 60)
    s = int(t % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def clear_dir(dir):
    if exists(dir):
        shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


def load_data(path, names):
    xl = pd.ExcelFile(path)
    real_data_list = []
    for epo in names:
        real_data_list.append(xl.parse(f'{epo}').values)
    return real_data_list


def get_state_noise_scales(self, len_vec, max_val):
    noise_vec = torch.ones(len_vec, device=self.device, dtype=torch.float) * max_val
    return noise_vec  # * ratio


def safe_cv2_crop(img, crop_size=600):
    try:
        h, w = img.shape[:2]
        target_size = min(crop_size, w, h)  # 自动适应最小尺寸

        x = (w - target_size) // 2 + 30
        y = (h - target_size) // 2 + min(crop_size, 100)

        # 边界检查
        if x < 0 or y < 0:
            print(f"警告：原图尺寸 {w}x{h} 小于目标尺寸 {crop_size}")
            return None

        cropped = img[y:y + target_size, x:x + target_size - 100]

        return cropped

    except Exception as e:
        print(f"处理时发生错误: {str(e)}")
        return None
