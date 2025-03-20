from collections import deque


import numpy as np


# 相机参数（需标定或通过参考物体计算）
focal_length_pixel = 800  # 焦距（像素）
sensor_width_mm = 36.0    # 传感器宽度（毫米）
pixel_per_mm = (1920 / sensor_width_mm)  # 像素/毫米（假设图像宽度1920）
scale_factor = sensor_width_mm / (focal_length_pixel * 1000)  # 米/像素

# 速度计算窗口（平滑用）
speed_window = deque(maxlen=5)  # 取最近5帧计算平均速度

def pixel_to_real(pixel_distance):
    """ 将像素距离转换为真实距离（米） """
    return pixel_distance * scale_factor

def calculate_speed(prev_pos, curr_pos, fps):
    """ 计算速度（米/秒） """
    dx = pixel_to_real(curr_pos[0] - prev_pos[0])
    dy = pixel_to_real(curr_pos[1] - prev_pos[1])
    displacement = np.sqrt(dx**2 + dy**2)
    return displacement * fps
