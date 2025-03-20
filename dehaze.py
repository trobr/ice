import time

import cv2
import numpy as np
import numexpr as ne
from numba import jit, prange


def fast_min_channel(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # 提取 R, G, B 通道
    return ne.evaluate("where((r <= g) & (r <= b), r, where(g <= b, g, b))")


def dark_channel(img, window_size=15):
    # min_channel = np.min(img, axis=2)
    min_channel = fast_min_channel(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(img, dark_ch, percentile=0.1):
    h, w = dark_ch.shape
    img_flat = img.reshape(-1, 3)
    dark_flat = dark_ch.ravel()
    
    num_pixels = int(h * w * percentile / 100)
    indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
    
    top_pixels = img_flat[indices]
    return np.max(top_pixels, axis=0)

def guided_filter(I, p, radius=40, eps=1e-3):
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(I, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(I*p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I*mean_p
    
    mean_II = cv2.boxFilter(I*I, -1, (radius, radius))
    var_I = mean_II - mean_I*mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    
    return (mean_a*I + mean_b) * 255


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    result[:, :, 2] = result[:, :, 2] - (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def dehaze(img, window_size=15, omega=0.9, t0=0.1, radius=40, eps=1e-3):
    s1 = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    
    # 计算暗通道
    dark = dark_channel(img, window_size)
    s2 = time.perf_counter()
    
    # 估计大气光
    atmospheric_light = estimate_atmospheric_light(img, dark)
    s3 = time.perf_counter()
    
    # 估计透射率
    img_normalized = img / atmospheric_light
    transmission = 1 - omega * dark_channel(img_normalized, window_size)
    s4 = time.perf_counter()
    
    # 使用导向滤波优化透射率
    transmission = guided_filter(gray, transmission, radius, eps)
    # transmission = guided_filter_numba(gray, transmission, radius, eps)
    s51 = time.perf_counter()
    transmission = np.clip(transmission, t0, 1.0)
    s5 = time.perf_counter()
    
    # 恢复无雾图像
    scene = np.zeros_like(img)
    for i in range(3):
        scene[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    s6 = time.perf_counter()
    
    # 限制像素范围并转换为8位图像
    scene = np.clip(scene*255, 0, 255).astype(np.uint8)
    s7 = time.perf_counter()
    print(f'----ttttt {s7-s1} {s7-s6} {s6-s5} {s5-s4} [{s5-s51} {s51-s4}] {s4-s3} {s3-s2} {s2-s1}')
    
    # 应用白平衡
    # return white_balance(scene)
    return scene
