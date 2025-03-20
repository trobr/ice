import cv2
import numpy as np

def valley_threshold_with_min(img, min_threshold=0, smooth_window=5):
    """基于直方图谷底的最小阈值二值化
    
    Args:
        img (numpy.ndarray): 灰度图像
        min_threshold (int): 阈值的最小值，默认为0
        smooth_window (int): 直方图平滑的窗口大小（必须为奇数），默认为5
        
    Returns:
        int: 最终阈值
        numpy.ndarray: 二值化图像
    """
    if len(img.shape) > 2:
        raise ValueError("输入必须是灰度图像")
    if smooth_window % 2 == 0:
        smooth_window += 1
    
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    
    # 平滑直方图（高斯滤波）
    smoothed_hist = cv2.GaussianBlur(hist, (smooth_window, 1), 0)
    
    candidate_threshold = find_valley_min(img, smoothed_hist)

    if candidate_threshold < 0:
        final_threshold = -1
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # 应用最小阈值约束
        final_threshold = max(candidate_threshold, min_threshold)
        # 执行二值化
        _, binary = cv2.threshold(img, final_threshold, 255, cv2.THRESH_BINARY)
    
    return final_threshold, binary

def find_valley_min(img, hist):
    """寻找平滑后直方图的谷底位置"""
    # 寻找可能的双峰区间
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append(i)
    
    # 若找不到双峰，则默认使用Otsu阈值
    if len(peaks) < 2:
        return -1
    
    # 取前两个主峰之间的谷底
    first_peak = peaks[0]
    second_peak = peaks[1]
    valley_region = hist[first_peak:second_peak]
    valley_pos = np.argmin(valley_region) + first_peak
    return valley_pos
