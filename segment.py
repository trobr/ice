import cv2
import torch
import numpy as np
from numba import njit, prange

from dehaze import dehaze
from valley import valley_threshold_with_min


def tensor2np(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def otsu_threshold_with_min(img, min_threshold=0):
    """使用大津法进行二值化，并可指定阈值的最小值
    
    Args:
        img (numpy.ndarray): 灰度图像
        min_threshold (int): 阈值的最小值，默认为0
    
    Returns:
        int: 最终使用的阈值
        numpy.ndarray: 二值化后的图像
    """
    # 确保图像是灰度图
    if len(img.shape) > 2:
        raise ValueError("Image must be grayscale.")
    
    # 计算直方图
    hist = np.histogram(img, bins=256, range=(0, 256))[0]
    total_pixels = img.size
    if total_pixels == 0:
        return 0, np.zeros_like(img)
    
    # 计算灰度总和和累积直方图
    sum_total = np.sum(np.arange(256) * hist)
    cum_sum = np.cumsum(hist)
    sum0 = np.cumsum(hist * np.arange(256))
    
    max_variance = -1
    best_threshold = 0
    
    # 遍历所有可能的阈值
    for t in range(256):
        if cum_sum[t] == 0 or cum_sum[t] == total_pixels:
            continue
        
        # 计算类间方差
        ω0 = cum_sum[t] / total_pixels
        ω1 = 1 - ω0
        μ0 = sum0[t] / cum_sum[t]
        μ1 = (sum_total - sum0[t]) / (total_pixels - cum_sum[t])
        variance = ω0 * ω1 * (μ0 - μ1) ** 2
        
        # 更新最佳阈值（方差相同时取较大的t）
        if variance > max_variance or (variance == max_variance and t > best_threshold):
            max_variance = variance
            best_threshold = t
    
    # 应用最小阈值限制
    final_threshold = max(best_threshold, min_threshold)
    
    # 生成二值化图像
    binary = np.where(img >= final_threshold, 255, 0).astype(np.uint8)
    
    return final_threshold, binary



def filter_masks_by_brightness(image, binary_mask, threshold=200):
    """
    根据连通区域的平均亮度过滤二值掩码
    
    参数：
    original_gray (numpy.ndarray): 原始灰度图像（单通道）
    binary_mask (numpy.ndarray): 二值化掩码（0和255）
    threshold (int): 亮度过滤阈值（0-255）
    
    返回：
    numpy.ndarray: 过滤后的二值掩码
    """
    original_gray = image
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # original_gray, _, _ = cv2.split(lab)
    # 输入校验
    assert len(original_gray.shape) == 2, "原始图像必须为灰度图"
    assert len(binary_mask.shape) == 2, "掩码必须为单通道"
    assert original_gray.shape == binary_mask.shape, "图像和掩码尺寸不一致"
    
    # 查找所有连通区域
    contours, _ = cv2.findContours(
        binary_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 创建过滤后的掩码
    filtered_mask = np.zeros_like(binary_mask)
    
    for cnt in contours:
        # 创建当前区域的掩码
        component_mask = np.zeros_like(binary_mask)
        cv2.drawContours(component_mask, [cnt], -1, 255, -1)
        
        # 计算区域平均亮度（使用原始灰度图）
        mean_val = cv2.mean(original_gray, mask=component_mask)[0]
        
        # 保留高亮度区域
        if mean_val >= threshold:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    
    return filtered_mask


def _adaptive_threshold_blocks(image, n, m, start_row, start_col, algo, min_threshold):
    """
    将图像分成 n 行 m 列的小块，每个小块单独做自适应阈值分割，再合并成一个整体。
    
    参数:
    - image: 输入的图像（灰度图）。
    - n: 行数，图像划分为 n 行。
    - m: 列数，图像划分为 m 列。
    
    返回:
    - 处理后的图像。
    """
    # 获取图像的大小
    height, width, *_ = image.shape

    # 创建一个空白的图像来存储合并后的结果
    result_image = np.zeros_like(image)

    # 计算每个块的高度和宽度
    block_height = height // n
    block_width = width // m
    
    # 处理每个小块
    for i in range(n):
        for j in range(m):
            # 计算当前块的起始和结束位置
            start_y = i * block_height
            end_y = (i + 1) * block_height if i != n - 1 else height
            start_x = j * block_width
            end_x = (j + 1) * block_width if j != m - 1 else width

            if i < start_row or j < start_col:
                result_image[start_y:end_y, start_x:end_x] = np.zeros_like(image[start_y:end_y, start_x:end_x])
                continue
            
            # 获取当前小块
            block = image[start_y:end_y, start_x:end_x]
            
            # 对当前小块进行自适应阈值处理
            # block_thresholded = cv2.adaptiveThreshold(block, 255, 
            #                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            #                                          cv2.THRESH_BINARY, 11, 2)
            # _, block_thresholded = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if algo == 'otsu_with_min':
                _, block_thresholded = otsu_threshold_with_min(block, min_threshold)
            elif algo == 'valley_with_min':
                _, block_thresholded = valley_threshold_with_min(block, min_threshold)

            # 将处理后的小块放回到结果图像对应的位置
            binary = filter_masks_by_brightness(block, block_thresholded)
            result_image[start_y:end_y, start_x:end_x] = binary
    
    return result_image


def adaptive_threshold_blocks(image, row=16, col=1, start_row=3, start_col=0, algo='otsu_with_min', min_threshold=180):
    if algo not in {'otsu_with_min', 'valley_with_min'}:
        raise ValueError(f"algo: {algo} not in ('otsu_with_min', 'valley_with_min')")

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = _adaptive_threshold_blocks(image, row, col, start_row, start_col, algo, min_threshold)
    kernel = np.ones((15, 15), dtype=np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    binary = cv2.dilate(binary, np.ones((7, 7), dtype=np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def histogram_stretching(img, clahe=False):
    # 计算当前像素的最小值和最大值
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # 仅对亮度通道 L 做均衡化
    if clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
    else:
        l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    # img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    
    # for c in range(3):
        # min_val = np.min(img[..., c])
        # max_val = np.max(img[..., c])
        # 线性拉伸公式
        # img[..., c] = ((img[..., c] - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # for c in range(3):
    #     img[..., c] = cv2.equalizeHist(img[..., c])

    return img    
