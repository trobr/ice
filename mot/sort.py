import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


from segment import adaptive_threshold_blocks, histogram_stretching
from dehaze import dehaze
from mot.base import TrackedObject, TrackedObjectMetaInfo


def preprocess_binary(binary, min_area=10000, max_size_ratio=0.9, min_size_ratio=0.1):
    """
    预处理二值图像：
    1. 移除小面积区域
    2. 移除超过画面90%大小的物体
    """
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取图像尺寸
    h, w = binary.shape[:2]
    max_w = w * max_size_ratio
    max_h = h * max_size_ratio
    min_w = w * min_size_ratio
    min_h = h * min_size_ratio
    
    filtered_contours = []
    areas = []
    for cnt in contours:
        # 过滤小面积
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # 过滤过大物体
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > max_w or bh > max_h or \
            (bw < min_w and bh < min_h):
            continue
            
        filtered_contours.append(cnt)
        areas.append(area)
        
    return filtered_contours, areas

def get_centroids(contours):
    """从轮廓计算质心"""
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))
    return centroids

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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    original_gray, _, _ = cv2.split(lab)
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


class SortTracker(object):
    def __init__(self, corrector, fps, max_lost=10, callback=None):
        self.tracked_objects = []
        self.max_lost = max_lost
        self.next_id = 0
        self.corrector = corrector
        self.fps = fps
        self.callback = callback

    def update(self, frame, ship_speed, centroids, data):
        frame_size = frame.shape[1], frame.shape[0]
        # 预测所有现有目标的位置
        for obj in self.tracked_objects:
            obj.predict()

        # 构建代价矩阵（预测位置与实际检测的欧氏距离）
        cost_matrix = []
        for obj in self.tracked_objects:
            row = []
            for centroid in centroids:
                dx = obj.predicted_position[0] - centroid[0]
                dy = obj.predicted_position[1] - centroid[1]
                distance = np.hypot(dx, dy)
                row.append(distance if distance < 300 else 1e5)  # 最大关联距离50像素
            cost_matrix.append(row)

        # 匈牙利算法匹配
        if cost_matrix:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []

        # 处理匹配结果
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] < 300:
                matches.append((r, c))

        # 更新匹配目标
        for r, c in matches:
            self.tracked_objects[r].update(centroids[c], ship_speed)

        # 标记未匹配的现有目标
        unmatched_existing = set(range(len(self.tracked_objects))) - set([r for r, _ in matches])
        # 逆序删除避免索引错误
        for i in sorted(unmatched_existing, reverse=True):
            self.tracked_objects[i].lost_count += 1
            # 删除没有连续丢失的目标或者没有初始化跟踪到2s的目标
            if self.tracked_objects[i].lost_count > self.max_lost or not self.tracked_objects[i].tracked:
                if self.callback:
                    self.callback(self.tracked_objects[i])
                self.tracked_objects.pop(i)
            
        # 处理新目标
        unmatched_new = set(range(len(centroids))) - set([c for _, c in matches])
        meta_info = TrackedObjectMetaInfo(data['latitude'], data['longitude'], data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
        for j in unmatched_new:
            new_obj = TrackedObject(self.next_id, centroids[j], frame_size, corrector=self.corrector, fps=self.fps)
            new_obj.set_meta_info(meta_info)
            self.tracked_objects.append(new_obj)
            self.next_id += 1

        return self.tracked_objects

