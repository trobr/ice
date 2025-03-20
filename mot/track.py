import os
import cv2
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


from segment import adaptive_threshold_blocks, histogram_stretching
from dehaze import dehaze
from mot.sort import SortTracker
from mot.ocv_inter import MultiObjectTracker
from mot.sift import SIFTMultiObjectTracker
from mot.base import TrackedObject, GeometricCorrector
from mot.base_detector import BaseDetector
from mot.sam_detector import SamDetector


class TrackedObject:
    def __init__(self, object_id, position, frame_size):
        self.object_id = object_id
        self.kalman = cv2.KalmanFilter(4, 2)  # 状态4（x,y,dx,dy），观测2（x,y）
        self.frame_size = frame_size  # (width, height)
        
        # 初始化卡尔曼参数
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], 
                                                [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-2 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
        
        # 初始状态
        self.kalman.statePost = np.array([[position[0]], 
                                        [position[1]], 
                                        [0], 
                                        [0]], dtype=np.float32)
        self.predicted_position = position
        self.lost_count = 0

    def predict(self):
        # 预测下一帧位置
        prediction = self.kalman.predict()
        x = np.clip(prediction[0], 0, self.frame_size[0])
        y = np.clip(prediction[1], 0, self.frame_size[1])
        self.predicted_position = (int(x), int(y))
        return self.predicted_position

    def update(self, measurement):
        # 用实际测量值更新滤波器
        self.kalman.correct(np.array([[np.float32(measurement[0])], 
                                    [np.float32(measurement[1])]]))
        self.predicted_position = (int(measurement[0]), int(measurement[1]))
        self.lost_count = 0

def preprocess_binary(binary, min_area=10000, max_size_ratio=0.8, min_size_ratio=0.1):
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
    failed_contours = []
    areas = []
    for cnt in contours:
        # 过滤小面积
        area = cv2.contourArea(cnt)
        if area < min_area:
            failed_contours.append(cnt)
            continue
            
        # 过滤过大物体
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > max_w or bh > max_h or \
            (bw < min_w and bh < min_h):
            failed_contours.append(cnt)
            continue
            
        filtered_contours.append(cnt)
        areas.append(area)
        
    return filtered_contours, failed_contours, areas

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

def track(frame_loader, save_file, camera_params):
    tracked_objects = []
    next_id = 1
    max_lost = 10  # 最大允许丢失帧数
    min_tracked_count = 5  # 最小跟踪帧数

    frame = next(frame_loader)
    shape = frame.shape[:2]
    fcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_file, fcc, 25, shape[::-1])

    # detector = BaseDetector(True)
    detector = SamDetector()
    corrector = GeometricCorrector(shape[0], shape[1],camera_params)
    tracker = SortTracker(corrector, max_lost)

    for frame in frame_loader:
        start = time.perf_counter()
        show_f = frame.copy()
        binary = detector.segment(frame, 0.3)
        print('---binary', binary ,binary.shape)
        cv2.imshow("binary", binary)
        cv2.waitKey(1)

        # 预处理：过滤小目标和大目标
        valid_contours, invalid_contours, areas = preprocess_binary(binary, min_area=2000)
        centroids = get_centroids(valid_contours)

        for cnt in valid_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(show_f, (x, y), (x+w, y+h), (0, 128, 128), 10)
        
        cv2.drawContours(show_f, invalid_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(show_f, valid_contours, -1, (0, 255, 0), 2)
        for cnt, area in zip(valid_contours, areas):
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.putText(show_f, f"area:{area}", 
                       (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 2)


        # 绘制有效目标的质心
        centroids = get_centroids(valid_contours)
        for (cx, cy) in centroids:
            cv2.circle(show_f, (cx, cy), 5, (255, 0, 0), -1)

        tracked_objects = tracker.update(frame, valid_contours, centroids)
        print('----tra', tracked_objects)
        # 绘制结果
        for obj in tracked_objects:
            if obj.tracked_count < min_tracked_count:
                continue
            # 绘制预测位置（蓝色）
            cv2.circle(show_f, obj.predicted_position, 5, (255, 0, 0), 2)
            # 绘制实际位置（绿色）
            cv2.circle(show_f, obj.measurement_position, 7, (0, 255, 0), 2)
            cv2.putText(show_f, f"ID:{obj.object_id}:{obj.speed:.2f}", 
                       (obj.predicted_position[0], obj.predicted_position[1]+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 2)

        rshow_f = cv2.resize(show_f, None, fx=0.3, fy=0.3)
        cv2.imshow("Tracking", rshow_f)
        out.write(show_f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.perf_counter()
        print(f"---time: {end-start}")

    cv2.destroyAllWindows()
    out.release()

