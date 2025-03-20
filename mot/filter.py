import cv2
import numpy as np
from collections import deque

class BaseStabilizer:
    """统一接口基类"""
    def __init__(self):
        pass
    
    def update(self, box: tuple) -> tuple:
        """输入格式: (x, y, w, h)"""
        raise NotImplementedError

# ---------------------------- 实现类 ----------------------------

class MovingAverageStabilizer(BaseStabilizer):
    """滑动平均滤波器"""
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def update(self, box):
        self.history.append(box)
        if len(self.history) == 0:
            return box
        return tuple(np.mean(self.history, axis=0).astype(int).tolist())

class EWMAStabilizer(BaseStabilizer):
    """指数加权移动平均滤波器"""
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.smoothed = None
        
    def update(self, box):
        if self.smoothed is None:
            self.smoothed = np.array(box, dtype=np.float32)
        else:
            self.smoothed = self.alpha * np.array(box) + (1 - self.alpha) * self.smoothed
        return tuple(self.smoothed.astype(int).tolist())

class KalmanStabilizer(BaseStabilizer):
    """卡尔曼滤波器（仅稳定中心坐标，保持宽高不变）"""
    def __init__(self, process_noise: float = 1e-4, meas_noise: float = 1e-1):
        super().__init__()
        # 初始化卡尔曼滤波器（状态：x, y, dx, dy）
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1,0,1,0], 
            [0,1,0,1], 
            [0,0,1,0], 
            [0,0,0,1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov = meas_noise * np.eye(2, dtype=np.float32)
        self.prev_box = None  # 保存宽高
        
    def update(self, box):
        x, y, w, h = box
        # 第一次初始化
        if self.prev_box is None:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.prev_box = (w, h)
            return (x, y, w, h)
        
        # 预测
        prediction = self.kf.predict()
        # 校正（使用当前检测中心）
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        # 保持宽高不变
        return (int(prediction[0][0]), int(prediction[1][0]), self.prev_box[0], self.prev_box[1])

class HybridStabilizer(BaseStabilizer):
    """组合滤波器（先EWMA再卡尔曼）"""
    def __init__(self, alpha=0.5, process_noise=1e-4):
        super().__init__()
        self.ewma = EWMAStabilizer(alpha)
        self.kalman = KalmanStabilizer(process_noise)
        
    def update(self, box):
        smoothed = self.ewma.update(box)
        return self.kalman.update(smoothed)

# ---------------------------- 使用示例 ----------------------------
if __name__ == "__main__":
    # 初始化任意一个滤波器
    stabilizer = KalmanStabilizer()  # 可替换为其他滤波器
    
    # 模拟输入框序列
    test_boxes = [
        (100, 100, 50, 50),
        (102, 101, 50, 50),
        (105, 103, 50, 50),
        (108, 105, 50, 50)
    ]
    
    # 处理每个框
    for box in test_boxes:
        stabilized_box = stabilizer.update(box)
        print(f"Input: {box} => Stabilized: {stabilized_box}")