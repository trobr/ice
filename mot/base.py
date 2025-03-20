
import math
import cv2
import numpy as np
from collections import deque


class TrackedObjectOld:
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
        self.measurement_position = position
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
        self.measurement_position = (int(measurement[0]), int(measurement[1]))
        self.lost_count = 0

    @property
    def speed(self):
        return 0


class TrackedObjectWithSpeed:
    def __init__(self, object_id, position, frame_size, fps=25):
        self.object_id = object_id
        self.kalman = cv2.KalmanFilter(4, 2)  # 状态4（x,y,dx,dy），观测2（x,y）
        self.frame_size = frame_size  # (width, height)
        self.fps = fps
        
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
        self.speed_history = deque(maxlen=self.fps)  # 保存最近1秒的速度记录
        self.velocity = (0.0, 0.0)  # (vx, vy) 单位：像素/秒

        self.predicted_position = position
        self.measurement_position = position
        self.lost_count = 0

    def predict(self):
        """ 预测目标位置并更新速度 """
        prediction = self.kalman.predict()
        
        # 约束位置在画面范围内
        x = np.clip(prediction[0], 0, self.frame_size[0])
        y = np.clip(prediction[1], 0, self.frame_size[1])
        
        # 从预测状态获取速度（单位：像素/帧）
        dx = prediction[2][0]
        dy = prediction[3][0]
        
        # 转换为像素/秒并保存历史
        self.velocity = (dx * self.fps, dy * self.fps)
        self.predicted_position = (int(x), int(y))
        return (int(x), int(y))

    def update(self, measurement):
        """ 更新测量值并重新计算速度 """
        # 转换为numpy数组格式
        measurement = np.array([[np.float32(measurement[0])], 
                               [np.float32(measurement[1])]])
        self.measurement_position = (int(measurement[0]), int(measurement[1]))
        self.lost_count = 0

        # 更新卡尔曼滤波器
        self.kalman.correct(measurement)
        
        # 从更新后的状态获取速度（单位：像素/帧）
        state = self.kalman.statePost
        dx = state[2][0]
        dy = state[3][0]
        
        # 转换为像素/秒
        current_speed = np.sqrt((dx*self.fps)**2 + (dy*self.fps)**2)
        self.speed_history.append(current_speed)
        
        # 更新瞬时速度矢量
        self.velocity = (dx * self.fps, dy * self.fps)
        
        # 返回更新后的位置
        return (int(state[0][0]), int(state[1][0]))

    @property
    def speed(self):
        """ 获取最近1秒的平均速度（标量）"""
        if len(self.speed_history) == 0:
            return 0.0
        return np.mean(self.speed_history)
    
    @property
    def velocity_vector(self):
        """ 获取当前速度矢量 (vx, vy) """
        return self.velocity

# ---------------------- 几何校正模块 ----------------------
class GeometricCorrector:
    def __init__(self, img_height, img_width, camera_params):
        """
        几何校正核心类
        :param img_height: 图像高度 (像素)
        :param img_width: 图像宽度 (像素)
        :param camera_params: 包含以下参数的字典
            - height: 相机高度 (米)
            - focal_length: 等效焦距 (米)
            - incline_angle: 镜头倾角 (度数)
            - photo_size: 等效35mm照片尺寸 (米) [宽度, 高度]
        """
        self.m = img_height
        self.n = img_width
        self.params = {
            'height': camera_params['height'],
            'focal_length': camera_params['focal_length'],
            'incline_angle': math.radians(camera_params['incline_angle']),
            'photo_size': camera_params['photo_size']
        }
        
        # 预计算行列分辨率
        self.row_res = self.params['photo_size'][0] / self.m
        self.col_res = self.params['photo_size'][1] / self.n
        
        # 预生成校正查找表
        self.deltaX_table = np.zeros(self.m)
        self.deltaY_table = np.zeros(self.m)
        self._precompute_correction()

    def _precompute_correction(self):
        """ 预计算每行的物理尺寸 """
        for i in range(self.m):
            y_pixel = (self.m/2 - i) * self.row_res
            focal = self.params['focal_length']
            angle = self.params['incline_angle']
            
            # 计算Y方向物理尺寸 (米/像素)
            denominator = math.cos(angle + math.atan(y_pixel/focal))**2 * (1 + (y_pixel/focal)**2)
            self.deltaY_table[i] = (self.row_res * self.params['height'] / focal) / denominator
            
            # 计算X方向物理尺寸 (米/像素)
            yy = self.params['height'] * math.tan(angle + math.atan(y_pixel/focal))
            self.deltaX_table[i] = self.col_res * math.sqrt(
                (self.params['height']**2 + yy**2) / 
                (focal**2 + y_pixel**2))
    
    def pixel_to_meter(self, x_pixel, y_pixel):
        """
        将像素坐标转换为物理坐标 (米)
        :param x_pixel: 像素X坐标 (0-based)
        :param y_pixel: 像素Y坐标 (0-based)
        :return: (x_meter, y_meter) 物理坐标
        """
        y = int(np.clip(y_pixel, 0, self.m-1))
        return (
            x_pixel * self.deltaX_table[y],
            y_pixel * self.deltaY_table[y]
        )

# ---------------------- 跟踪对象模块 ----------------------
class TrackedObjectWithRealSpeed:
    def __init__(self, object_id, pixel_pos, frame_size, corrector, fps=25):
        """
        跟踪对象类（物理单位版本）
        :param object_id: 目标唯一ID
        :param pixel_pos: 初始像素位置 (x, y)
        :param corrector: GeometricCorrector实例
        :param fps: 视频帧率
        """
        self.object_id = object_id
        self.corrector = corrector
        self.fps = fps
        
        # 转换初始位置到物理坐标
        init_x, init_y = corrector.pixel_to_meter(*pixel_pos)
        
        # 初始化卡尔曼滤波器（状态：x,y,vx,vy）
        self.kalman = cv2.KalmanFilter(4, 2)
        self._init_kalman(init_x, init_y)
        
        # 速度跟踪相关
        self.speed_history = deque(maxlen=fps)  # 保存1秒内的速度记录
        self.velocity = (0.0, 0.0)  # 当前速度 (m/s)

        self.predicted_position = pixel_pos
        self.measurement_position = pixel_pos
        self.lost_count = 0
        self.tracked_count = 0

    def _init_kalman(self, init_x, init_y):
        """ 初始化卡尔曼滤波器参数 """
        # 状态转移矩阵 (匀速模型)
        self.kalman.transitionMatrix = np.array([
            [1,0,1/self.fps,0],
            [0,1,0,1/self.fps],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=np.float32)
        
        # 观测矩阵
        self.kalman.measurementMatrix = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ], dtype=np.float32)
        
        # 协方差矩阵
        self.kalman.processNoiseCov = 1e-4 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 1e-2 * np.eye(4, dtype=np.float32)
        
        # 初始状态
        self.kalman.statePost = np.array([
            [init_x],
            [init_y],
            [0],  # 初始速度vx
            [0]   # 初始速度vy
        ], dtype=np.float32)

    def predict(self):
        """ 预测下一时刻状态（物理坐标） """
        prediction = self.kalman.predict()
        self.predicted_position = self.measurement_position
        return (prediction[0][0], prediction[1][0])

    def update(self, pixel_measurement):
        """
        更新测量值（输入为像素坐标）
        :param pixel_measurement: (x_pixel, y_pixel)
        """
        self.measurement_position = (int(pixel_measurement[0]), int(pixel_measurement[1]))
        self.lost_count = 0
        self.tracked_count += 1

        # 转换为物理坐标
        x_meter, y_meter = self.corrector.pixel_to_meter(*pixel_measurement)
        
        # 更新卡尔曼滤波器
        measurement = np.array([[x_meter], [y_meter]], dtype=np.float32)
        self.kalman.correct(measurement)
        
        # 从状态获取速度
        state = self.kalman.statePost
        vx = state[2][0]
        vy = state[3][0]
        
        # 记录速度
        speed = np.sqrt(vx**2 + vy**2)
        self.speed_history.append(speed)
        self.velocity = (vx, vy)
        
        return (state[0][0], state[1][0])

    @property
    def speed(self):
        """ 获取平均速度（m/s） """
        if not self.speed_history:
            return 0.0
        return np.mean(self.speed_history)
    
    @property
    def velocity_vector(self):
        """ 获取当前速度矢量 (vx, vy) m/s """
        return self.velocity


TrackedObject = TrackedObjectWithRealSpeed
