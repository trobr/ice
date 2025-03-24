import cv2
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
from haversine import haversine, Unit

class ShipDataLoader:
    def __init__(self, data_path, video_path, start_time=None, fps_target=1):
        """
        :param data_path: 船位数据文件路径
        :param video_path: 视频文件路径
        :param start_time: 视频开始时间（可选）
        """
        # 加载并预处理船位数据
        self.df = self._load_and_preprocess_data(data_path)
        
        # 初始化视频加载器
        self.video_generator = self._video_loader(video_path, fps_target)

        # 插值后的秒级数据迭代器
        self.secondly_data = self._generate_secondly_data()

        # 第一秒为0
        self.data_time, self.ship_data = next(self.secondly_data)
        self.data_time, self.ship_data = next(self.secondly_data)
        
        # 时间相关参数
        self.start_time = start_time or self.interpolated_df.index[1]
        self.current_time = self.start_time
        

    def _load_and_preprocess_data(self, path):
        """加载并预处理原始船位数据"""
        df = pd.read_csv(path, parse_dates={'datetime': ['date', 'time_utc']})
        df = df.sort_values('datetime').set_index('datetime')
        return df

    def _interpolate_seconds(self):
        """使用线性插值生成秒级数据"""
        # 生成完整的时间索引（秒级）
        full_idx = pd.date_range(
            start=self.df.index[0],
            end=self.df.index[-1],
            freq='S'
        )
        
        # 重新采样并插值
        return self.df.reindex(full_idx).interpolate(method='time')

    def _calculate_speed(self, df):
        """计算船速（米/秒）"""
        speeds = []
        prev_row = None
        
        for _, row in df.iterrows():
            if prev_row is not None:
                # 使用haversine公式计算距离（米）
                distance = haversine(
                    (prev_row.latitude, prev_row.longitude),
                    (row.latitude, row.longitude),
                    unit=Unit.METERS
                )
                # 时间差（秒）
                time_diff = (row.name - prev_row.name).total_seconds()
                speed = distance / time_diff if time_diff > 0 else 0.0
            else:
                speed = 0.0
                
            speeds.append(speed)
            prev_row = row
            
        df['speed'] = speeds
        return df

    def _generate_secondly_data(self):
        """生成秒级数据生成器"""
        self.interpolated_df = self._interpolate_seconds()
        speed_df = self._calculate_speed(self.interpolated_df)
        
        for timestamp, row in speed_df.iterrows():
            yield timestamp, row
    
    def _video_loader(self, path, fps_target=1):
        print('----fps_target', fps_target)
        cap = cv2.VideoCapture(path)
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(round(fps_src / fps_target)))  # 精确计算跳帧步长
        print('---step', step)
        
        frame_pos = 0  # 记录原始帧位置
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 计算精确时间戳
            timestamp = self.start_time + timedelta(seconds=frame_pos/fps_src)
            
            # 跳帧并累计时间
            for _ in range(step-1):
                cap.grab()
                frame_pos += 1
            
            yield timestamp, cv2.resize(frame, None, fx=0.5, fy=0.5)
            frame_pos += 1  # 处理当前帧
        
        cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # 遍历后续视频帧寻找最佳匹配
            video_time, frame = next(self.video_generator)
            current_diff = abs((video_time - self.data_time).total_seconds())
            print('---current_diff', video_time, self.data_time, current_diff)
            
            # 找到时间差最小的帧
            if current_diff >= 1:
                self.data_time, self.ship_data = next(self.secondly_data)
            if current_diff > 10:
                raise RuntimeError("视频时间与船位数据时间差过大")
                    
            return {
                'timestamp': self.data_time,
                'latitude': self.ship_data.latitude,
                'longitude': self.ship_data.longitude,
                'speed': self.ship_data.speed,
                'frame': frame,
            }
        except StopIteration:
            raise StopIteration

class EnhancedShipDataLoader(ShipDataLoader):
    def _interpolate_seconds(self):
        """增强版时间插值方法，支持任意时间间隔"""
        # 转换为Unix时间戳（秒级精度）
        timestamps = self.df.index.view('int64') // 10**9
        lat = self.df.latitude.values
        lon = self.df.longitude.values

        # 创建插值函数（使用球面线性插值）
        lat_interp = interp1d(timestamps, lat, kind='linear', 
                            fill_value="extrapolate")
        lon_interp = interp1d(timestamps, lon, kind='linear',
                            fill_value="extrapolate")

        # 生成目标时间序列（秒级）
        start_ts = timestamps[0]
        end_ts = timestamps[-1]
        target_ts = np.arange(start_ts, end_ts+1, dtype=int)

        # 执行插值
        interpolated_df = pd.DataFrame({
            'timestamp': pd.to_datetime(target_ts, unit='s'),
            'latitude': lat_interp(target_ts),
            'longitude': lon_interp(target_ts)
        }).set_index('timestamp')

        return interpolated_df

    # def _calculate_speed(self, df):
    #     """改进版速度计算（考虑地球曲率）"""
    #     from geographiclib.geodesic import Geodesic
        
    #     coords = df[['latitude', 'longitude']].values
    #     speeds = [0.0]  # 初始速度为0
        
    #     for i in range(1, len(coords)):
    #         g = Geodesic.WGS84.Inverse(*coords[i-1], *coords[i])
    #         distance = g['s12']  # 单位：米
    #         time_diff = (df.index[i] - df.index[i-1]).total_seconds()
    #         speeds.append(distance / time_diff if time_diff > 0 else 0.0)
            
    #     df['speed'] = speeds
    #     return df


# 使用示例
if __name__ == "__main__":
    # 初始化数据加载器
    loader = EnhancedShipDataLoader(
        data_path='/Users/trobr/Downloads/Track_gps.csv',
        video_path='/Users/trobr/Downloads/D03_20240910083145.mp4',
        fps_target=5,
    )
    
    # 迭代获取每秒数据
    for data in loader:
        print(f"时间: {data['timestamp']}")
        print(f"纬度: {data['latitude']:.6f}, 经度: {data['longitude']:.6f}")
        print(f"航速: {data['speed']:.2f} m/s")
        cv2.imshow('Video', data['frame'])
        if cv2.waitKey(0) == 27:
            break
