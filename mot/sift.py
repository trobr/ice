import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from mot.base import TrackedObject

class SIFTMultiObjectTracker:
    def __init__(self, min_matches=15):
        self.sift = cv2.SIFT_create()
        # FLANN参数和匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.tracked_objects = []  # 跟踪中的目标列表
        self.next_id = 0            # 下一个可用的目标ID
        self.min_matches = min_matches  # 最小匹配数量阈值

    def update(self, frame, valid_contours, centroids, mask=None):
        detections = []
        for cnt in valid_contours:
            detections.append(cv2.boundingRect(cnt))

        # 提取当前帧的检测特征
        current_features = []
        for bbox in detections:
            x, y, w, h = [int(v) for v in bbox]
            roi = frame[y:y+h, x:x+w]
            
            # 在ROI中提取SIFT特征
            kp, des = self.sift.detectAndCompute(roi, mask)
            
            # 转换关键点坐标到原图坐标系
            if kp is not None:
                for p in kp:
                    p.pt = (p.pt[0] + x, p.pt[1] + y)
            
            current_features.append({
                'kp': kp,
                'des': des,
                'bbox': (x, y, w, h)
            })

        # 初始化阶段：直接添加所有检测为目标
        if not self.tracked_objects:
            for feat in current_features:
                self.tracked_objects.append({
                    'id': self.next_id,
                    'bbox': feat['bbox'],
                    'kp': feat['kp'],
                    'des': feat['des']
                })
                self.next_id += 1
            return self._get_tracking_results(frame)

        # 构建成本矩阵（使用负匹配数量作为成本）
        cost_matrix = np.zeros((len(self.tracked_objects), len(current_features)))
        for i, obj in enumerate(self.tracked_objects):
            if obj['des'] is None or len(obj['des']) < 2:
                continue
            for j, curr in enumerate(current_features):
                if curr['des'] is None or len(curr['des']) < 2:
                    continue
                
                # 特征匹配
                matches = self.flann.knnMatch(obj['des'], curr['des'], k=2)
                
                # 应用比率测试
                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)
                
                # 记录匹配数量（成本为负匹配数）
                cost_matrix[i, j] = -len(good) if len(good) > self.min_matches else 0

        # 匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < -self.min_matches:
                matches.append((r, c))

        # 更新匹配的目标
        updated_objects = []
        matched_rows = set()
        matched_cols = set()
        for r, c in matches:
            # 更新目标位置和特征
            self.tracked_objects[r]['bbox'] = current_features[c]['bbox']
            self.tracked_objects[r]['kp'] = current_features[c]['kp']
            self.tracked_objects[r]['des'] = current_features[c]['des']
            updated_objects.append(self.tracked_objects[r])
            matched_rows.add(r)
            matched_cols.add(c)

        # 添加未匹配的新检测
        for j, feat in enumerate(current_features):
            if j not in matched_cols:
                updated_objects.append({
                    'id': self.next_id,
                    'bbox': feat['bbox'],
                    'kp': feat['kp'],
                    'des': feat['des']
                })
                self.next_id += 1

        self.tracked_objects = updated_objects
        return self._get_tracking_results(frame)

    def _get_tracking_results(self, frame):
        frame_size = frame.shape[1], frame.shape[0]
        
        return [TrackedObject(
            obj['id'], obj['bbox'][:2], frame_size) for obj in self.tracked_objects]

# 使用示例
if __name__ == "__main__":
    tracker = SIFTMultiObjectTracker()
    
    # 模拟输入（需要实际视频帧和检测结果）
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_detections = [(100, 100, 50, 50), (200, 200, 60, 60)]
    
    results = tracker.update(dummy_frame, dummy_detections)
    print("Tracking results:", results)
