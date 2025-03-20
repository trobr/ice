import cv2


from mot.base import TrackedObject


class MultiDaSiamRPN:
    def __init__(self, model_config):
        model_config = {
            "model": "path/to/dasiamrpn_model.onnx",
            "kernel_cls1": "path/to/dasiamrpn_kernel_cls1.onnx",
            "kernel_r1": "path/to/dasiamrpn_kernel_r1.onnx"
        }
        self.trackers = []  # 存储(跟踪器实例, 目标ID, 是否激活)
        self.next_id = 1
        self.model_config = model_config  # 模型文件路径配置
    
    def add_target(self, frame, bbox):
        """ 新增跟踪目标 """
        tracker = cv2.TrackerDaSiamRPN_create(**self.model_config)
        tracker.init(frame, bbox)
        self.trackers.append( (tracker, self.next_id, True) )
        self.next_id +=1
    
    def update(self, frame):
        """ 更新所有跟踪器 """
        frame_size = frame.shape[1], frame.shape[0]
        results = []
        active_trackers = []
        
        for tracker, obj_id, _ in self.trackers:
            success, bbox = tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cx = x + w//2
                cy = y + h//2
                results.append(TrackedObject(obj_id, (cx, cy), frame_size) )
                active_trackers.append( (tracker, obj_id, True) )
            else:
                # 跟踪失败，保留历史记录（可选）
                active_trackers.append( (tracker, obj_id, False) )
        
        # 更新有效跟踪器列表
        self.trackers = active_trackers
        return results

    def remove_inactive(self, max_failures=5):
        """ 移除连续跟踪失败的目标 """
        clean_trackers = []
        for tracker, obj_id, active in self.trackers:
            if active:
                clean_trackers.append( (tracker, obj_id, active) )
            else:
                # 统计连续失败次数（需扩展状态记录）
                pass  # 实现略
        self.trackers = clean_trackers
