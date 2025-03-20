import cv2


from mot.base import TrackedObject


class MultiObjectTracker:
    def __init__(self, tracker_type='goturn'):
        self.trackers = []
        self.tracker_types = {
            'siam': cv2.TrackerDaSiamRPN_create,
            'vit': cv2.TrackerVit_create,
            'goturn': cv2.TrackerGOTURN_create,
            # 'KCF': cv2.TrackerKCF_create,
            # 'CSRT': cv2.TrackerCSRT_create,
            # 'MOSSE': cv2.TrackerMOSSE_create
        }
        self.tracker_class = self.tracker_types[tracker_type]

    def update(self, frame, contours, centroids):
        # 初始化新检测目标
        frame_size = frame.shape[1], frame.shape[0]
        new_boxes = [cv2.boundingRect(c) for c in contours]
        for box in new_boxes:
            tracker = self.tracker_class()
            tracker.init(frame, box)
            self.trackers.append(tracker)

        # 更新现有跟踪器
        valid_tracks = []
        results = []
        for idx, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                x,y,w,h = [int(v) for v in box]
                cx = x + w//2
                cy = y + h//2
                results.append(TrackedObject(idx, (cx, cy), frame_size) )
                valid_tracks.append(tracker)
        
        self.trackers = valid_tracks
        return results

