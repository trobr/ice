import cv2
import time
from segment import adaptive_threshold_blocks, histogram_stretching
from dehaze import dehaze


class BaseDetector(object):
    def __init__(self, enable_dehaze=False):
        self.enable_dehaze = enable_dehaze

    def segment(self, frame, min_threshold=180):
        s1 = time.perf_counter()
        frame = histogram_stretching(frame, True)
        s2 = time.perf_counter()
        if self.enable_dehaze:
            frame = dehaze(frame)
        s3 = time.perf_counter()
        res = adaptive_threshold_blocks(frame, min_threshold=min_threshold)
        s4 = time.perf_counter()
        print(f"histogram_stretching: {frame.shape} {s2 - s1:.2f}s, dehaze: {s3 - s2:.2f}s, adaptive_threshold_blocks: {s4 - s3:.2f}s")
        return res
