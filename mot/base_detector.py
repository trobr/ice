import cv2
from segment import adaptive_threshold_blocks, histogram_stretching
from dehaze import dehaze


class BaseDetector(object):
    def __init__(self, enable_dehaze=False):
        self.enable_dehaze = enable_dehaze

    def segment(self, frame, min_threshold=180):
        frame = histogram_stretching(frame)
        if self.enable_dehaze:
            frame = dehaze(frame)
        return adaptive_threshold_blocks(frame, min_threshold=min_threshold)
