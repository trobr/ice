import cv2

from segment_anything_comfy import GroundingDinoModelLoader, SAMModelLoader, GroundingDinoSAMSegment
from utils import cv2_img_to_tensor, tensor_to_cv2_img


class SamDetector(object):
    def __init__(self, dino_model_name='GroundingDINO_SwinB (938MB)', sam_model_name='sam_vit_h (2.56GB)'):
        dino_loader = GroundingDinoModelLoader()
        sam_loader = SAMModelLoader()

        self.dino_model, *_ = dino_loader.main(dino_model_name)
        self.sam_model, *_ = sam_loader.main(sam_model_name)
        self.segmenter = GroundingDinoSAMSegment()

    def segment(self, img, threshold=0.9):
        img = cv2_img_to_tensor(img)
        res, mask = self.segmenter.main(self.dino_model, self.sam_model, img, 'floating ice.high light.white', threshold)
        return tensor_to_cv2_img(mask)[0][..., 0]
