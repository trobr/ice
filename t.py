import cv2

from segment_anything_comfy import GroundingDinoModelLoader, SAMModelLoader, GroundingDinoSAMSegment
from utils import cv2_img_to_tensor, tensor_to_cv2_img


img = cv2.imread('/Users/trobr/trobr/github/ComfyUI/input/flow3.png')
img = cv2_img_to_tensor(img)


gloader = GroundingDinoModelLoader()
sloader = SAMModelLoader()

dino_model, *_ = gloader.main('GroundingDINO_SwinB (938MB)')
sam_model, *_ = sloader.main('sam_vit_h (2.56GB)')

segmenter = GroundingDinoSAMSegment()
res, mask = segmenter.main(dino_model, sam_model, img, 'floating ice.high light.white', 0.3)

res = tensor_to_cv2_img(res)
mask = tensor_to_cv2_img(mask)

cv2.imshow('res', res[0])
cv2.imshow('mask', mask[0])
cv2.waitKey(0)

