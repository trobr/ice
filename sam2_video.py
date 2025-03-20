# import torch
# from sam2.build_sam import build_sam2_video_predictor

# checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# import os
# print(os.path.exists(checkpoint))
# print(os.path.exists(model_cfg))

# predictor = build_sam2_video_predictor(model_cfg, checkpoint, device='cpu')

# # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
# with torch.inference_mode():
#     state = predictor.init_state('/Users/trobr/Downloads/ice1/D03_20250202121012.mp4_20250305_143053.mp4')

#     # add new prompts and instantly get the output on the same frame
#     frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, 'Segment individual ice floes floating on the water surface. The ice floes appear bright and white, contrasting with the darker water background. Focus on accurately detecting and isolating separate ice chunks while preserving their natural shapes. Ignore reflections and minor noise. Ensure clear boundaries between ice and water.')

#     # propagate the prompts to get masklets throughout the video
#     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         print(frame_idx, object_ids, masks)

# mac ModuleNotFoundError: No module named 'decord'


import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def show_anns_cv2(anns, image, borders=True):
    if len(anns) == 0:
        return
    
    # 缩小原图为一半大小
    image_resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    for ann in sorted(anns, key=lambda x: x['area'], reverse=True):
        mask = ann['segmentation'].astype(np.uint8)
        mask_resized = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))
        
        # 生成随机颜色
        color_mask = np.random.randint(0, 255, (1, 3), dtype=np.uint8)[0]
        
        # 叠加掩码 (半透明)
        overlay = image_resized.copy()
        overlay[mask_resized > 0] = (
            overlay[mask_resized > 0] * 0.5 + color_mask * 0.5
        ).astype(np.uint8)

        # 画轮廓
        if borders:
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01 * cv2.arcLength(contour, True), closed=True) for contour in contours]
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)  # 红色轮廓

        # 显示图像
        cv2.imshow('Segmented Mask', overlay)
        if cv2.waitKey(0) & 0xFF == 27:  # 按 `Esc` 退出
            break

    cv2.destroyAllWindows()

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2 = build_sam2(model_cfg, checkpoint, device='cpu')
mask_generator = SAM2AutomaticMaskGenerator(sam2)

image = cv2.imread('/Users/trobr/Downloads/flow/5/DSCN0600.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with torch.inference_mode():
    masks = mask_generator.generate(image)
    show_anns_cv2(masks, image)
    # plt.figure(figsize=(20, 20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show() 

