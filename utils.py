import cv2
import torch
import numpy as np

from segment import adaptive_threshold_blocks, histogram_stretching
from dehaze import dehaze


def cv2_img_to_tensor(img):
    """
    将OpenCV图像转换为PyTorch张量
    
    参数：
    img (numpy.ndarray): OpenCV图像 [h, w, c] bgr
    
    返回：
    torch.Tensor: PyTorch张量 [b, c, h, w]
    """
    # torch.from_numpy(cv2.cvtColor(immm, cv2.COLOR_BGR2RGB).astype(np.float32)/255.)[None,]
    # 将图像从BGR转换为RGB
    # # img = histogram_stretching(img)
    # img = dehaze(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将numpy数组转换为PyTorch张量
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)[None, ]
    return tensor


def tensor_to_cv2_img(tensor):
    """
    将PyTorch张量转换为OpenCV图像
    
    参数：
    tensor (torch.Tensor): PyTorch张量 [b, c, h, w]
    
    返回：
    List[numpy.ndarray]: OpenCV图像 List[[h, w, c]] bgr
    """
    # 将张量转换为numpy数组
    result = []
    for t in tensor:
        img = np.clip(255. * t.cpu().numpy(), 0, 255).astype(np.uint8)
        # 将RGB转换为BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result.append(img)
    return result

