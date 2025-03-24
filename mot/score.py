import cv2
import numpy as np
from skimage import measure
from scipy.stats import entropy
from scipy.stats import entropy, norm
from sklearn.mixture import GaussianMixture
from skimage import feature, measure


class MultiMaskEvaluator:
    def __init__(self, image):
        self.image = image

    def _create_mask_from_contour(self, contour):
        """根据单个轮廓生成掩膜"""
        mask = np.zeros_like(self.image, dtype=np.uint8)
        # mask = np.zeros(self.image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return mask
    
    # ----------------- 核心评分方法 -----------------
    def score_s1(self, mask):
        """基于当前掩膜的双峰性评分"""
        foreground = self.image[mask == 255]
        background = self.image[mask == 0]
        if len(foreground) == 0 or len(background) == 0:
            return 0.0
        
        ω_f = len(foreground) / self.image.size
        ω_b = len(background) / self.image.size
        μ_diff = np.mean(foreground) - np.mean(background)
        var_b = ω_f * ω_b * μ_diff**2
        var_total = np.var(self.image) / 10
        return min(var_b / var_total, 1.0) if var_total != 0 else 0.0

    def score_s2(self):
        """全局对比度评分（与掩膜无关）"""
        return np.std(self.image) / 127.5

    def score_s3(self, mask):
        """亮度差异评分"""
        print('---mask', mask.shape, self.image.shape, mask.dtype, self.image.dtype)
        mean_fg = cv2.mean(self.image, mask=mask[..., 0])[0]
        mean_bg = cv2.mean(self.image, mask=cv2.bitwise_not(mask[..., 0]))[0]
        return max(0.0, (mean_fg - mean_bg) / 255.0)

    def score_s4(self, mask, local_std_threshold=10):
        """雾化误分割评分"""
        kernel = np.ones((3,3), np.float32)/9
        local_mean = cv2.filter2D(self.image, -1, kernel)
        local_sq_mean = cv2.filter2D(self.image**2, -1, kernel)
        local_std = np.sqrt(np.clip(local_sq_mean - local_mean**2, 0, None))
        
        fg_area = np.sum(mask == 255)
        if fg_area == 0:
            return 0.0
        low_contrast = np.sum((mask == 255) & (local_std < local_std_threshold))
        return 1.0 - (low_contrast / fg_area)

    def score_s6(self, mask):
        """区域连通性评分（改进版）"""
        labels = measure.label(mask, connectivity=2)
        regions = measure.regionprops(labels)
        
        if not regions:
            return 0.0
        
        areas = [r.area for r in regions]
        max_area = max(areas)
        total_area = sum(areas)
        return (max_area / total_area) * np.sqrt(len(areas)/100.0)
    
    def score_s7(self, mask):
        """局部对比度一致性（动态阈值）"""
        fg = self.image[mask == 255]
        bg = self.image[mask == 0]
        if len(fg) < 10 or len(bg) < 10:
            return 0.0
        
        fg_std = np.std(fg)
        bg_std = np.std(bg)
        return np.clip((fg_std - bg_std)/50.0, 0.0, 1.0)
    
    def score_s8(self, mask):
        """信息熵差异评分（改进版）"""
        fg = self.image[mask == 255].astype(np.uint8)
        bg = self.image[mask == 0].astype(np.uint8)
        
        hist_fg = cv2.calcHist([fg], [0], None, [256], [0,256])
        hist_bg = cv2.calcHist([bg], [0], None, [256], [0,256])
        
        eps = 1e-10
        h_fg = entropy(hist_fg.flatten() + eps)
        h_bg = entropy(hist_bg.flatten() + eps)
        return np.abs(h_fg - h_bg) / 8.0
    
    def score_s9(self, mask):
        """亮度分布拟合评分（高斯混合模型）"""
        try:
            data = self.image.flatten().reshape(-1,1)
            gmm = GaussianMixture(n_components=2, covariance_type='diag')
            gmm.fit(data)
            
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            if len(means) < 2:
                return 0.0
                
            # 确保正确的排序
            if means[0] > means[1]:
                mu_fg, mu_bg = means[0], means[1]
                sigma_fg, sigma_bg = stds[0], stds[1]
            else:
                mu_fg, mu_bg = means[1], means[0]
                sigma_fg, sigma_bg = stds[1], stds[0]
                
            # 计算分离度
            separation = (mu_fg - mu_bg) / (sigma_fg + sigma_bg)
            return np.clip(separation / 3.0, 0.0, 1.0)  # 3σ原则
        except:
            return 0.0

    def score_s10(self, contour):
        """形状凸性评分"""
        # 基础凸性计算
        hull = cv2.convexHull(contour)
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(hull)
        
        # 防止除零错误
        if hull_area < 1e-6:
            return 0.0
        
        # 基础凸性得分
        convex_base = np.clip(contour_area / hull_area, 0.0, 1.0)
        
        # 凹陷深度分析
        max_defect = 0
        if len(contour) > 3:  # 至少需要4个点才能形成凹陷
            defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
            if defects is not None:
                for i in range(defects.shape[0]):
                    _, _, farthest_idx, defect_depth = defects[i, 0]
                    max_defect = max(max_defect, defect_depth/256.0)  # OpenCV返回值为256倍实际距离
        
        # 综合评分
        depth_penalty = max(0, 1 - max_defect/100.0)  # β=20
        print('---depth_penalty', depth_penalty, convex_base, max_defect)
        return (convex_base ** 0.5) * depth_penalty

    def mask_nms(self, results, iou_threshold=0.5, score_threshold=0.3):
        """
        基于掩膜IoU的非极大值抑制
        :param results: evaluate_contours返回的结果列表（需已按total_score降序排序）
        :param iou_threshold: 重叠阈值，大于此值则抑制低分区域
        :param score_threshold: 最低保留分数阈值
        :return: 经过筛选的结果列表
        """
        keep = []
        suppressed = set()

        # 预处理：确保结果已按分数降序排列
        sorted_results = sorted(results, key=lambda x: -x['total_score'])

        for i in range(len(sorted_results)):
            if i in suppressed:
                continue
            
            current = sorted_results[i]
            if current['total_score'] < score_threshold:
                continue
                
            keep.append(current)
            
            # 计算与后续所有候选的IoU
            for j in range(i+1, len(sorted_results)):
                if j in suppressed:
                    continue
                    
                # 加速计算：优先检查外接矩形是否重叠
                x1, y1, w1, h1 = cv2.boundingRect(current['contour'])
                x2, y2, w2, h2 = cv2.boundingRect(sorted_results[j]['contour'])
                
                # 计算矩形IoU作为快速筛选
                rect_iou = self._box_iou((x1,y1,x1+w1,y1+h1), (x2,y2,x2+w2,y2+h2))
                if rect_iou < 0.1:  # 如果矩形IoU过低则跳过精确计算
                    continue
                    
                # 精确计算掩膜IoU
                # mask_iou = self._mask_iou(current['mask'], sorted_results[j]['mask'])
                if rect_iou > iou_threshold:
                    suppressed.add(j)

        return keep

    def _box_iou(self, box1, box2):
        """计算两个矩形框的IoU（用于快速筛选）"""
        # 转换格式：x1,y1,x2,y2
        b1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
        b2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]
        
        # 计算交集区域
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return intersection_area / (area1 + area2 - intersection_area)

    def _mask_iou(self, mask1, mask2):
        """精确计算两个二值掩膜的IoU"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou if not np.isnan(iou) else 0.0

    # ----------------- 完整评分流程 -----------------
    def evaluate_contours(self, valid_contours, weights=None, nms_iou=0.5, score_threshold=0.1):
        """
        对多个轮廓进行评分
        :param valid_contours: 轮廓列表，每个元素为OpenCV格式的轮廓
        :param weights: 各评分的权重字典，默认等权重
        :return: 包含每个轮廓评分的字典列表
        """
        results = []
        
        # 预计算全局对比度（所有掩膜共享）
        # s2 = self.score_s2()
        
        for contour in valid_contours:
            mask = self._create_mask_from_contour(contour)
            scores = {
                'S1': self.score_s1(mask),
                # # 'S2': s2,  # 全局评分复用
                # 'S3': self.score_s3(mask),
                # # 'S4': self.score_s4(mask),
                # # 'S6': self.score_s6(mask),
                # # 'S7': self.score_s7(mask),
                # 'S8': self.score_s8(mask),
                # # 'S9': self.score_s9(mask),
                'S10': self.score_s10(contour),
            }
            
            # 计算总分
            if weights is None:
                total = np.mean(list(scores.values()))
            else:
                total = sum(scores[k]*weights[k] for k in scores) / sum(weights.values())

            print('---score', scores, total)

            # 保存结果（包含掩膜和评分）
            results.append({
                # 'mask': mask,
                'total_score': total,
                'detail_scores': scores,
                'contour': contour
            })
        
        # 按总分排序
        results = sorted(results, key=lambda x: -x['total_score'])
        return self.mask_nms(results, nms_iou, score_threshold)

# 使用示例
if __name__ == "__main__":
    # 1. 加载图像并获取候选轮廓
    image_path = "/Users/trobr/Downloads/flow3.png"
    image = cv2.imread(image_path)

    # 生成候选轮廓（示例使用Otsu+形态学操作）
    _, otsu_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    cleaned_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    evaluator = MultiMaskEvaluator(image, contours)

    # 2. 过滤有效轮廓（示例过滤小区域）
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # 3. 执行评分
    results = evaluator.evaluate_contours(valid_contours)
    
    # 4. 输出结果
    print(f"共评估 {len(results)} 个候选区域")
    for i, res in enumerate(results[:3]):  # 显示前三名
        print(f"\n排名 {i+1} 的区域:")
        print(f"总分: {res['total_score']:.3f}")
        print("详细评分:")
        for k, v in res['detail_scores'].items():
            print(f"{k}: {v:.3f}")
