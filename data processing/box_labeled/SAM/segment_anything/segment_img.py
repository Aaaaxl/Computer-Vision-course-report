from segment_anything.build_sam import  *
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# from general_inference import convert
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import json
import time


def convert_tensor_to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
    return img


def ensure_mask_2d(mask):
    """
    确保 mask 为二维数组。如果 mask 有多余的维度，则尝试 squeeze 或取第一通道。
    """
    # 如果 mask 是 Tensor，也转换为 numpy 数组
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # 如果 mask 是布尔类型且 ndim 大于2，尝试 squeeze 或取第一通道
    if mask.ndim > 2:
        # 如果最后一维的大小为1，直接 squeeze
        if mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        else:
            # 否则，假设 mask 的形状为 (C, H, W)，取第一个通道
            mask = mask[0]
    return mask


def visualize_masks(img, masks):
    img = convert_tensor_to_numpy(img)
    # 拷贝一份图像用于绘制
    overlay = img.copy()
    # 随机生成颜色
    colors = []
    for _ in range(len(masks)):
        colors.append([random.randint(0, 255) for _ in range(3)])

        # 对每个 mask 进行叠加显示
        for i, mask_info in enumerate(masks):
            # segmentation 应该是二值 mask。确保其为二维数组。
            mask = mask_info["segmentation"]
            mask = ensure_mask_2d(mask)

            # 检查 mask 是否与图像尺寸匹配
            if mask.shape != overlay.shape[:2]:
                print(f"警告：mask shape {mask.shape} 与图像 shape {overlay.shape[:2]} 不匹配，进行调整")
                mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)

            color = np.array(colors[i], dtype=np.uint8)
            # 创建彩色 mask
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[mask.astype(bool)] = color

            # 叠加时使用一定透明度
            alpha = 0.5
            overlay = np.where(colored_mask > 0, (1 - alpha) * overlay + alpha * colored_mask, overlay).astype(np.uint8)

            # 绘制边界框
            bbox = mask_info["bbox"]  # 格式为 [x, y, w, h]
            x, y, w, h = map(int, bbox)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color.tolist(), 2)

        # 显示叠加结果
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()