# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from typing import Optional, Tuple
from .performance_monitor import get_perf_monitor


class YOLODetector:
    """使用YOLOv8進行物體檢測並投影到3D座標系"""
    
    def __init__(self, model_name: str = "yolov8m", device: str = "cuda:0", conf_threshold: float = 0.5):
        """
        初始化YOLO檢測器
        
        Args:
            model_name: YOLOv8模型大小 ("nano", "small", "medium", "large")
            device: 推理裝置 ("cuda:0" 或 "cpu")
            conf_threshold: 置信度閾值
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics包未安裝，請執行: pip install ultralytics")
        
        # 載入官方預訓練權重
        import os
        local_model_path = os.path.join(os.path.dirname(__file__), f"{model_name}.pt")
        if os.path.exists(local_model_path):
            model_path = local_model_path
        else:
            model_path = f"{model_name}.pt"
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Performance monitoring
        self.perf_monitor = get_perf_monitor()
        self.perf_monitor.set_device(device)
        
        print(f"[YOLODetector] 載入模型: {model_path} on {device}")
        print(f"[YOLODetector] Performance monitoring: ENABLED")
    
    def detect(self, rgb_image: torch.Tensor) -> Optional[dict]:
        """
        在RGB影象上執行YOLO檢測
        
        Args:
            rgb_image: (H, W, 3) 或 (B, H, W, 3) 張量，值範圍[0, 255]
        
        Returns:
            檢測結果字典或None (包含boxes, confidences, num_detections)
        """
        # 轉換為numpy（YOLO期望numpy）
        if isinstance(rgb_image, torch.Tensor):
            is_batch = len(rgb_image.shape) == 4
            if not is_batch:
                rgb_image = rgb_image.unsqueeze(0)  # 添加批次維度
            rgb_np = rgb_image.cpu().numpy()
        else:
            is_batch = len(rgb_image.shape) == 4
            if not is_batch:
                rgb_image = np.expand_dims(rgb_image, 0)
            rgb_np = rgb_image
        
        # 確保是uint8且值範圍正確
        if rgb_np.dtype != np.uint8:
            rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8) if rgb_np.max() <= 1 else rgb_np.astype(np.uint8)
        
        # 儲存原始大小
        orig_h, orig_w = rgb_np.shape[1], rgb_np.shape[2]
        
        # ========== 關鍵修復：Resize到能被32整除的尺寸 ==========
        target_h = ((orig_h + 31) // 32) * 32
        target_w = ((orig_w + 31) // 32) * 32
        
        # 批量Resize (必須先轉換為float32，PyTorch不支持uint8的雙線性插值)
        rgb_resized_list = []
        for img in rgb_np:  # img is HWC
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)) / 255.0  # CHW, [0, 1]
            img_resized = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),  # Add batch dimension
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0).numpy()  # Back to HWC
            rgb_resized_list.append(img_resized)
        
        rgb_resized_np = np.stack(rgb_resized_list, axis=0)
        
        # 執行推理
        with torch.no_grad():
            results = self.model.predict(rgb_resized_np, conf=self.conf_threshold, verbose=False)
        
        if results is None or len(results) == 0:
            return None
        
        result = results[0]  # 第一個結果
        
        # 提取檢測框
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()  # (N,)
        
        # ========== 尺度轉換回原始尺寸 ==========
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        return {
            "boxes": boxes,        # (N, 4) 格式: [x1, y1, x2, y2]
            "confidences": confs,
            "num_detections": len(boxes)
        }
    
    def detect_batch(self, rgb_images: torch.Tensor) -> list:
        """
        在批量RGB圖像上運行YOLO檢測 (批次處理以提高效能)
        
        Args:
            rgb_images: (B, H, W, 3) 張量，值範圍[0, 255]，設備為cuda
        
        Returns:
            檢測結果列表，包含B個字典或None
        """
        self.perf_monitor.start_timer("yolo_detect_batch_total")
        
        if not isinstance(rgb_images, torch.Tensor):
            rgb_images = torch.tensor(rgb_images, device=self.device)
            
        # 儲存原始大小以供後續縮放
        orig_h, orig_w = rgb_images.shape[1], rgb_images.shape[2]
        
        # ========== 關鍵修復：Resize到能被32整除的尺寸 ==========
        # YOLOv8要求輸入高度和寬度能被32整除
        # 原始(80, 128) -> Resize到(96, 128) 都能被32整除
        target_h = ((orig_h + 31) // 32) * 32  # 80 -> 96
        target_w = ((orig_w + 31) // 32) * 32  # 128 -> 128
        
        self.perf_monitor.start_timer("yolo_preprocess")
        
        # 執行Resize (使用雙線性插值)
        # 重要: 必須先轉換為float32，因為PyTorch不支持uint8的雙線性插值
        # OPTIMIZATION: Keep in GPU, avoid unnecessary CPU transfers
        rgb_float = rgb_images.permute(0, 3, 1, 2).float() / 255.0  # BHWC -> BCHW, [0, 1]
        rgb_resized = torch.nn.functional.interpolate(
            rgb_float,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        self.perf_monitor.end_timer("yolo_preprocess")
        
        # ========== 執行YOLO推理 ==========
        self.perf_monitor.start_timer("yolo_inference")
        with torch.no_grad():
            results = self.model(rgb_resized, conf=self.conf_threshold, verbose=False)
        self.perf_monitor.end_timer("yolo_inference")
        
        # ========== 尺度轉換：將偵測框from resized image轉換回原始尺寸 ==========
        self.perf_monitor.start_timer("yolo_postprocess")
        
        batch_results = []
        scale_x = orig_w / target_w
        scale_y = orig_h / target_h
        
        for result in results:
            if result is None or len(result) == 0:
                batch_results.append(None)
                continue
                
            # OPTIMIZATION: Minimize GPU-CPU transfers
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) in resized coords
            confs = result.boxes.conf.cpu().numpy()  # (N,)
            
            # 縮放回原始圖像座標 (vectorized operation)
            boxes[:, [0, 2]] *= scale_x  # x1, x2 縮放
            boxes[:, [1, 3]] *= scale_y  # y1, y2 縮放
            
            batch_results.append({
                "boxes": boxes,
                "confidences": confs,
                "num_detections": len(boxes)
            })
        
        self.perf_monitor.end_timer("yolo_postprocess")
        self.perf_monitor.end_timer("yolo_detect_batch_total")
        
        return batch_results
    
    def get_center_object(self, detections: dict, image_height: int, image_width: int) -> Optional[Tuple[np.ndarray, int]]:
        """
        選擇離影象中心最近的檢測物體
        
        Args:
            detections: detect()的返回結果
            image_height: 影象高度
            image_width: 影象寬度
        
        Returns:
            (bbox, idx) 其中bbox格式[x1, y1, x2, y2]，idx是在檢測框陣列中的索引
            如果沒有檢測到物體則返回None
        """
        if detections is None or detections["num_detections"] == 0:
            return None
        
        boxes = detections["boxes"]
        
        # 計算每個框的中心
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
        
        # 影象中心
        img_center_x = image_width / 2.0
        img_center_y = image_height / 2.0
        
        # 計算到影象中心的歐幾里得距離
        distances = np.sqrt((centers_x - img_center_x)**2 + (centers_y - img_center_y)**2)
        
        # 找到最近的檢測框索引
        min_idx = np.argmin(distances)
        
        return boxes[min_idx], min_idx
    
    def project_2d_to_3d(self, bbox_2d: np.ndarray, depth_image: torch.Tensor, 
                        fx: float, fy: float, cx: float, cy: float) -> Optional[np.ndarray]:
        """
        將2D邊界框投影到3D，獲得8個角點座標（相機座標系）
        
        Args:
            bbox_2d: (4,) 格式[x1, y1, x2, y2]，畫素座標
            depth_image: (H, W, 1) 深度圖，單位米
            fx, fy: 相機內參（焦距）
            cx, cy: 相機內參（主點）
        
        Returns:
            (8, 3) 陣列，8個3D點座標（相機座標系）
            或 None 如果投影失敗
        """
        try:
            x1, y1, x2, y2 = bbox_2d
            x1, y1, x2, y2 = int(np.clip(x1, 0, depth_image.shape[1]-1)), int(np.clip(y1, 0, depth_image.shape[0]-1)), \
                             int(np.clip(x2, 0, depth_image.shape[1]-1)), int(np.clip(y2, 0, depth_image.shape[0]-1))
            
            # 獲取邊界框區域的深度
            if x2 <= x1 or y2 <= y1:
                return None
                
            depth_region = depth_image[y1:y2+1, x1:x2+1, 0]
            
            if depth_region.numel() == 0:
                return None
            
            # 使用中值深度（更魯棒）
            valid_depths = depth_region[~torch.isnan(depth_region) & ~torch.isinf(depth_region)]
            if len(valid_depths) == 0:
                return None
            
            z_median = torch.median(valid_depths).item()
            
            # 8個角點的2D座標（矩形的8個頂點）
            corners_2d = np.array([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2],  # 前4個：近框四個角
                [x1, y1], [x2, y1], [x1, y2], [x2, y2],  # 後4個：遠框四個角（相同深度）
            ])
            
            # 轉換為3D（相機座標系）
            # x = (u - cx) * z / fx
            # y = (v - cy) * z / fy
            # z = z
            corners_3d = []
            for u, v in corners_2d:
                x = (u - cx) * z_median / fx
                y = (v - cy) * z_median / fy
                z = z_median
                corners_3d.append([x, y, z])
            
            return np.array(corners_3d)
        
        except Exception as e:
            print(f"[YOLODetector] 3D投影錯誤: {e}")
            return None
