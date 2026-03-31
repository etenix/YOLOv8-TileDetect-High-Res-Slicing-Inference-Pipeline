import cv2
import os
import numpy as np
from ultralytics import YOLO

# 1. 配置路径
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
model_path = os.path.join(desktop_path, "best.pt") # 确保 best.pt 在桌面
image_path = os.path.join(desktop_path, "1F.tif")    # 你的大图文件名
output_path = os.path.join(desktop_path, "Result_Full.jpg")

# 2. 加载模型
model = YOLO(model_path)

# 3. 读取大图
full_img = cv2.imread(image_path)
h, w = full_img.shape[:2]

# 定义切片大小和重叠（重叠 50 像素防止格子被切断）
tile_size = 640
overlap = 50
stride = tile_size - overlap

total_count = 0
print("正在处理大图切片推理...")

# 4. 滑动窗口循环
for y in range(0, h, stride):
    for x in range(0, w, stride):
        # 确保切片不越界
        y_end = min(y + tile_size, h)
        x_end = min(x + tile_size, w)
        y_start = max(0, y_end - tile_size)
        x_start = max(0, x_end - tile_size)
        
        # 截取切片
        tile = full_img[y_start:y_end, x_start:x_end]
        
        # 预测（stream=True 节省显存）
        results = model.predict(tile, conf=0.4, verbose=False)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() # 获取左上角和右下角坐标
            for box in boxes:
                # 5. 坐标还原：加上当前切片的起始坐标 (x_start, y_start)
                real_x1 = int(box[0] + x_start)
                real_y1 = int(box[1] + y_start)
                real_x2 = int(box[2] + x_start)
                real_y2 = int(box[3] + y_start)
                
                # 简单的去重逻辑：如果中心点在当前步长的有效范围内才计数
                # 防止重叠部分的格子被重复计算
                cx, cy = (real_x1 + real_x2)/2, (real_y1 + real_y2)/2
                if (x <= cx < x + stride) and (y <= cy < y + stride):
                    total_count += 1
                    cv2.rectangle(full_img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)

# 6. 保存和输出
print("-" * 30)
print(f"检测完成！整张大图中的格子总数为: {total_count}")
cv2.imwrite(output_path, full_img)
print(f"标注后的完整大图已保存至桌面: Result_Full.jpg")
print("-" * 30)