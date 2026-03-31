import cv2
import os
import numpy as np

def slice_image(image_path, output_folder, tile_size=640, overlap=50):
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w, _ = img.shape
    stride = tile_size - overlap
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 获取切片坐标，确保不越界
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = img[y_start:y_end, x_start:x_end]
            
            # 过滤掉纯白或内容极少的切片（可选）
            if np.mean(tile) > 250: continue 
            
            cv2.imwrite(f"{output_folder}/tile_{y}_{x}.png", tile)

# 执行切片
slice_image('C:/Users/etenix/Desktop/2F.tif', 'C:/Users/etenix/Desktop/train_tiles')
print("切片完成，请检查 train_tiles 文件夹")