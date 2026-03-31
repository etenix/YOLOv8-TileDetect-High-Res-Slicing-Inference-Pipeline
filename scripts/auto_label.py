import cv2
import os
import numpy as np

def debug_auto_label(tile_folder):
    files = [f for f in os.listdir(tile_folder) if f.endswith(('.png', '.jpg'))]
    if not files:
        print("文件夹内没找到图片！")
        return

    for filename in files:
        img_path = os.path.join(tile_folder, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 全局阈值：黑白反转 (INV)，让线条变白，背景变黑
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

        # 2. 再次反转图像：让格子内部变白，线条变黑
        # 此时白色区域是我们要抓取的轮廓
        inverted = cv2.bitwise_not(thresh)

        # 3. 腐蚀：由于线条非常细，1次迭代即可，确保格子互不相连
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(inverted, kernel, iterations=1)

        # 4. 查找轮廓：确保使用上面处理好的变量 'eroded'
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_data = []
        h, w = img.shape[:2]

        for cnt in contours:
            # 获取外接矩形
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # 5. 尺寸过滤：剔除整张图的大框和极小的噪点
            # 假设格子宽度正常在 5-150 像素之间
            if 5 < bw < (w * 0.9) and 5 < bh < (h * 0.9):
                # 计算 YOLO 归一化坐标
                cx = (bx + bw/2.0) / w
                cy = (by + bh/2.0) / h
                nw = bw / float(w)
                nh = bh / float(h)
                
                yolo_data.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                # 画框标注
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)

        print(f"正在检查: {filename}, 抓取到数量: {len(yolo_data)}")
        
        # 展示结果
        cv2.imshow('Binary_Look', eroded)
        cv2.imshow('Result_Look', img)
        
        key = cv2.waitKey(0)
        if key == 27: # 按 ESC 键退出调试
            break
        
        # 保存结果
        if yolo_data:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_data))

tile_path = 'C:/Users/etenix/Desktop/train_tiles'
debug_auto_label(tile_path)
cv2.destroyAllWindows()