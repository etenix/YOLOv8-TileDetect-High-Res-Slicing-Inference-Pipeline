from ultralytics import YOLO

# 必须加上这一行判断
if __name__ == '__main__':
    # 1. 加载模型
    model = YOLO('yolov8n.pt')

    # 2. 开始训练
    # 建议加上 workers=0 如果之后还报错，但通常加上 if 就能解决
    results = model.train(
        data='C:/Users/etenix/Desktop/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4  # Windows 建议设置成 CPU 核心数的一半或更小
    )