import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect_and_count_tiles(image_name, output_name):
    """
    大解像度の画像に対してスライディングウィンドウ推論を行い、
    タイル（瓷砖）の総数をカウントして結果を保存する
    """
    # 1. パスの設定（スクリプトの場所を基準に設定）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # モデルパス: models/best.pt
    model_path = os.path.join(BASE_DIR, '..', 'models', 'best.pt')
    # 入力画像パス: data/1F.tif
    image_path = os.path.join(BASE_DIR, '..', 'data', image_name)
    # 出力先パス（プロジェクトルートに保存）
    output_path = os.path.join(BASE_DIR, '..', output_name)

    # 2. モデルのロード
    if not os.path.exists(model_path):
        print(f"Error: モデルファイルが見つかりません: {model_path}")
        return
    model = YOLO(model_path)

    # 3. オリジナル画像の読み込み
    full_img = cv2.imread(image_path)
    if full_img is None:
        print(f"Error: 画像を読み込めませんでした: {image_path}")
        return
    
    h, w = full_img.shape[:2]

    # 切片サイズとオーバーラップの設定
    tile_size = 640
    overlap = 50
    stride = tile_size - overlap

    total_count = 0
    print(f"推論開始: {image_name} ({w}x{h})")

    # 4. スライディングウィンドウによる走査
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 画像の境界を超えないように座標を調整（パッチサイズの維持）
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # タイルの切り出し
            tile = full_img[y_start:y_end, x_start:x_end]
            
            # YOLOv8による推論（verbose=Falseでログ出力を抑制）
            results = model.predict(tile, conf=0.4, verbose=False)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
                for box in boxes:
                    # 5. 座標の復元（タイル内座標から全体座標へ変換）
                    real_x1 = int(box[0] + x_start)
                    real_y1 = int(box[1] + y_start)
                    real_x2 = int(box[2] + x_start)
                    real_y2 = int(box[3] + y_start)
                    
                    # 重複カウント防止ロジック：
                    # 検出されたタイルの中心点が現在のストライド範囲内にある場合のみカウントする
                    cx, cy = (real_x1 + real_x2) / 2, (real_y1 + real_y2) / 2
                    if (x <= cx < x + stride) and (y <= cy < y + stride):
                        total_count += 1
                        # 全体画像にバウンディングボックスを描画（緑色）
                        cv2.rectangle(full_img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 2)

    # 6. 結果の保存と出力
    print("-" * 30)
    print(f"検知完了！タイルの総数: {total_count}")
    cv2.imwrite(output_path, full_img)
    print(f"結果画像を保存しました: {output_path}")
    print("-" * 30)

if __name__ == '__main__':
    # dataフォルダ内のファイル名と出力ファイル名を指定して実行
    detect_and_count_tiles('1F.tif', 'Result_Full.jpg')