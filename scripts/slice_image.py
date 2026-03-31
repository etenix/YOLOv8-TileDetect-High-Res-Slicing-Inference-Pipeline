import cv2
import os
import numpy as np

def slice_image(image_path, output_folder, tile_size=640, overlap=50):
    """
    高解像度のタイル（瓷砖）画像を学習用パッチに分割するスクリプト
    """
    # 画像データの読み込み
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 指定された画像が見つかりません: {image_path}")
        return
    
    h, w, _ = img.shape
    stride = tile_size - overlap  # タイルのストライド（移動幅）を計算
    
    # 保存先ディレクトリの作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # スライディングウィンドウ方式によるスライス処理
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 画像の境界を超えないように座標を調整
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # 画像の切り出し（スライシング）
            tile = img[y_start:y_end, x_start:x_end]
            
            # フィルタリング：情報量が少ない純白（背景）のタイルを除外
            if np.mean(tile) > 250:
                continue 
            
            # ファイル名の生成と保存処理
            save_name = f"tile_{y}_{x}.png"
            cv2.imwrite(os.path.join(output_folder, save_name), tile)

# --- 実行セクション ---
# GitHubのディレクトリ構造に合わせた相対パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMG = os.path.join(BASE_DIR, "..", "data", "your_image.tif") # dataフォルダ内の画像名に変更
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "train_tiles")

# 処理の実行
slice_image(INPUT_IMG, OUTPUT_DIR)
print(f"処理が完了しました。出力先: {OUTPUT_DIR}")