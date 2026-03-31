import cv2
import os
import numpy as np

def auto_label_tiles(tile_folder):
    """
    OpenCVを使用して分割済みタイル画像からアノテーション(YOLO形式)を自動生成する
    """
    # フォルダ内の画像ファイル一覧を取得
    files = [f for f in os.listdir(tile_folder) if f.endswith(('.png', '.jpg'))]
    if not files:
        print(f"画像が見つかりません: {tile_folder}")
        return

    for filename in files:
        img_path = os.path.join(tile_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 二値化処理：白黒反転 (INV) により目地を白、背景を黒にする
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

        # 2. 再反転：タイル内部を白、目地を黒にする
        # これにより、findContoursがタイルの形状を「物体」として認識できる
        inverted = cv2.bitwise_not(thresh)

        # 3. 腐食処理 (Erosion)：タイル同士の固着を防ぎ、個別に分離させる
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(inverted, kernel, iterations=1)

        # 4. 輪郭抽出：最外周の輪郭のみを取得
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_data = []
        h, w = img.shape[:2]

        for cnt in contours:
            # 各輪郭の外接矩形を取得
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # 5. フィルタリング：画像全体を囲む枠や微細なノイズを除去
            # タイルのサイズが画像の90%未満、かつ一定以上の大きさを持つものを採用
            if 5 < bw < (w * 0.9) and 5 < bh < (h * 0.9):
                # YOLO形式への正規化計算 (0.0 ~ 1.0)
                # $$cx = \frac{bx + \frac{bw}{2}}{w}, cy = \frac{by + \frac{bh}{2}}{h}$$
                cx = (bx + bw / 2.0) / w
                cy = (by + bh / 2.0) / h
                nw = bw / float(w)
                nh = bh / float(h)
                
                # クラスID 0 (タイル) としてデータを追加
                yolo_data.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                # デバッグ用に赤枠を描画
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 1)

        print(f"Processing: {filename} | Detected Tiles: {len(yolo_data)}")
        
        # プレビュー表示 (ESCキーで終了)
        cv2.imshow('Binary_Look', eroded)
        cv2.imshow('Result_Look', img)
        
        if cv2.waitKey(1) == 27:
            break
        
        # YOLO形式のテキストファイル保存
        if yolo_data:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_data))

if __name__ == '__main__':
    # スクリプトの場所を基準とした相対パスの設定
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # train_tiles フォルダはプロジェクトルートに配置されている想定
    TILE_PATH = os.path.join(BASE_DIR, "..", "train_tiles")
    
    auto_label_tiles(TILE_PATH)
    cv2.destroyAllWindows()
    print("プレアノテーションが完了しました。")