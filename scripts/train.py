import os
from ultralytics import YOLO

def train_tile_model():
    """
    YOLOv8を使用してタイルの検知モデルを学習させるスクリプト
    """
    # 1. パスの設定（スクリプトの場所を基準にプロジェクトルートを特定）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # プロジェクトルートにある data.yaml のパスを指定
    # (data.yaml 内の 'path' もプロジェクトルートを指すように設定してください)
    data_yaml_path = os.path.join(BASE_DIR, '..', 'data.yaml')
    
    # 2. モデルのロード
    # 学習済みの yolov8n (Nano) モデルをベースに使用（軽量・高速）
    model = YOLO('yolov8n.pt')

    # 3. 学習の実行 (Training)
    # Windows環境でのマルチプロセス問題を避けるため、workersの数に注意
    results = model.train(
        data=data_yaml_path,    # データセット設定ファイルのパス
        epochs=100,             # 学習のエポック数
        imgsz=640,              # 入力画像のサイズ
        batch=16,               # バッチサイズ
        device=0,               # 使用するGPUデバイスID
        workers=4,              # データ読み込みの並列スレッド数
        project='../runs',      # 結果の保存先
        name='tile_detection'   # 学習セッション名
    )
    
    print("学習が完了しました。")
    print(f"モデルの保存先: {results.save_dir}")

if __name__ == '__main__':
    # Windowsにおけるマルチプロセス実行を保護するためのエントリポイント
    train_tile_model()