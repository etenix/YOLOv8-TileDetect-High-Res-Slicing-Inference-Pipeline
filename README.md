YOLOv8-TileDetect: High-Res Slicing Inference Pipeline
超高解像度画像におけるタイルの自動検知・統計システム

1. プロジェクト概要
本プロジェクトは、高解像度の画像に含まれるタイルを、YOLOv8 を用いて高精度に検知・カウントするパイプラインです。通常の物体検知では消失しやすい細線情報を保持するため、「スライシング（切片化）推論」を採用しています。

2. 解決した技術的課題
解像度の問題: 入力画像をリサイズするとタイルが消失する問題を、タイル状の切片処理（640px）で解決。
アノテーションの効率化: 数千個のタイルを手動で囲む代わりに、OpenCV を用いた**自動プレアノテーション（Auto-labeling）**により工数を大幅に削減。
重複検知の回避: 重なり（Overlap）部分におけるタイルの二重カウントを、中心点座標判定ロジックにより排除。

3. ワークフローと使用方法
Step 1: 画像の切片化 (Slicing)
大画像を学習に適した 640x640 のタイルに分割します。
python scripts/slice_image.py
Step 2: 自動プレアノテーション (Auto-labeling)
OpenCV の二値化とモルフォロジー演算（腐食・膨張）により、タイルの外形を自動抽出し YOLO 形式の .txt を生成します。
python scripts/auto_label.py
Step 3: モデルの学習 (Training)
NVIDIA GPU を活用し、YOLOv8n モデルで転移学習を実施します。
python scripts/train.py
Step 4: 大画像の自動検知 (Inference)
学習済みモデル（best.pt）を使用し、大画像全体をスキャンしてタイルの総数を算出します。
python scripts/detect_full.py

4. 環境構築
Python 3.12+
PyTorch 2.4+ (CUDA 12.4 対応推奨)
Ultralytics (YOLOv8)
OpenCV-Python

5. 実績 (Results)
推論精度: mAP50 0.944 / mAP50-95 0.939
処理速度: 1sliceあたり約 1.9ms
