import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np  # <-- 1. numpyをインポート
from typing import List # <-- 型ヒント用にインポート

# Flaskアプリの初期化
app = Flask(__name__)
CORS(app)

# 学習済みモデルの読み込み
try:
    model = joblib.load('har_random_forest.joblib')
    print("モデルの読み込みに成功しました。")
except Exception as e:
    print(f"モデル読み込み中にエラーが発生しました: {e}")
    model = None

# ラベルIDと行動名のマッピング
# (学習時の {0: 'stay', 1: 'walk', 2: 'jog'} に対応)
LABEL_MAP = {0: 'stay', 1: 'walk', 2: 'jog'}

# --- ▼▼▼ 2. load.py から特徴量抽出関数を丸ごとコピー ▼▼▼ ---
# (tqdmは予測時には不要なので削除)
def extract_features(segments: np.ndarray) -> np.ndarray:
    """
    各セグメント(window)から統計的特徴量を抽出する
    入力: (N, window_size, 3)
    出力: (N, num_features = 21)
    """
    all_features = []
    
    # segments は (N, 100, 3) の形状
    for segment in segments:
        # segment は (100, 3) の形状
        features = []
        
        # 軸ごと (i=0, 1, 2) に特徴量を計算
        for i in range(segment.shape[1]):
            axis_data = segment[:, i] # 1つの軸の100個のデータ
            
            features.append(np.mean(axis_data))
            features.append(np.std(axis_data))
            features.append(np.min(axis_data))
            features.append(np.max(axis_data))
            features.append(np.median(axis_data))
            features.append(np.quantile(axis_data, 0.25)) # 25%点
            features.append(np.quantile(axis_data, 0.75)) # 75%点
            
        # 1セグメントから (3軸 * 7特徴量 = 21個) の特徴が抽出された
        all_features.append(features)
        
    return np.array(all_features)
# --- ▲▲▲ ここまで ---


@app.route('/')
def home():
    return "Activity Prediction API is running! (v2: server-side features)"

@app.route('/predict', methods=['POST'])
def predict():
    """
    POSTリクエストで生のセンサーデータ(100x3)を受け取り、
    サーバー側で特徴量抽出し、行動を予測して返すエンドポイント
    """
    if model is None:
        return jsonify({'error': 'モデルが読み込まれていません'}), 500

    try:
        # リクエストからJSONデータを取得
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'リクエストデータがJSON形式ではありません'}), 400

        # --- ▼▼▼ 3. 予測ロジックを大幅に変更 ▼▼▼ ---
        
        # (1) クライアントから "sensor_data" というキーで 
        #     {x:..., y:..., z:...} のリストが送られてくると想定
        raw_sensor_data = data.get('sensor_data')
        
        if not raw_sensor_data or not isinstance(raw_sensor_data, list):
            return jsonify({'error': 'sensor_data (リスト形式) が見つかりません'}), 400

        # (2) データをNumpy配列 (100, 3) に変換
        #    学習時の WINDOW_SIZE=100 と一致させる
        window_size = 100 
        if len(raw_sensor_data) != window_size:
            return jsonify({'error': f'データ数が {len(raw_sensor_data)} 件です。{window_size} 件必要です。'}), 400

        # {x:.., y:.., z:..} のリストを (100, 3) のNumpy配列に変換
        segment_data = np.array(
            [[item['x'], item['y'], item['z']] for item in raw_sensor_data]
        )
        
        # (3) 特徴量抽出関数 (extract_features) に渡す
        #     入力形状を (1, 100, 3) にする
        input_for_extraction = np.array([segment_data])
        
        features_21 = extract_features(input_for_extraction)
        # これで features_21 の形状は (1, 21) になる
        
        # (4) モデルを使って予測を実行 (21個の特徴量を渡す)
        prediction_id = model.predict(features_21) 
        
        # (5) 結果を返す (ロジックは以前と同じ)
        predicted_label_id = int(prediction_id[0])
        predicted_label_name = LABEL_MAP.get(predicted_label_id, 'unknown')

        return jsonify({
            'predicted_id': predicted_label_id,
            'predicted_label': predicted_label_name
        })
        # --- ▲▲▲ 予測ロジックの変更ここまで ---

    except Exception as e:
        # その他のエラー
        return jsonify({'error': f'予測中にエラーが発生しました: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)