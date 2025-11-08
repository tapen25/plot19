import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- 1. この行をインポートに追加

# Flaskアプリの初期化
app = Flask(__name__)
CORS(app)  # <-- 2. この行を追加 (すべてのオリジンからの通信を許可)

# 学習済みモデルの読み込み
try:
    model = joblib.load('activity_model.joblib')
    print("モデルの読み込みに成功しました。")
except FileNotFoundError:
    print("エラー: モデルファイル 'activity_model.joblib' が見つかりません。")
    model = None
except Exception as e:
    print(f"モデル読み込み中にエラーが発生しました: {e}")
    model = None

# ラベルIDと行動名のマッピング
# (学習時の {0: 'stay', 1: 'walk', 2: 'jog'} に対応)
LABEL_MAP = {0: 'stay', 1: 'walk', 2: 'jog'}

@app.route('/')
def home():
    # ルートURLへの簡単な疎通確認
    return "Activity Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    POSTリクエストでJSONデータを受け取り、行動を予測して返すエンドポイント
    """
    if model is None:
        return jsonify({'error': 'モデルが読み込まれていません'}), 500

    try:
        # リクエストからJSONデータを取得
        data = request.get_json()

        # データがJSON形式でない、または空の場合のエラーハンドリング
        if data is None:
            return jsonify({'error': 'リクエストデータがJSON形式ではありません'}), 400

        # 学習時に使用した特徴量のリスト
        features_columns = ['std_acc', 'min_acc', 'energy']

        # 受け取ったJSONデータをPandas DataFrameに変換
        # (入力は単一の予測リクエスト { "mean_acc": 0.9, ... } と想定)
        # model.predict() は2D配列を期待するため、[data] のようにリストでラップする
        input_df = pd.DataFrame([data], columns=features_columns)

        # 特徴量が不足している場合のチェック
        if input_df.isnull().values.any():
             return jsonify({'error': '特徴量が不足しています。5つの特徴量(mean_acc, std_acc, max_acc, min_acc, energy)が必要です。'}), 400

        # モデルを使って予測を実行
        prediction_id = model.predict(input_df) # [0], [1], [2] のいずれかを返す
        
        # 予測されたID（例: 2）を取得
        predicted_label_id = int(prediction_id[0])
        
        # IDを人間が読める行動名（例: 'jog'）に変換
        predicted_label_name = LABEL_MAP.get(predicted_label_id, 'unknown')

        # 予測結果をJSONで返す
        return jsonify({
            'predicted_id': predicted_label_id,
            'predicted_label': predicted_label_name
        })

    except Exception as e:
        # その他のエラー
        return jsonify({'error': f'予測中にエラーが発生しました: {str(e)}'}), 500

if __name__ == '__main__':
    # ローカルテスト用（RenderではGunicornが使われる）
    app.run(debug=True, host='0.0.0.0', port=5000)