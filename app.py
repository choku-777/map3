from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import pandas as pd
import jageocoder
import tempfile
from io import StringIO
import csv
from dotenv import load_dotenv
import chardet
import time
from pathlib import Path
import uuid
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from scipy.spatial.distance import pdist, squareform

# 環境変数の読み込み
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_FOLDER'] = 'temp'  # 一時ファイル用のフォルダ
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB制限
app.secret_key = 'your-secret-key'  # セッション管理用のシークレットキー

# Google Maps APIキーを設定
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# 必要なフォルダの作成
for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# jageocoderの初期化（環境変数を使用）
try:
    # 環境変数からの初期化を試みる
    jageocoder.init()
    print("jageocoderの初期化が成功しました")
except Exception as e:
    print(f"jageocoderの初期化中にエラーが発生しました: {str(e)}")

# リトライ設定
MAX_RETRIES = 3  # 最大リトライ回数
RETRY_DELAY = 1  # リトライ間隔（秒）

def save_to_temp_file(df):
    """データフレームを一時ファイルに保存し、ファイル名を返す"""
    # ユニークなファイル名を生成
    filename = f"geocoded_{uuid.uuid4().hex}.csv"
    filepath = os.path.join(app.config['TEMP_FOLDER'], filename)
    
    # CSVとして保存
    df.to_csv(filepath, index=False, encoding='utf-8')
    return filename

def get_temp_file_path(filename):
    """一時ファイルの完全パスを取得"""
    return os.path.join(app.config['TEMP_FOLDER'], filename)

def calculate_optimal_route(df, start_lat, start_lng):
    """最適な訪問順序を計算する"""
    # 緯度経度が有効な地点のみを抽出
    valid_points = df[df['latitude'].notna() & df['longitude'].notna()].copy()
    
    if len(valid_points) == 0:
        return None, None

    # スタート/ゴール地点を含めた全ての地点の配列を作成
    points = np.array([[lat, lng] for lat, lng in zip(valid_points['latitude'], valid_points['longitude'])])
    
    # 距離行列を計算（ヒュベニの公式を使用）
    def haversine_distance(p1, p2):
        R = 6371  # 地球の半径（km）
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    # 距離行列を作成
    distances = squareform(pdist(points, lambda x, y: haversine_distance(x, y)))
    
    # TSPを解く
    permutation, distance = solve_tsp_dynamic_programming(distances)
    
    # 結果を整形：訪問順序をそのまま返す
    return permutation, distance

@app.route('/')
def index():
    return render_template('index.html')

def find_address_column(df):
    """住所が含まれる可能性の高い列名を検索"""
    # 住所カラムの候補となるキーワード
    address_keywords = ['住所', 'address', '所在地', '場所', '地点', '地域']
    
    # 列名を小文字に変換して検索
    columns_lower = [col.lower() for col in df.columns]
    
    # キーワードに完全一致する列名を探す
    for keyword in address_keywords:
        for i, col in enumerate(columns_lower):
            if keyword == col:
                return df.columns[i]
    
    # 部分一致する列名を探す
    for keyword in address_keywords:
        for i, col in enumerate(columns_lower):
            if keyword in col:
                return df.columns[i]
    
    # 見つからない場合は最初の列を返す
    return df.columns[0]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(url_for('index'))
    
    if file and file.filename.endswith('.csv'):
        try:
            # ファイルの内容を読み込む
            file_content = file.read()
            
            # ファイルが空かどうかチェック
            if not file_content:
                flash('CSVファイルが空です')
                return redirect(url_for('index'))
            
            # エンコーディングを自動検出
            encoding_detect = chardet.detect(file_content)
            detected_encoding = encoding_detect['encoding']
            
            # エンコーディング情報をログに出力
            print(f"検出されたエンコーディング: {detected_encoding}, 信頼度: {encoding_detect['confidence']}")
            
            # エンコーディングの候補リスト
            encodings = [
                detected_encoding,  # 検出されたエンコーディング
                'utf-8',           # UTF-8
                'shift-jis',       # Shift-JIS
                'cp932',           # Windows日本語
                'euc-jp'          # 日本語EUC
            ]
            
            df = None
            successful_encoding = None
            
            # 各エンコーディングを試す
            for encoding in encodings:
                if encoding is None:
                    continue
                try:
                    # バイト列からデータフレームを作成
                    df = pd.read_csv(StringIO(file_content.decode(encoding)))
                    successful_encoding = encoding
                    print(f"成功したエンコーディング: {encoding}")
                    break
                except Exception as e:
                    print(f"{encoding}でのデコードに失敗: {str(e)}")
                    continue
            
            if df is None:
                flash('ファイルのエンコーディングを特定できませんでした。')
                return redirect(url_for('index'))
            
            if df.empty:
                flash('CSVファイルにデータが含まれていません')
                return redirect(url_for('index'))
            
            # カラム名の確認
            if len(df.columns) == 0:
                flash('CSVファイルにカラムが含まれていません')
                return redirect(url_for('index'))

            # 住所カラムを選択
            address_column = request.form.get('address_column')
            if not address_column:
                # カラム名が指定されていない場合、住所カラムを自動検出
                address_column = find_address_column(df)
                flash(f'住所カラムが指定されていないため、"{address_column}" を使用します')
            elif address_column not in df.columns:
                flash(f'指定された住所カラム "{address_column}" が見つかりません。自動検出した "{find_address_column(df)}" を使用します')
                address_column = find_address_column(df)
            
            # スタート地点の情報
            start_point = {
                '店名': 'スタート/ゴール地点',
                '住所': '群馬県藤岡市中187番地1',
                'latitude': 36.274841,
                'longitude': 139.06868
            }
            
            # スタート地点をデータフレームに追加
            df = pd.concat([pd.DataFrame([start_point]), df], ignore_index=True)
            
            # ジオコーディング処理
            lat_list = []
            lng_list = []
            success_count = 0
            fail_count = 0
            
            # スタート地点の座標は既に設定済みなのでスキップ
            lat_list.append(start_point['latitude'])
            lng_list.append(start_point['longitude'])
            success_count += 1
            
            # 2行目以降の住所をジオコーディング
            for address in df.iloc[1:][address_column]:
                try:
                    if pd.isna(address):
                        lat_list.append(None)
                        lng_list.append(None)
                        fail_count += 1
                        continue

                    address_str = str(address).strip()
                    print(f"処理中の住所: {address_str}")

                    # ジオコーディングのリトライ処理
                    retry_count = 0
                    result = None

                    while retry_count < MAX_RETRIES:
                        try:
                            results = jageocoder.searchNode(address_str)
                            if results and len(results) > 0:
                                result = results[0]
                                break
                            retry_count += 1
                            time.sleep(RETRY_DELAY)
                        except Exception as e:
                            print(f"リトライ {retry_count + 1}/{MAX_RETRIES}: {str(e)}")
                            retry_count += 1
                            time.sleep(RETRY_DELAY)

                    if result and hasattr(result, 'node'):
                        node = result.node
                        print(f"ジオコーディング結果: {node.get_fullname()}")
                        
                        try:
                            lat = float(node.y) if node.y is not None else None
                            lng = float(node.x) if node.x is not None else None
                            if lat is not None and lng is not None:
                                lat_list.append(lat)
                                lng_list.append(lng)
                                success_count += 1
                                continue
                        except (ValueError, TypeError) as e:
                            print(f"座標値の変換エラー: {str(e)}")

                    # searchメソッドで再試行
                    retry_count = 0
                    while retry_count < MAX_RETRIES:
                        try:
                            result = jageocoder.search(address_str)
                            if result and 'candidates' in result and len(result['candidates']) > 0:
                                candidate = result['candidates'][0]
                                print(f"ジオコーディング結果 (search): {candidate}")
                                try:
                                    lat = float(candidate.get('y'))
                                    lng = float(candidate.get('x'))
                                    if lat is not None and lng is not None:
                                        lat_list.append(lat)
                                        lng_list.append(lng)
                                        success_count += 1
                                        break
                                except (ValueError, TypeError) as e:
                                    print(f"座標値の変換エラー (search): {str(e)}")
                            retry_count += 1
                            time.sleep(RETRY_DELAY)
                        except Exception as e:
                            print(f"リトライ (search) {retry_count + 1}/{MAX_RETRIES}: {str(e)}")
                            retry_count += 1
                            time.sleep(RETRY_DELAY)
                    
                    if retry_count >= MAX_RETRIES:
                        print(f"住所 '{address_str}' のジオコーディングに失敗しました")
                        lat_list.append(None)
                        lng_list.append(None)
                        fail_count += 1

                except Exception as e:
                    print(f"ジオコーディングエラー: {str(e)} - 住所: {address}")
                    lat_list.append(None)
                    lng_list.append(None)
                    fail_count += 1

            # 結果をデータフレームに追加
            df['latitude'] = lat_list
            df['longitude'] = lng_list
            
            # 処理結果の統計を表示
            total = len(df) - 1  # スタート地点を除外
            success_rate = (success_count - 1) / total * 100 if total > 0 else 0  # スタート地点を除外
            flash(f'処理完了: 全{total}件中、成功{success_count-1}件 ({success_rate:.1f}%)、失敗{fail_count}件')
            
            # 緯度経度が有効な地点のみを抽出
            valid_points = df[df['latitude'].notna() & df['longitude'].notna()].copy()
            
            if not valid_points.empty:
                # 最適ルートを計算
                route_indices, total_distance = calculate_optimal_route(valid_points, None, None)
                
                if route_indices is not None:
                    # データを最適な順序で並び替え
                    valid_points_reordered = valid_points.iloc[route_indices].copy()
                    valid_points_reordered['訪問順序'] = range(1, len(valid_points_reordered) + 1)
                    
                    # 無効な地点（緯度経度がない）を末尾に追加
                    invalid_points = df[df['latitude'].isna() | df['longitude'].isna()]
                    if not invalid_points.empty:
                        invalid_points['訪問順序'] = None
                        df_reordered = pd.concat([valid_points_reordered, invalid_points])
                    else:
                        df_reordered = valid_points_reordered
                    
                    # 総移動距離をキロメートル単位で記録
                    total_distance_km = round(total_distance, 2)
                    
                    # 結果を一時ファイルに保存
                    temp_filename = save_to_temp_file(df_reordered)
                    
                    return render_template('result.html',
                                        table=df_reordered.to_html(classes='table table-striped', index=False),
                                        filename=temp_filename,
                                        total_distance=total_distance_km,
                                        start_point={'lat': start_point['latitude'], 
                                                   'lng': start_point['longitude']})
            
            # ルート計算できない場合は元のデータを表示
            temp_filename = save_to_temp_file(df)
            return render_template('result.html',
                                table=df.to_html(classes='table table-striped', index=False),
                                filename=temp_filename,
                                start_point={'lat': df['latitude'].iloc[0] if not df.empty else None, 
                                           'lng': df['longitude'].iloc[0] if not df.empty else None})

        except Exception as e:
            flash(f'CSVファイルの処理中にエラーが発生しました: {str(e)}')
            return redirect(url_for('index'))
    
    flash('許可されていないファイル形式です。CSVファイルをアップロードしてください。')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """ジオコーディング結果をダウンロード"""
    if not filename or '..' in filename:  # セキュリティチェック
        flash('無効なファイル名です')
        return redirect(url_for('index'))
    
    try:
        filepath = get_temp_file_path(filename)
        if not os.path.exists(filepath):
            flash('ファイルが見つかりません')
            return redirect(url_for('index'))
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name='geocoded_addresses.csv'
        )
    except Exception as e:
        flash(f'ファイルのダウンロード中にエラーが発生しました: {str(e)}')
        return redirect(url_for('index'))

# 定期的な一時ファイルのクリーンアップ
def cleanup_temp_files(max_age_hours=24):
    """古い一時ファイルを削除"""
    temp_dir = Path(app.config['TEMP_FOLDER'])
    current_time = time.time()
    
    for temp_file in temp_dir.glob('geocoded_*.csv'):
        # ファイルの経過時間をチェック
        file_age_hours = (current_time - temp_file.stat().st_mtime) / 3600
        if file_age_hours > max_age_hours:
            try:
                temp_file.unlink()  # ファイルを削除
                print(f"古い一時ファイルを削除しました: {temp_file}")
            except Exception as e:
                print(f"一時ファイルの削除に失敗しました: {e}")

#if __name__ == '__main__':
    # 起動時に古い一時ファイルをクリーンアップ
    #cleanup_temp_files()

    # Railway の PORT 環境変数を取得してアプリを起動
    #port = int(os.environ.get("PORT", 8080))  
    #app.run(host='0.0.0.0', port=port, debug=False)  # debug=False にする
