#ライブラリのimport
import os
import pandas as pd
import warnings
import numpy as np
import pandas as pd
#%matplotlib inline#Jupyter Notebook 専用のマジックコマンド。メンテ用で利用
import matplotlib.pyplot as plt
import re
import time
import shutil
import shap
import locale
import seaborn as sns
import matplotlib as mpl
from dateutil.relativedelta import relativedelta
from IPython.display import display, clear_output
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from datetime import datetime
from datetime import timedelta
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, time

# フォント設定の変更（日本語対応のため）
mpl.rcParams['font.family'] = 'MS Gothic'

from functions_v1 import calculate_hourly_counts,calculate_business_time_base,calculate_business_time_order,calculate_business_time_reception,calculate_median_lt,find_best_lag_range,create_lagged_features,add_part_supplier_info,find_columns_with_word_in_name,calculate_elapsed_time_since_last_dispatch,timedelta_to_hhmmss,set_arrival_flag,drop_columns_with_word,calculate_window_width,display_data_app,process_shiresakibin_flag,feature_engineering

#生データのパス
folder_path_zaikoMB = '生データ/在庫推移MB'
folder_path_LTMB = '生データ/所在管理MB'
folder_path_kumitate = '生データ/組立実績MB'
folder_path_tehaisu = '生データ/手配必要数'
folder_path_tehaiunyo = '生データ/手配運用情報'
folder_path_pitch = '生データ/不等ピッチ係数'
folder_path_kotei = '生データ/使用工程'
#中間成果物のパス
folder_path_interproduct = '中間成果物'
file_path_zaikodata = '中間成果物/在庫推移MBデータ_統合済.csv'
file_path_LTdata = '中間成果物/所在管理MBデータ_統合済.csv'
file_path_kumitate = '中間成果物/組立実績MBデータ_統合済.csv'
file_path_kumitate2 = '中間成果物/組立実績MBデータ_加重済.csv'
file_path_pitch = '中間成果物/不等ピッチデータ_統合済.csv'
file_path_kotei = '中間成果物/使用工程データ_統合済.csv'
file_path_arrivalflag = '中間成果物/仕入先ダイヤフラグ.csv'
file_path_rack = '中間成果物/間口別情報.csv'
file_path_teikibin2 = '中間成果物/定期便前処理.csv'
file_path_date= '中間成果物/日付ファイル_開始日と終了日記載.txt'
file_path_zaikodata_extract = '中間成果物/在庫推移MBデータ_統合済&特定日時抽出済.csv'
file_path_LTdata_extract = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
file_path_kumitate_extract = '中間成果物/組立実績MBデータ_統合済&特定日時抽出済.csv'
file_path_tehaisu_with_tehaiunyo = '中間成果物/手配数データ_手配運用情報統合済'#.csvいらない
file_path_LTdata_extract_with_tehaisu = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済&手配数と手配運用情報統合済.csv'
file_path_weekly_data = '中間成果物/週単位のデータ.csv'
file_path_weekly_data_with_kumitate = '中間成果物/週単位のデータ_組立統合済.csv'
file_path_weekly_data_with_kumitate_and_pitch = '中間成果物/週単位のデータ_組立&不等ピッチ統合済.csv'
file_path_weekly_data_with_kumitate_and_pitch_and_kotei = '中間成果物/週単位のデータ_組立&不等ピッチ＆使用工程統合済.csv'
file_path_weekly_data_with_kumitate_and_pitch_and_kotei_and_others = '中間成果物/週単位のデータ_組立&不等ピッチ＆使用工程統合済＆必要変数追記.csv'
file_path_weekly_data_with_kumitate_and_pitch_and_kotei_and_others_cleaned = '中間成果物/週単位のデータ_組立&不等ピッチ績＆使用工程統合済＆必要変数追記_クリーニング済.csv'
file_path_daily_data='中間成果物/日単位の在庫データ.csv'
file_path_daily_tehaidata='中間成果物/日単位の手配データ.csv'
file_path_merged_daily_data = '中間成果物/日単位のデータ.csv'
file_path_merged_daily_data_with_others = '中間成果物/日単位のデータ_必要変数追加.csv'
file_path_merged_daily_data_with_others_cleaning = '中間成果物/日単位のデータ_必要変数追加_前処理済.csv'
#最終成果物のパス
folder_path_finalproduct = '最終成果物'

start_date = '2023-10-01'
end_date = '2024-03-31'

file_path = file_path_rack
merged_data_for_robot_and_maguchi = pd.read_csv(file_path, encoding='shift_jis')
merged_data_for_robot_and_maguchi['日時'] = pd.to_datetime(merged_data_for_robot_and_maguchi['日時'])

file_path = file_path_arrivalflag
arrival_times_df = pd.read_csv(file_path, encoding='shift_jis')

file_path = file_path_kumitate2
kumitate_data = pd.read_csv(file_path, encoding='shift_jis')
kumitate_data['日時'] = pd.to_datetime(kumitate_data['日時'], errors='coerce')

file_path = file_path_teikibin2
teikibin_df = pd.read_csv(file_path, encoding='shift_jis')
teikibin_df['日時'] = pd.to_datetime(teikibin_df['日時'])

#確認項目
#日付にダブりがないか

file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
df = pd.read_csv(file_path, encoding='shift_jis')
# 品番列の空白を削除
df['品番'] = df['品番'].str.strip()
# データフレームの列名を表示
#columns = df.columns.tolist()
#print(columns)

file_path = '中間成果物/在庫推移MBデータ_統合済&特定日時抽出済.csv'
df2 = pd.read_csv(file_path, encoding='shift_jis')
# 品番列の空白を削除
df2['品番'] = df2['品番'].str.strip()
# '計測日時'をdatatime型に変換
df2['計測日時'] = pd.to_datetime(df2['計測日時'], errors='coerce')
df2 = df2.rename(columns={'計測日時': '日時'})

# 印刷日時、入庫日時、出庫日時、検収日時をdatetime型に変換
df['発注日時'] = pd.to_datetime(df['発注日時'], errors='coerce')
df['印刷日時'] = pd.to_datetime(df['印刷日時'], errors='coerce')
df['順立装置入庫日時'] = pd.to_datetime(df['順立装置入庫日時'], errors='coerce')
df['順立装置出庫日時'] = pd.to_datetime(df['順立装置出庫日時'], errors='coerce')
df['検収日時'] = pd.to_datetime(df['検収日時'], errors='coerce')

# 設定
part_number = '9031150A015' #'34989ECB020'
order_time_col = '発注日時'
reception_time_col = '検収日時'
target_time_col = '順立装置入庫日時'
leave_time_col = '順立装置出庫日時'

# 結果を保存するためのデータフレームを初期化
results_df = pd.DataFrame(columns=['品番','仕入先名','平均在庫','Ridge回帰の平均誤差', 'Ridge回帰のマイナス方向の最大誤差', 'Ridge回帰のプラス方向の最大誤差',
                                       'ランダムフォレストの平均誤差', 'ランダムフォレストのマイナス方向の最大誤差', 'ランダムフォレストのプラス方向の最大誤差'],dtype=object)

# 全ての警告を無視する
warnings.filterwarnings('ignore')

# Streamlitアプリケーションのタイトル
st.title("在庫変動要因分析（仮）App")
# 品番リスト
unique_hinban_list = df['品番'].unique()

# サイドバーに品番選択ボックスを作成
st.sidebar.header("品番選択")

with st.sidebar.form(key='my_form'):
    # サイドバーに品番選択ボックスを作成
    product = st.selectbox("品番を選択してください", unique_hinban_list)
    
    # 「適用」ボタンをフォーム内に追加
    apply_button = st.form_submit_button(label='適用')

# 適用ボタンが押されたときの処理
if apply_button:
    st.write(f"選択された品番: {product}")
    # ここに実行したい処理を記述
    #execute_task(product)

#品番の数だけループを回す
count = 0
for part_number in [product]:#unique_hinban_list:
    
    filtered_df = df[df['品番'] == part_number]
    unique_hinban_list2 = filtered_df['仕入先名'].unique()
    supply = str(unique_hinban_list2[0])
    count = count + 1
    print( part_number, supply, count, len(df['品番'].unique()))
    
    #part_number = str(part_number)
    
    # タイムスタンプ系のデータ処理
    # ある品番の1時間毎の発注かんばん数、検収かんばん数、入庫かんばん数を計算
    hourly_counts_of_order, _ , _ = calculate_hourly_counts(df, part_number, order_time_col, start_date, end_date)
    hourly_counts_of_out, _ , _ = calculate_hourly_counts(df, part_number, leave_time_col, start_date, end_date)
    hourly_counts_of_in, _ , _ = calculate_hourly_counts(df, part_number, target_time_col, start_date, end_date)
    hourly_counts_of_reception, delivery_info, reception_times = calculate_hourly_counts(df, part_number, reception_time_col, start_date, end_date)

    # 非稼動日時間をの取り除いて、発注〜入庫LT、検収〜入庫LT（日単位）の中央値を計算。
    # 時間遅れ計算のベースとする
    median_lt_order, median_lt_reception = calculate_median_lt(part_number,df)
    
    # 発注日時は2山ある。発注して4日後に納入せよとかある、土日の影響？（要確認）
    # 発注かんばん数の最適な影響時間範囲を見つける
    min_lag =int(median_lt_order * 24)-4  # LT中央値を基準に最小遅れ時間を設定
    max_lag =int(median_lt_order * 24)+4  # LT中央値を基準に最大遅れ時間を設定
    best_corr_order, best_range_start_order, best_range_end_order = find_best_lag_range(hourly_counts_of_order, hourly_counts_of_in, min_lag, max_lag, '発注かんばん数')

    # 検収スタンプは伝票単位＆リアルタイムではないため、信用できない
    # 検収かんばん数の最適な影響時間範囲を見つける
    min_lag = int(median_lt_reception * 24)-4  # LT中央値を基準に最小遅れ時間を設定
    max_lag = int(median_lt_reception * 24)+4  # LT中央値を基準に最大遅れ時間を設定
    best_corr_reception, best_range_start_reception, best_range_end_reception = find_best_lag_range(hourly_counts_of_reception, hourly_counts_of_in, min_lag, max_lag, '納入かんばん数')
    
    # 確認用
    #print(f"Best range for 発注: {best_range_start_order}時間前から{best_range_end_order}時間前まで")
    #print(f"Best correlation for 発注: {best_corr_order}")
    #print(f"検収〜入庫LT中央値：{median_lt_reception}日,検収〜入庫時間中央値：{median_lt_reception*24}時間")
    #print(f"Best range for 検収: {best_range_start_reception}時間前から{best_range_end_reception}時間前まで")
    #print(f"Best correlation for 検収: {best_corr_reception}")

    # 最適な影響時間範囲に基づいて説明変数を作成
    lagged_features_order = create_lagged_features(hourly_counts_of_order, hourly_counts_of_in, hourly_counts_of_out, best_range_start_order, best_range_end_order, '発注かんばん数', delivery_info, reception_times)
    lagged_features_reception = create_lagged_features(hourly_counts_of_reception, hourly_counts_of_in, hourly_counts_of_out, best_range_start_reception, best_range_end_reception, '納入かんばん数', delivery_info, reception_times)

    # 前処理
    # 重複のあるtarget 列を削除
    lagged_features_reception = lagged_features_reception.drop(columns=['入庫かんばん数（t）'])
    lagged_features_reception = lagged_features_reception.drop(columns=['出庫かんばん数（t）'])
    # lagged_features作成
    # 最適な影響時間範囲に基づいた発注かんばん数と、検収かんばん数を統合
    lagged_features = lagged_features_order.join(lagged_features_reception, how='outer')

    #lagged_featuresに情報追加
    lagged_features['在庫増減数(t)'] = lagged_features['入庫かんばん数（t）'] - lagged_features['出庫かんばん数（t）']#在庫増減数を計算
    lagged_features['発注かんばん数(t)'] = hourly_counts_of_order#発注かんばん数(t)を計算
    lagged_features['納入かんばん数(t)'] = hourly_counts_of_reception#納入かんばん数(t)を計算
    lagged_features = add_part_supplier_info(df, lagged_features, part_number)#品番と仕入先名を追加
    lagged_features = lagged_features.rename(columns={'仕入先工場名': '発送場所名'})
    lagged_features, median_interval = calculate_elapsed_time_since_last_dispatch(lagged_features)# 過去の出庫からの経過時間を計算
    lagged_features = pd.merge(lagged_features, df2[['日時', '品番','在庫数（箱）']], on=['品番', '日時'], how='left')#自動ラック在庫結合
    lagged_features = pd.merge(lagged_features, merged_data_for_robot_and_maguchi, on=['日時'], how='left')#1時間ああたりの間口別在庫の計算
    # lagged_features の各列の型をチェックして、適切な値で NaN を置き換える
    for col in lagged_features.columns:
        if pd.api.types.is_timedelta64_dtype(lagged_features[col]):
            lagged_features[col] = lagged_features[col].fillna(pd.Timedelta(0))
        else:
            lagged_features[col] = lagged_features[col].fillna(0)
    lagged_features = process_shiresakibin_flag(lagged_features, arrival_times_df)#仕入先便到着フラグ計算
    lagged_features = pd.merge(lagged_features,kumitate_data[['日時','生産台数_加重平均済','計画生産台数_加重平均済','計画達成率_加重平均済']], on='日時', how='left')# lagged_features と kumitate_data を日時でマージ
    
    #timelag = int((best_range_start_order + best_range_end_order)/2)#最適な発注かんばん数の幅
    #timelag12 = int((best_range_start_reception + best_range_end_reception)/2)#最適な納入かんばん数の幅
    
    best_range_order = int((best_range_start_order + best_range_end_order)/2)#最適な発注かんばん数の幅
    best_range_reception = int((best_range_start_reception + best_range_end_reception)/2)#最適な納入かんばん数の幅
    
    #定期便
    lagged_features = pd.merge(lagged_features, teikibin_df[['日時', '荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)']], on='日時', how='left')
    #特徴量エンジニアリング
    lagged_features = feature_engineering(lagged_features)

    #ローリング特徴量
    #解析窓で計算
    lagged_features = calculate_window_width(lagged_features, best_range_end_order, 0, best_range_order, best_range_reception)

    # 不要な列を削除
    #lagged_features = lagged_features.drop(['生産台数_加重平均済','生産台数_加重平均済'],axis=1)

    # NaN値を処理する（例: 0で埋める）
    lagged_features = lagged_features.fillna(0)
    
    
    # columns_printは'発行かんばん'を含む列名
    columns_enter = find_columns_with_word_in_name(lagged_features, '入庫かんばん数（t-0~')
    lagged_features['部品置き場からの投入'] = lagged_features[columns_enter] - lagged_features[f'発注かんばん数（t-{best_range_order}~t-{best_range_order*2}）']
    # columns_printは'発注かんばん'を含む列名
    columns_order = find_columns_with_word_in_name(lagged_features, '発注かんばん数（t-')
    # columns_printは'発注かんばん'を含む列名
    columns_reception = find_columns_with_word_in_name(lagged_features, '納入かんばん数（t-')
    lagged_features['納入フレ（負は未納や正は挽回納入数を表す）'] = lagged_features[columns_reception] - lagged_features[columns_order]
    

    #------------------------------------------------------------------------------------------------------------------
    #削除、今は24年度のデータがないから
    start = '2023-12-30'
    end = '2024-03-31'
    # 日付範囲に基づいてフィルタリングして削除
    lagged_features= lagged_features[~((lagged_features['日時'] >= start) & (lagged_features['日時'] <= end))]
    #------------------------------------------------------------------------------------------------------------------

    data_temp = lagged_features#遅れ分削除
    data = data_temp.iloc[300:]#遅れ分削除
    end_hours_ago = 0
    reception_timelag = best_range_reception
    #data['差分']=data[f'発注かんばん数（t-{timelag}~t-{timelag*2}）']-data[f'納入かんばん数（t-{reception_timelag}~t-{timelag+reception_timelag}）']
    # 説明変数の定義
    X = data[[f'発注かんばん数（t-{best_range_order}~t-{best_range_order*2}）',f'計画生産台数_加重平均（t-{end_hours_ago}~t-{best_range_order}）',f'計画達成率_加重平均（t-{end_hours_ago}~t-{best_range_order}）','部品置き場からの投入',
              '納入フレ（負は未納や正は挽回納入数を表す）','仕入先便到着フラグ','荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)',
              f'間口_A1の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_A2の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B1の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B2の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_B3の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B4の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'部品置き場からの入庫（t-{end_hours_ago}~t-{best_range_order}）',f'部品置き場で滞留（t-{end_hours_ago}~t-{best_range_order}）',f'定期便にモノ無し（t-{end_hours_ago}~t-{best_range_order}）']]
    # 目的変数の定義
    #★
    y = data[f'在庫増減数（t-0~t-{best_range_order}）']
    #y = data[f'在庫増減数(t)']

    # データを学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Lasso回帰モデルの作成
    ridge = Ridge(alpha=0.1)
    # モデルの訓練
    ridge.fit(X_train, y_train)
    # 予測
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)
    # 評価
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    max_error_train = max_error(y_train, y_pred_train)
    max_error_test = max_error(y_test, y_pred_test)
    # マイナス方向の最大誤差を計算
    min_error_train = np.min(y_train - y_pred_train)
    min_error_test = np.min(y_test - y_pred_test)

    #print(f'Ridge回帰 - 訓練データのMSE: {mse_train}')
    #print(f'Ridge回帰 - テストデータのMSE: {mse_test}')
    #print(f'Ridge回帰 - 訓練データの最大誤差: {max_error_train}')
    #print(f'Ridge回帰 - テストデータの最大誤差: {max_error_test}')
    #print(f'Ridge回帰 - 訓練データのマイナス方向の最大誤差: {min_error_train}')
    #print(f'Ridge回帰 - テストデータのマイナス方向の最大誤差: {min_error_test}')
    # 平均誤差を計算
    mae = mean_absolute_error(y_test, y_pred_test)
    #print(f'Ridge回帰 - テストデータの平均誤差: {mae}')

    #--------------------------------------------------------------------------------------------------------

    # ランダムフォレストモデルの訓練
    rf_model = RandomForestRegressor(n_estimators=10, max_depth=20,random_state=42)
    rf_model.fit(X_train, y_train)
    # テストデータで予測し、MSEを計算
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'ランダムフォレスト - テストデータのMSE: {mse}')
    # 最大誤差を計算
    max_err = max_error(y_test, y_pred)
    print(f'ランダムフォレスト - テストデータの最大誤差: {max_err}')
    # マイナス方向の最大誤差を計算
    min_err = np.min(y_test - y_pred)
    print(f'ランダムフォレスト - テストデータのマイナス方向の最大誤差: {min_err}')
    # 平均誤差を計算
    mae2 = mean_absolute_error(y_test, y_pred)
    print(f'ランダムフォレスト - テストデータの平均誤差: {mae2}')
    #--------------------------------------------------------------------------------------------------------
    
    unique_hinban_list = lagged_features['仕入先名'].unique()
    supply = str(unique_hinban_list[0])
    zaikozaiko = lagged_features['在庫数（箱）'].mean()
    
    # 結果をデータフレームに追加
    results_df = results_df.append({'品番': part_number,'仕入先名':supply,'平均在庫':zaikozaiko,'Ridge回帰の平均誤差': mae, 'Ridge回帰のマイナス方向の最大誤差': min_error_test, 'Ridge回帰のプラス方向の最大誤差': max_error_test,
                                    'ランダムフォレストの平均誤差': mae2, 'ランダムフォレストのマイナス方向の最大誤差': min_err, 'ランダムフォレストのプラス方向の最大誤差': max_err}, ignore_index=True)
                                    
    print("終了")

# データフレームXから100行目から300行目までのデータを選択
#X_subset = X.iloc[0:3000]
# モデルを使ってX_subsetから予測値を計算
#y_pred_subset = rf_model.predict(X_subset)
# y_test_subset を用意する必要がある
# この例では単に y_test の対応する部分を選択することを仮定
#y_test_subset = y_test.loc[X_subset.index]

#インデックスが300スタートなのでリセット
data = data.reset_index(drop=True)

# サイドバーで開始日と終了日を選択
st.sidebar.header("在庫推移可視化")

# データの最小日時と最大日時を取得
min_datetime = data['日時'].min()
max_datetime = data['日時'].max()

print(min_datetime,max_datetime)

default_values = {
    'start_date': min_datetime.date(),
    'end_date': max_datetime.date(),
    'start_time': datetime.strptime("00:00", "%H:%M").time(),  # 0:00として初期化
    'end_time': datetime.strptime("23:00", "%H:%M").time(),  # 23:00として初期化
    'button_clicked': False
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# サイドバーにフォームの作成
with st.sidebar.form(key='filter_form'):
    st.session_state.start_date = st.date_input("開始日", st.session_state.start_date)
    st.session_state.end_date = st.date_input("終了日", st.session_state.end_date)
    start_time_hours = st.slider("開始時間", 0, 23, st.session_state.start_time.hour, format="%02d:00")
    end_time_hours = st.slider("終了時間", 0, 23, st.session_state.end_time.hour, format="%02d:00")
    
    # 時間を更新
    st.session_state.start_time = time(start_time_hours, 0)
    st.session_state.end_time = time(end_time_hours, 0)
    
    # フォームの送信ボタン
    submit_button = st.form_submit_button(label='適用')

    if submit_button:
        st.session_state.button_clicked = True

# フォームが送信された場合の処理
if submit_button:
    st.session_state.button_clicked = True

# ボタンが押された場合のみ処理を実行
if st.session_state.button_clicked:
    # 開始日時と終了日時を結合
    start_datetime = datetime.combine(st.session_state.start_date, st.session_state.start_time)
    end_datetime = datetime.combine(st.session_state.end_date, st.session_state.end_time)

    # start_datetimeとend_datetimeに対応するインデックスを見つける
    start_index = data.index[data['日時'] == start_datetime].tolist()
    end_index = data.index[data['日時'] == end_datetime].tolist()

    st.write(f"開始日時: {start_datetime}, インデックス: {start_index}")
    st.write(f"終了日時: {end_datetime}, インデックス: {end_index}")
else:
    st.write("開始日、終了日、開始時間、終了時間を選択し、実行ボタンを押してください。")
    
    
# SHAP計算
explainer = shap.TreeExplainer(rf_model, feature_dependence='tree_path_dependent', model_output='margin')
shap_values = explainer.shap_values(X)

first_datetime_df = data['日時'].iloc[0]
print(f"dataの日時列の最初の値: {first_datetime_df}")

# リストから整数に変換
start_index_int = start_index[0]#-300
end_index_int = end_index[0]#-300

#start = start_index_int#0#0
#end = end_index_int#2999

df = data.iloc[start_index_int:end_index_int]
print(df.head())

first_datetime_df = df.iloc[0]
print(f"dfの日時列の最初の値: {first_datetime_df}")

#X_subset = X.iloc[start:end]
# モデルを使ってX_subsetから予測値を計算
#y_pred_subset = rf_model.predict(X_subset)

df['日時'] = pd.to_datetime(df['日時'])
df.set_index('日時', inplace=True)

#df2 = df['在庫増減数（t-52~t-0）']
df2 = df['在庫数（箱）']
print(df2.head())

#在庫数（箱）を計算する
#yyyy = df[f'在庫数（箱）（t-{best_range_order}）']
#y_base_subset = yyyy[start_idx:end_idx]

#在庫増減数の平均値を確認用
#mean_value = y.mean()

# SHAP値からデータフレームを作成
shap_df = pd.DataFrame(shap_values, columns=X.columns)

# データフレームの平均SHAP値に基づいて特徴量を並び替え
shap_df_mean = shap_df.abs().mean().sort_values(ascending=False)
sorted_columns = shap_df_mean.index

shap_df_sorted = shap_df[sorted_columns]

dfdf = shap_df_sorted.iloc[start_index_int:end_index_int].T

# プロットするデータの範囲をスライス
dfdf_subset = dfdf#.iloc[:, start_idx:end_idx]

dfdf_subset2 = dfdf_subset

# 前の値との差分を計算
# 差分と差分判定をfor文で計算
#difference = [None]  # 最初の差分はなし
#for i in range(1,len(df2_subset)):
#    diff = df2_subset.iloc[i] - df2_subset.iloc[i-1]
#    difference.append(diff)
#    if i < len(dfdf_subset2):
#        if diff > 0:
#            dfdf_subset2.iloc[i] = dfdf_subset2.iloc[i]
#        elif diff < 0:
#            dfdf_subset2.iloc[i] = -1*dfdf_subset2.iloc[i]

#--------------------------------------------------------------------------------------------

df = dfdf_subset2

# dfの列数とdf2_subsetのインデックス数を確認
print(f"data index: {len(data)}")
print(f"df columns: {len(df.columns)}")
#print(f"df2_subset index: {len(df2_subset.index)}")
print(f"shap_df_sorted index: {len(shap_df_sorted)}")
print(f"dfdf index: {len(dfdf)}")
print(f"dfdf_subset2 index: {len(dfdf_subset2)}")


# カラーマップの選択
cmap = 'RdBu_r'  # 青から赤に変化するカラーマップ

#df2_subset.index = df2_subset.index.strftime('%Y-%m-%d-%H')
df.columns = df2.index.strftime('%Y-%m-%d-%H')

#行の並びを反転
df_reversed = df.iloc[::-1]

# インデックスをリセット
df2_subset_df = df2.to_frame().reset_index()

# データフレームの行と列を入れ替え
df_transposed = df.transpose()
# インデックスをリセットして日時列を作成
df_transposed.reset_index(inplace=True)
# インデックス列の名前を '日時' に変更
df_transposed.rename(columns={'index': '日時'}, inplace=True)

#説明変数
zzz = X.iloc[start_index_int:end_index_int]#[start_idx:end_idx]
# インデックスをリセット
zzz = zzz.reset_index(drop=True)
#日時列
temp_time = df_transposed.reset_index(drop=True)

first_datetime_df1 = data['日時'].iloc[0]
first_datetime_df2 = temp_time['日時'].iloc[0]
first_datetime_df3 = df_transposed['日時'].iloc[0]
print(f"dataの日時列の最初の値: {first_datetime_df1}")
print(f"df_transposedの日時列の最初の値: {first_datetime_df3}")
print(f"temp_timeの日時列の最初の値: {first_datetime_df2}")

# data1とdata2を結合
merged_df = pd.concat([temp_time[['日時']], zzz], axis=1)

# 関数を呼び出して表示
display_data_app(df2_subset_df, df_transposed, merged_df)

# 全体SHAPプロットの生成
#fig, ax = plt.subplots()
#shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
# プロットをStreamlitで表示
#st.pyplot(fig)

