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

# フォント設定の変更（日本語対応のため）
mpl.rcParams['font.family'] = 'MS Gothic'

from functions_v1 import calculate_hourly_counts,calculate_business_time_base,calculate_business_time_order,calculate_business_time_reception,calculate_median_lt,find_best_lag_range,create_lagged_features,add_part_supplier_info,find_columns_with_word_in_name,calculate_elapsed_time_since_last_dispatch,timedelta_to_hhmmss,set_arrival_flag,drop_columns_with_word,calculate_window_width

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

file_path = file_path_arrivalflag
arrival_times_df = pd.read_csv(file_path, encoding='shift_jis')

file_path = file_path_kumitate2
kumitate_data = pd.read_csv(file_path, encoding='shift_jis')

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
st.title("在庫変動要因分析（仮）")
# 品番リスト
unique_hinban_list = df['品番'].unique()
# 品番を選択してください
product = st.selectbox("品番を選択してください", unique_hinban_list)

#品番の数だけループを回す
count = 0
for part_number in [product]:#unique_hinban_list:
    
    filtered_df = df[df['品番'] == part_number]
    unique_hinban_list2 = filtered_df['仕入先名'].unique()
    supply = str(unique_hinban_list2[0])
    count = count + 1
    print( part_number, supply, count, len(df['品番'].unique()))
    
    #part_number = str(part_number)
    
    # ある品番の1時間毎の発注かんばん数、検収かんばん数、入庫かんばん数を計算
    hourly_counts_of_order, _ , _ = calculate_hourly_counts(df, part_number, order_time_col, start_date, end_date)
    hourly_counts_of_out, _ , _ = calculate_hourly_counts(df, part_number, leave_time_col, start_date, end_date)
    hourly_counts_of_in, _ , _ = calculate_hourly_counts(df, part_number, target_time_col, start_date, end_date)
    hourly_counts_of_reception, delivery_info, reception_times = calculate_hourly_counts(df, part_number, reception_time_col, start_date, end_date)

    # 非稼動日時間をの取り除いて、発注〜入庫LT、検収〜入庫LT（日単位）の中央値を計算。これを時間遅れ計算のベースとする
    median_lt_order, median_lt_reception = calculate_median_lt(part_number,df)
    
    # 発注日時は2山ある。発注して4日後に納入せよとかある、土日のせい？
    # 発注かんばん数の最適な影響時間範囲を見つける
    min_lag =int(median_lt_order * 24)-4  # ここで最小遅れ時間を設定
    max_lag =int(median_lt_order * 24)+4  # ここで最大遅れ時間を設定
    best_corr_order, best_range_start_order, best_range_end_order = find_best_lag_range(hourly_counts_of_order, hourly_counts_of_in, min_lag, max_lag, '発注かんばん数')
    #print(f"Best range for 発注: {best_range_start_order}時間前から{best_range_end_order}時間前まで")
    #print(f"Best correlation for 発注: {best_corr_order}")

    # 検収かんばん数の最適な影響時間範囲を見つける
    #print(f"検収〜入庫LT中央値：{median_lt_reception}日,検収〜入庫時間中央値：{median_lt_reception*24}時間")
    min_lag = int(median_lt_reception * 24)-4  # ここで最小遅れ時間を設定
    max_lag = int(median_lt_reception * 24)+4  # ここで最大遅れ時間を設定
    best_corr_reception, best_range_start_reception, best_range_end_reception = find_best_lag_range(hourly_counts_of_reception, hourly_counts_of_in, min_lag, max_lag, '納入かんばん数')
    #print(f"Best range for 検収: {best_range_start_reception}時間前から{best_range_end_reception}時間前まで")
    #print(f"Best correlation for 検収: {best_corr_reception}")

    # 最適な影響時間範囲に基づいて説明変数を作成
    lagged_features_order = create_lagged_features(hourly_counts_of_order, hourly_counts_of_in, hourly_counts_of_out, best_range_start_order, best_range_end_order, '発注かんばん数', delivery_info, reception_times)
    lagged_features_reception = create_lagged_features(hourly_counts_of_reception, hourly_counts_of_in, hourly_counts_of_out, best_range_start_reception, best_range_end_reception, '納入かんばん数', delivery_info, reception_times)

    # 重複のあるtarget 列を削除
    lagged_features_reception = lagged_features_reception.drop(columns=['入庫かんばん数（t）'])
    lagged_features_reception = lagged_features_reception.drop(columns=['出庫かんばん数（t）'])
    # 合体
    lagged_features = lagged_features_order.join(lagged_features_reception, how='outer')

    #在庫増減数を計算
    lagged_features['在庫増減数(t)'] = lagged_features['入庫かんばん数（t）'] - lagged_features['出庫かんばん数（t）']

    # columns_printは'発注かんばん'を含む列名
    columns_order = find_columns_with_word_in_name(lagged_features, '発注かんばん')
    # columns_printは'発注かんばん'を含む列名
    columns_reception = find_columns_with_word_in_name(lagged_features, '納入かんばん')
    lagged_features['納入フレ（負は未納や正は挽回納入数を表す）'] = lagged_features[columns_reception] - lagged_features[columns_order]

    #発注かんばん数(t)、納入かんばん数(t)を計算
    lagged_features['発注かんばん数(t)'] = hourly_counts_of_order
    lagged_features['納入かんばん数(t)'] = hourly_counts_of_reception

    # lagged_features に品番と仕入先名を追加
    lagged_features = add_part_supplier_info(df, lagged_features, part_number)
    lagged_features = lagged_features.rename(columns={'仕入先工場名': '発送場所名'})

    # 過去の出庫からの経過時間を計算
    lagged_features, median_interval = calculate_elapsed_time_since_last_dispatch(lagged_features)

    #自動ラック在庫結合
    lagged_features = pd.merge(lagged_features, df2[['日時', '品番','在庫数（箱）']], on=['品番', '日時'], how='left')
    
    merged_data_for_robot_and_maguchi['日時'] = pd.to_datetime(merged_data_for_robot_and_maguchi['日時'])
    lagged_features = pd.merge(lagged_features, merged_data_for_robot_and_maguchi, on=['日時'], how='left')

    lagged_features = lagged_features.fillna(0)  # NaNを0に置き換え

    #平均納入時間がtimedelta64[ns]の型になっている。0days 00:00:00みたいな形
    #print(lagged_features.dtypes)

    # arrival_times_dfの仕入先名が一致する行を抽出
    # 仕入先名と発送場所名が一致する行を抽出
    matched_arrival_times_df = arrival_times_df[
        (arrival_times_df['仕入先名'].isin(lagged_features['仕入先名'])) &
        (arrival_times_df['発送場所名'].isin(lagged_features['発送場所名']))
    ]

    # arrival_times_dfの仕入先名列をlagged_featuresに結合
    lagged_features2 = lagged_features.merge(matched_arrival_times_df, on=['仕入先名','発送場所名'], how='left')


    # '納入便'を含む列名を見つける
    # columns_delivery_numは'納入便'を含む列名
    columns_delibery_num = find_columns_with_word_in_name(lagged_features2, '納入便')

    # 納入便がつく列をint型に変換
    lagged_features2[columns_delibery_num] = pd.to_numeric(lagged_features2[columns_delibery_num], errors='coerce').fillna(0).astype(int)

    # '平均納入時間' を含む列名を見つける
    # columns_delibery_timesは'平均納入時間' を含む列名
    columns_delibery_times = find_columns_with_word_in_name(lagged_features2, '平均納入時間')
    # 納入便の列に基づいて対応する「早着」「定刻」「遅着」の情報を追加
    lagged_features2['早着'] = lagged_features2.apply(lambda row: row[f'{int(row[columns_delibery_num])}便_早着'] if row[columns_delibery_num] != 0 else '00:00:00', axis=1)
    lagged_features2['定刻'] = lagged_features2.apply(lambda row: row[f'{int(row[columns_delibery_num])}便_定刻'] if row[columns_delibery_num] != 0 else '00:00:00', axis=1)
    lagged_features2['遅着'] = lagged_features2.apply(lambda row: row[f'{int(row[columns_delibery_num])}便_遅着'] if row[columns_delibery_num] != 0 else '00:00:00', axis=1)

    # timedelta形式に変換
    lagged_features2[columns_delibery_times] = pd.to_timedelta(lagged_features2[columns_delibery_times])

    # 変換を適用
    lagged_features2[columns_delibery_times] = lagged_features2[columns_delibery_times].apply(timedelta_to_hhmmss)

    # 新しい列を追加し、条件に基づいて「仕入先便到着フラグ」を設定
    lagged_features2['仕入先便到着フラグ'] = lagged_features2.apply(set_arrival_flag, columns_delibery_num=columns_delibery_num,columns_delibery_times=columns_delibery_times, axis=1)

    # 特定の文字列を含む列を削除する
    lagged_features2 = drop_columns_with_word(lagged_features2, '早着')
    lagged_features2 = drop_columns_with_word(lagged_features2, '定刻')
    lagged_features2 = drop_columns_with_word(lagged_features2, '遅着')
    lagged_features2 = drop_columns_with_word(lagged_features2, '受入')
    lagged_features2 = drop_columns_with_word(lagged_features2, '平均納入時間')

    timelag = int((best_range_start_order + best_range_end_order)/2)
    timelag2 = int((best_range_start_reception + best_range_end_reception)/2)

    # lagged_features2 と kumitate_data を日時でマージ
    kumitate_data['日時'] = pd.to_datetime(kumitate_data['日時'], errors='coerce')
    lagged_features3 = pd.merge(lagged_features2, kumitate_data[['日時', '生産台数_加重平均済','計画生産台数_加重平均済','計画達成率_加重平均済']], on='日時', how='left')

    # 影響のある生産台数を計算
    #lagged_features3 = calculate_window_width(lagged_features3, median_interval,0,timelag)

    #解析窓で計算
    lagged_features3 = calculate_window_width(lagged_features3, best_range_end_order, 0, timelag, timelag2)

    # 不要な列を削除
    #lagged_features3 = lagged_features3.drop(['生産台数_加重平均済','生産台数_加重平均済'],axis=1)

    # NaN値を処理する（例: 0で埋める）
    lagged_features3 = lagged_features3.fillna(0)
    
    lagged_features3 = pd.merge(lagged_features3, teikibin_df[['日時', '荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)']], on='日時', how='left')
    
    # columns_printは'発行かんばん'を含む列名
    columns_enter = find_columns_with_word_in_name(lagged_features3, '入庫かんばん数（t-0~')
    lagged_features3['部品置き場からの投入'] = lagged_features3[columns_enter] - lagged_features3[f'発注かんばん数（t-{timelag}~t-{timelag*2}）']

    #------------------------------------------------------------------------------------------------------------------
    #削除、今は24年度のデータがないから
    start = '2023-12-30'
    end = '2024-03-31'
    # 日付範囲に基づいてフィルタリングして削除
    lagged_features3= lagged_features3[~((lagged_features3['日時'] >= start) & (lagged_features3['日時'] <= end))]
    #------------------------------------------------------------------------------------------------------------------

    data = lagged_features3.iloc[300:]#遅れ分削除
    end_hours_ago = 0
    reception_timelag = timelag2
    #data['差分']=data[f'発注かんばん数（t-{timelag}~t-{timelag*2}）']-data[f'納入かんばん数（t-{reception_timelag}~t-{timelag+reception_timelag}）']
    # 説明変数の定義
    X = data[[f'発注かんばん数（t-{timelag}~t-{timelag*2}）',f'計画生産台数_加重平均（t-{end_hours_ago}~t-{timelag}）',f'計画達成率_加重平均（t-{end_hours_ago}~t-{timelag}）',
              '納入フレ（負は未納や正は挽回納入数を表す）','部品置き場からの投入','仕入先便到着フラグ','荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)',
              '在庫数（箱）合計_A1','在庫数（箱）合計_A2', '在庫数（箱）合計_B1', '在庫数（箱）合計_B2','在庫数（箱）合計_B3', '在庫数（箱）合計_B4']]
    # 目的変数の定義
    #y = data[f'在庫増減数（t-0~t-{timelag}）']
    y = data[f'在庫増減数(t)']

    # データを学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #--------------------------------------------------------------------------------------------------------

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
    
    unique_hinban_list = lagged_features3['仕入先名'].unique()
    supply = str(unique_hinban_list[0])
    zaikozaiko = lagged_features3['在庫数（箱）'].mean()
    
    # 結果をデータフレームに追加
    results_df = results_df.append({'品番': part_number,'仕入先名':supply,'平均在庫':zaikozaiko,'Ridge回帰の平均誤差': mae, 'Ridge回帰のマイナス方向の最大誤差': min_error_test, 'Ridge回帰のプラス方向の最大誤差': max_error_test,
                                    'ランダムフォレストの平均誤差': mae2, 'ランダムフォレストのマイナス方向の最大誤差': min_err, 'ランダムフォレストのプラス方向の最大誤差': max_err}, ignore_index=True)
                                    
    print("終了")

# 結果を表示
# 日付をindexに設定表示のため一時的に日付をindexに設定
#lagged_features3 = lagged_features3.set_index('日時')
# データフレームの各列の型を確認する
#print(lagged_features3.dtypes)
# 特定の日付範囲を指定
#temp_start_date = '2023-10-11'
#temp_end_date = '2024-01-04'
# 日付範囲でフィルタリング
#specific_date_range = lagged_features3[(lagged_features3.index >= temp_start_date) & (lagged_features3.index <= temp_end_date)]
# フィルタリングされたデータを表示
#specific_date_range = specific_date_range.reset_index()
#lagged_features3 = lagged_features3.reset_index()
#specific_date_range.head(10)

# データフレームXから100行目から300行目までのデータを選択
X_subset = X.iloc[0:5000]
# モデルを使ってX_subsetから予測値を計算
y_pred_subset = rf_model.predict(X_subset)
# y_test_subset を用意する必要がある
# この例では単に y_test の対応する部分を選択することを仮定
y_test_subset = y_test.loc[X_subset.index]
# SHAP計算
explainer = shap.TreeExplainer(rf_model, feature_dependence='tree_path_dependent', model_output='margin')
shap_values_subset = explainer.shap_values(X_subset)

# 図の作成
plt.figure(figsize=(25, 10))

start = 0
end = 2999

#df= data[data['品番'] == '019128GA010']
#data = lagged_features3
df = data.iloc[start:end]
print(df.head())

X_subset = X.iloc[start:end]
# モデルを使ってX_subsetから予測値を計算
y_pred_subset = rf_model.predict(X_subset)

df['日時'] = pd.to_datetime(df['日時'])
df.set_index('日時', inplace=True)

#df2 = df['在庫増減数（t-52~t-0）']
df2 = df['在庫数（箱）']
print(df2.head())

# プロットするデータの範囲を指定（例：最初の30個のデータを表示）
start_idx =900
end_idx =950

# プロットするデータの範囲をスライス
df2_subset = df2[start_idx:end_idx]
y_pred_subset = y_pred_subset[start_idx:end_idx]
#-----------------------------------------------------------------------
yyyy = df[f'在庫数（箱）（t-{timelag}）']
y_base_subset = yyyy[start_idx:end_idx]
#y_base_subset = df['在庫増減数(t)'].shift(1)
#-----------------------------------------------------------------------

print("df2",len(df2))
print("df2_subset",len(df2_subset))
print("y_pred_subset",len(y_pred_subset))

# 折れ線グラフのサブプロット
plt.subplot(2, 1, 1) # (rows, columns, subplot number)
plt.plot(df2_subset.index, df2_subset,  linestyle='-', color='red',label='Actual')
plt.plot(df2_subset.index, y_pred_subset+y_base_subset, linestyle='-', color='blue', label='Predicted')
#plt.title('順立装置内の在庫推移（品番：82824ECE010、品名：CONNECTOR, WIRING HARNESS）')
# データの平均値を計算
mean_value = y.mean()
# 平均線の追加
plt.axhline(y=mean_value, color='black', linestyle='--', label=f'Average: {mean_value:.2f}')
plt.legend()  # 凡例の表示
plt.xticks(ticks=df2_subset.index, labels=df2_subset.index.strftime('%Y-%m-%d-%H'), rotation=90)  # 日付フォーマットは必要に応じて調整
plt.title('順立装置内の在庫増減数（品番：9010512A018)')
plt.tight_layout()

#--------------------------------------------------------------------------

# SHAP値からデータフレームを作成
shap_df = pd.DataFrame(shap_values_subset, columns=X.columns)

# データフレームの平均SHAP値に基づいて特徴量を並び替え
shap_df_mean = shap_df.abs().mean().sort_values(ascending=False)
sorted_columns = shap_df_mean.index

shap_df_sorted = shap_df[sorted_columns]

dfdf = shap_df_sorted.iloc[start:end].T

# プロットするデータの範囲をスライス
dfdf_subset = dfdf.iloc[:, start_idx:end_idx]

print("shap_df",len(shap_df))
print("shap_df_sortedt",len(shap_df_sorted))
print("dfdf",len(dfdf))
print("dfdf_subset",len(dfdf_subset))

dfdf_subset2 = dfdf_subset

# y_base_subset の値で dfdf_subset の各列を割る
#for i, col in enumerate(dfdf_subset.columns):
    #if y_base_subset.iloc[i] == 0:
        #y_base_subset2.iloc[i] = dfdf_subset[col]
    #else:
        #dfdf_subset2[col] = dfdf_subset[col] / y_base_subset.iloc[i]
    
# データフレームの最小値と最大値を取得
#vmin = dfdf_subset2.min().min()
#vmax = dfdf_subset2.max().max()
    
# カラーマップ（ヒートマップ）のサブプロット
plt.subplot(2, 1, 2)
sns.heatmap(dfdf_subset2, cmap='bwr', cbar=True, center=0)#, vmin=vmin, vmax=vmax) # Transpose the DataFrame
plt.title('寄与度カラーマップ ')
plt.xticks(np.arange(0.5, len(df2_subset.index)), df2_subset.index.strftime('%Y-%m-%d-%H'), rotation=90)
# 各セルに値を追記
X = X.reindex(columns=shap_df_sorted.columns)
XX = X[start:end].T
XX_subset = XX.iloc[:, start_idx:end_idx]

for i in range(XX_subset.shape[0]):
    for j in range(XX_subset.shape[1]):
        formatted_value = "{:.1f}".format(XX_subset.iloc[i, j])
        text = plt.text(j+0.5, i+0.5, formatted_value,
                       ha="center", va="center", color="black")
plt.tight_layout()

filename = 'モ'+ '.png'
plt.savefig(filename)

#--------------------------------------------------------------------------------------------

df = dfdf_subset2

# カラーマップの選択
cmap = 'RdBu_r'  # 青から赤に変化するカラーマップ

# カラーマップの表示ボタン
if st.button("結果を表示"):
    st.session_state['button_clicked'] = True
    
if st.button("リセット"):
    st.session_state['button_clicked'] = False

# カラーマップの表示処理
if st.session_state['button_clicked']:
    try:
        
        #df2_subset.index = df2_subset.index.strftime('%Y-%m-%d-%H')
        df.columns = df2_subset.index.strftime('%Y-%m-%d-%H')
        
        # 折れ線グラフの表示
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df2_subset.index.strftime('%Y-%m-%d-%H'),
            y=df2_subset,
            mode='lines+markers',
            name='実績'
        ))
        
        
        # 2つ目の折れ線グラフ
        fig_line.add_trace(go.Scatter(
            x=df2_subset.index.strftime('%Y-%m-%d-%H'),
            #y=y_pred_subset+y_base_subset,
            y=y_pred_subset+df2_subset.shift(1),
            mode='lines+markers',
            name='AI推定値'
        ))
        
        fig_line.update_layout(
            title='在庫推移',
            yaxis_title='在庫数（箱）'
        )
        
        # 行の並びを反転
        df_reversed = df.iloc[::-1]

        # ヒートマップの表示
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=df_reversed.values,
            x=df_reversed.columns,
            y=df_reversed.index,
            colorscale=cmap,
            zmid=0,  # 0を中心にする
            colorbar=dict(title="影響度")  # カラーバーにタイトルを付ける
        ))

        fig_heatmap.update_layout(
            title='変動要因'
        )

        
        #print(df2_subset.index)
        #print(np.arange(len(df2_subset.index)))
        #print(df2_subset.index.strftime('%Y-%m-%d-%H'))
        #print(tickvals)
        
        st.plotly_chart(fig_line)
        st.plotly_chart(fig_heatmap)
        
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
    
print(y_base_subset)
print(y_pred_subset)


