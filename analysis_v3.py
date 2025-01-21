#分析用

#! ライブラリのimport
import os
import pandas as pd
import warnings
import numpy as np
import pandas as pd
#%matplotlib inline#Jupyter Notebook 専用のマジックコマンド。メンテ用で利用
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import matplotlib as mpl
import re
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
import pickle
from sklearn.preprocessing import StandardScaler

#! 自作ライブラリのimport
#データ読み取り用
from main_v3 import create_hinban_info
from read_v3 import read_data, process_Activedata, read_activedata_from_IBMDB2
#データ前処理用
from functions_v3 import display_message,display_corr_matrix, calculate_hourly_counts,calculate_business_time_base,calculate_business_time_order, \
    calculate_business_time_reception,calculate_median_lt,find_best_lag_range,create_lagged_features,add_part_supplier_info, \
        find_columns_with_word_in_name,calculate_elapsed_time_since_last_dispatch, \
            calculate_window_width,process_shiresakibin_flag,feature_engineering, \
                plot_inventory_graph, display_shap_contributions,plot_inventory_graph2

#! フォント設定の変更（日本語対応のため）
mpl.rcParams['font.family'] = 'MS Gothic'
    
def show_analysis(product):

    #!学習期間（解析期間）任意に設定できるように。直近1年とかで
    #* ＜ローカルデータ利用する場合＞
    start_date = '2024-05-01-00'
    end_date = '2024-08-31-00'
    #*＜実行時間で日時を選択する場合＞
    #current_time = datetime.now()# 現在の実行時間を取得
    #end_date = (current_time - timedelta(days=1)).strftime('%Y-%m-%d-%H')# end_dateを実行時間の前日
    #start_date = (current_time - timedelta(days=1) - timedelta(days=180)).strftime('%Y-%m-%d-%H')# start_dateを実行時間の前日からさらに半年前

    #! 前処理済みのデータをダウンロード
    #! 実行内容の説明
    # 実行状態の表示
    display_message("**ステップ１の処理を受け付けました。データの抽出処理を開始します。**")
    #! データの抽出
    AutomatedRack_Details_df, arrival_times_df, kumitate_df, teikibin_df, Timestamp_df, zaiko_df = read_data(start_date, end_date)
    # 実行状態の表示
    display_message("**データの抽出処理が完了しました。次にデータの前処理と統合処理を開始します。**")

    #! 稼働フラグの計算
    # 同じ日時ごとに入庫数・出庫数を合計
    kado_df = zaiko_df.groupby('日時')[['入庫数（箱）', '出庫数（箱）']].sum().reset_index()
    kado_df["入出庫数（箱）"] = kado_df["入庫数（箱）"]-kado_df["出庫数（箱）"]
    # '入庫数（箱）'か'出庫数（箱）'のどちらかがX個以上なら稼働フラグを1とする
    x = 5
    kado_df['稼働フラグ'] = ((kado_df['入庫数（箱）'] >= x) | (kado_df['出庫数（箱）'] >= x)).astype(int)
    display_message("**稼働時間の計算が完了しました。**")
    st.dataframe(kado_df.head(50000))

    #! 設定
    order_time_col = '発注日時'
    reception_time_col = '検収日時'
    target_time_col = '順立装置入庫日時'
    leave_time_col = '順立装置出庫日時'

    #! 全ての警告を無視する
    warnings.filterwarnings('ignore')
        
    #-------------------------------------------------------------
    
    #! 結果を保存するためのデータフレームを初期化
    results_df = pd.DataFrame(columns=['品番','仕入先名','平均在庫','Ridge回帰の平均誤差', 'Ridge回帰のマイナス方向の最大誤差', 'Ridge回帰のプラス方向の最大誤差',
                                           'ランダムフォレストの平均誤差', 'ランダムフォレストのマイナス方向の最大誤差', 'ランダムフォレストのプラス方向の最大誤差'],dtype=object)
    

    #! 全品番の傾向確認
    # 品番ごとの発注検収LTの中央値を計算
    hatyukensyu_median_lt = Timestamp_df.groupby('品番')['発注検収LT'].median().reset_index()
    hatyukensyu_median_lt.columns = ['品番', '発注検収LTの中央値']
    # 発注検収LTの中央値を大きい順にソート
    hatyukensyu_median_lt = hatyukensyu_median_lt.sort_values(by='発注検収LTの中央値', ascending=False).reset_index(drop=True)
    display_message("**実績のかんばん回転日数を計算しました。**")
    st.dataframe(hatyukensyu_median_lt)

    #! 品番の数だけループを回す
    #! 今は1品番で
    count = 0
    for unique_product in [product]:
    #!　以下は全品番動作テスト用
    #!　ユニークな '品番_整備室' 列を作成し、for分で回す
    #hinban_seibishitsu_df = create_hinban_info()
    #for unique_product in hinban_seibishitsu_df['品番_整備室']:
        #count = count + 1

        #if count < 260:
            #continue
        
        # 確認用：実行時の条件確認
        # filtered_Timestamp_df = Timestamp_df[Timestamp_df['品番'] == part_number]#特定品番のデータを抽出
        # suppliers = filtered_Timestamp_df['仕入先名'].unique()#該当仕入先名を抽出
        # supplier = str(suppliers[0])
        # count = count + 1
        # print("品番：", part_number)
        # print("仕入先名：", supplier)
        # print("ユニークな品番の数：", len(Timestamp_df['品番'].unique()))
        # print("ループ：", count)

        #! 品番、整備室コードを抽出
        part_number = unique_product.split('_')[0]
        seibishitsu = unique_product.split('_')[1]

        #! 実行結果の確認
        # 実行状態の表示
        display_message(f"**品番{part_number}_整備室コード{seibishitsu}に対して処理を開始します。**")

        # if part_number == "01912ECB040":
        #     continue#スキップ
        
        #! 内容：関所毎のかんばん数（1時間単位）を計算
        #! Args：関所毎のタイムスタンプデータ、開始時間、終了時間
        #! Return：関所毎のかんばん数（1時間単位）
        hourly_counts_of_order, _ , _ , kyoten = calculate_hourly_counts(Timestamp_df, part_number, seibishitsu, order_time_col, start_date, end_date)#発注
        hourly_counts_of_reception, delivery_info, reception_times, _ = calculate_hourly_counts(Timestamp_df, part_number, seibishitsu, reception_time_col, start_date, end_date)#検収
        hourly_counts_of_in, _ , _ , _  = calculate_hourly_counts(Timestamp_df, part_number, seibishitsu, target_time_col, start_date, end_date)#入庫
        hourly_counts_of_out, _ , _ , _ = calculate_hourly_counts(Timestamp_df, part_number, seibishitsu, leave_time_col, start_date, end_date)#出庫

        #! 内容：時間遅れを計算。発注から入庫までの時間、検収から入庫までの時間を計算（非稼動日時間をの取り除いて）
        #! Args：品番、関所毎のタイムスタンプデータ
        #! Return：発注〜入庫LT、検収〜入庫LT（日単位）の中央値
        median_lt_order, median_lt_reception = calculate_median_lt(part_number, Timestamp_df)
        #median_lt_order = median_lt_order*24
        #median_lt_reception = median_lt_reception*24
        display_message(f"**休日を削除した場合、発注入庫LT={median_lt_order}と検収入庫LT={median_lt_reception}となりました**")
        #st.dataframe(hourly_counts_of_order)
        #todo メモ：常に部品置き場などで滞留していることもあり、品番G117362010_Yは検収入庫LTの中央値が10になる。理論上は5程度？
        filtered_Timestamp_df = Timestamp_df[Timestamp_df['品番'] == part_number]
        kensyu = filtered_Timestamp_df["発注検収LT"].median()
        nyuuko = filtered_Timestamp_df["発注順立装置入庫LT"].median()
        kaisyu = filtered_Timestamp_df["発注回収LT"].median()
        display_message(f"**休日を削除しない場合、発注検収LT={kensyu}と発注順立装置入庫LT={nyuuko}と発注回収LT={kaisyu}となりました**")
        WWWW = kaisyu - kensyu
        
        # Todo：発注日時は2山ある。発注して4日後に納入せよとかある、土日の影響？
        #! 内容：発注かんばん数の最適な影響時間範囲を見つける
        #! Args：1時間ごとの発注かんばん数、1時間ごとの入庫かんばん数、探索時間範囲
        #! Return：最適相関値、最適開始遅れ、終了範囲遅れ
        min_lag =int(median_lt_order * 24)-4  # LT中央値を基準に最小遅れ時間を設定
        max_lag =int(median_lt_order * 24)+4  # LT中央値を基準に最大遅れ時間を設定
        best_corr_order, best_range_start_order, best_range_end_order = find_best_lag_range(hourly_counts_of_order, hourly_counts_of_in, min_lag, max_lag, '発注かんばん数')

        #! 内容：検収かんばん数の最適な影響時間範囲を見つける
        #! Args：1時間ごとの検収かんばん数、1時間ごとの入庫かんばん数、探索時間範囲
        #! Return：最適相関値、最適開始遅れ、終了範囲遅れ
        min_lag = int(median_lt_reception * 24)-4  # LT中央値を基準に最小遅れ時間を設定
        max_lag = int(median_lt_reception * 24)+4  # LT中央値を基準に最大遅れ時間を設定
        best_corr_reception, best_range_start_reception, best_range_end_reception = find_best_lag_range(hourly_counts_of_reception, hourly_counts_of_in, min_lag, max_lag, '納入かんばん数')
        
        # 確認用：実行結果の確認
        #print(f"Best range for 発注: {best_range_start_order}時間前から{best_range_end_order}時間前まで")
        #print(f"Best correlation for 発注: {best_corr_order}")
        #print(f"検収〜入庫LT中央値：{median_lt_reception}日,検収〜入庫時間中央値：{median_lt_reception*24}時間")
        #print(f"Best range for 検収: {best_range_start_reception}時間前から{best_range_end_reception}時間前まで")
        #print(f"Best correlation for 検収: {best_corr_reception}")
        
        #st.header("納入時間の確認")
        #st.dataframe(reception_times)
        #st.dataframe(delivery_info)

        #! 内容：最適な影響時間範囲に基づいて発注かんばん数と検収かんばん数を計算
        #! Args：1時間ごとの発注かんばん数、1時間ごとの入庫かんばん数、最適時間遅れ範囲
        #! Return：最適時間遅れで計算した発注かんばん数、入庫かんばん数
        lagged_features_order = create_lagged_features(hourly_counts_of_order, hourly_counts_of_in, hourly_counts_of_out, best_range_start_order, best_range_end_order, '発注かんばん数', delivery_info, reception_times)
        lagged_features_reception = create_lagged_features(hourly_counts_of_reception, hourly_counts_of_in, hourly_counts_of_out, best_range_start_reception, best_range_end_reception, '納入かんばん数', delivery_info, reception_times)
        # 前処理：重複のあるtarget 列を削除
        lagged_features_reception = lagged_features_reception.drop(columns=['入庫かんばん数（t）'])
        lagged_features_reception = lagged_features_reception.drop(columns=['出庫かんばん数（t）'])
        # 最適な影響時間範囲に基づいた発注かんばん数と、検収かんばん数を統合
        lagged_features = lagged_features_order.join(lagged_features_reception, how='outer')

        reception_times = reception_times.to_frame()
        lagged_features = pd.merge(lagged_features, reception_times, on=['イベント時間'], how='left')
        delivery_info = delivery_info.to_frame()
        lagged_features = pd.merge(lagged_features, delivery_info, on=['イベント時間'], how='left')

        #確認：実行結果
        display_message(f"**1時間あたりの関所別のかんばん数の計算が完了しました。**")
        st.dataframe(lagged_features)

        #! lagged_featuresに変数追加
        #! Result：「拠点所番地」列、「整備室コード」列の追加
        lagged_features['品番'] = part_number
        lagged_features['拠点所番地'] = kyoten
        lagged_features['整備室コード'] = seibishitsu

        #! lagged_featuresに変数追加
        #! Result：「在庫増減数（t）」列、「発注かんばん数（t）」列、「納入かんばん数（t）」列の追加
        lagged_features['在庫増減数（t）'] = lagged_features['入庫かんばん数（t）'] - lagged_features['出庫かんばん数（t）']#在庫増減数を計算
        lagged_features['発注かんばん数（t）'] = hourly_counts_of_order# 発注かんばん数(t)を計算
        lagged_features['納入かんばん数（t）'] = hourly_counts_of_reception# 納入かんばん数(t)を計算

        #! lagged_featuresに変数追加
        #! What：「品番」列と「整備室コード」列をもとに、「仕入先名」列、「発送場所名」列を探し、統合
        #! Result：「仕入先名」列、「発送場所名（名称変更。旧仕入れ先工場名）」列の追加
        lagged_features = add_part_supplier_info(Timestamp_df, lagged_features, seibishitsu)
        lagged_features = lagged_features.rename(columns={'仕入先工場名': '発送場所名'})# コラム名変更

        #! 過去の出庫からの経過時間を計算
        lagged_features, median_interval_out = calculate_elapsed_time_since_last_dispatch(lagged_features)

        # 0を除いた数値の中央値を計算
        # median_value_out = lagged_features[lagged_features['出庫かんばん数（t）'] != 0]['出庫かんばん数（t）'].median()

        #! lagged_featuresに変数追加
        #! What：「拠点所番地」列をもとに在庫数を紐づける
        # まず、無効な値を NaN に変換
        zaiko_df['拠点所番地'] = pd.to_numeric(zaiko_df['拠点所番地'], errors='coerce')
        # 品番ごとに欠損値（NaN）を埋める(前方埋め後方埋め)
        zaiko_df['拠点所番地'] = zaiko_df.groupby('品番')['拠点所番地'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
        # それでも置換できないものはNaN を 0 で埋める
        zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].fillna(0).astype(int).astype(str)
        # 両方のデータフレームの '拠点所番地' 列を文字列型に変換
        lagged_features['拠点所番地'] = lagged_features['拠点所番地'].astype(int).astype(str)
        zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].astype(int).astype(str)
        lagged_features = pd.merge(lagged_features, zaiko_df[['日時', '品番', '在庫数（箱）','拠点所番地']], on=['品番', '日時', '拠点所番地'], how='left')#! 自動ラック在庫結合
        
        #! 在庫データの欠損時間を埋める
        # '日時' 列でデータをソート
        lagged_features = lagged_features.sort_values(by=['品番', '日時'])
        # 在庫数（箱）が NULL の場合、前の時間の在庫数（箱）で補完
        #lagged_features['在庫数（箱）'] = lagged_features.groupby('品番')['在庫数（箱）'].transform(lambda x: x.fillna(method='ffill'))
        # todo 在庫数（箱）がNULLのとき、前の時間の在庫増減数（t）+在庫数（t）で補完する
        for idx in lagged_features.index:
            if pd.isnull(lagged_features.loc[idx,'在庫数（箱）']):
                if idx > 0:
                    lagged_features.loc[idx,'在庫数（箱）'] = lagged_features.loc[idx - 1,'在庫数（箱）'] + lagged_features.loc[idx-1,'在庫増減数（t）']

        lagged_features = pd.merge(lagged_features, AutomatedRack_Details_df, on=['日時'], how='left')#! 1時間ああたりの間口別在庫の計算
        for col in lagged_features.columns:
            if pd.api.types.is_timedelta64_dtype(lagged_features[col]):
                lagged_features[col] = lagged_features[col].fillna(pd.Timedelta(0))
            else:
                lagged_features[col] = lagged_features[col].fillna(0)

        #! 仕入先便到着フラグ計算
        #! 一致する仕入れ先フラグが見つからない場合、エラーを出す
        lagged_features, matched_arrival_times_df = process_shiresakibin_flag(lagged_features, arrival_times_df)

        #! lagged_features と kumitate_df を日時で統合
        lagged_features = pd.merge(lagged_features, kumitate_df[['日時','整備室コード','生産台数_加重平均済','計画生産台数_加重平均済','計画達成率_加重平均済']], on=['日時', '整備室コード'], how='left')
    
        #! 最適な遅れ時間を計算
        best_range_order = int((best_range_start_order + best_range_end_order)/2)#最適な発注かんばん数の幅
        best_range_reception = int((best_range_start_reception + best_range_end_reception)/2)#最適な納入かんばん数の幅
        #st.write(f"発注入庫LT：{best_range_order},検収入庫LT：{best_range_reception}")

        #todo 
        # 所在管理データの準備
        Timestamp_df = Timestamp_df.rename(columns={'仕入先工場名': '発送場所名'})# コラム名変更
        shiresaki = lagged_features['仕入先名'].unique()
        shiresaki_hassou = lagged_features['発送場所名'].unique()
        Timestamp_filtered_df = Timestamp_df[(Timestamp_df['品番'] == part_number) & (Timestamp_df['仕入先名'] == shiresaki[0]) & (Timestamp_df['発送場所名'] == shiresaki_hassou[0]) & (Timestamp_df['整備室コード'] == seibishitsu)]# 条件を満たす行を抽出
        #st.header("かんばんデータ確認")
        #st.dataframe(Timestamp_filtered_df.head(10000))
        # 仕入先ダイヤの準備
        matched_arrival_times_df = matched_arrival_times_df.rename(columns={'受入': '整備室コード'})# コラム名変更
        # 統合する列の選別
        columns_to_extract_t = ['かんばんシリアル','納入日', '納入便','検収日時','仕入先名', '発送場所名', '整備室コード']
        columns_to_extract_l = matched_arrival_times_df.filter(regex='便_定刻').columns.tolist() + ['仕入先名', '発送場所名', '整備室コード']
        # 統合
        Timestamp_filtered_df = pd.merge(Timestamp_filtered_df[columns_to_extract_t], matched_arrival_times_df[columns_to_extract_l], on=['仕入先名', '発送場所名', '整備室コード'], how='left')
        #実行結果の確認
        #st.header("所在管理データの抽出確認")
        #st.write(len(Timestamp_filtered_df))
        #st.dataframe(Timestamp_filtered_df.head(10000))

        #! 納入予定かんばん数（t）の計算関数
        def calculate_scheduled_nouyu_kanban(df, start_date, end_date):
            """
            指定期間内の納入データを抽出し、納入予定日時ごとに集計する関数。

            Args:
                df (pd.DataFrame): データフレーム。
                start_date (str): 抽出開始日（例：'2024/3/5'）。
                end_date (str): 抽出終了日。

            Returns:
                pd.DataFrame: 納入予定日時ごとの集計結果を格納したデータフレーム。
            """
            # 日付をdatetime形式に変換
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

            # ① "納入日"列が期間内に該当する行を抽出
            filtered_df = df[(pd.to_datetime(df['納入日']) >= start_date) & (pd.to_datetime(df['納入日']) < end_date)]

            #st.header("定刻便確認")
            #st.dataframe(filtered_df)

            # ② 抽出したデータに対して処理
            # ②-1 "納入便"列から数値を取得
            filtered_df['B'] = filtered_df['納入便'].astype(int)

            # ②-2 "B便_定刻"列の値を取得して新しい列"納入予定時間"に格納
            filtered_df['納入予定時間'] = filtered_df.apply(lambda row: row[f"{row['B']}便_定刻"] if f"{row['B']}便_定刻" in df.columns else None, axis=1)

            # ②-3 "納入予定時間"列が0時～8時の場合に"納入日_補正"列を1日後に設定
            filtered_df['納入予定時間'] = pd.to_datetime(filtered_df['納入予定時間'], format='%H:%M:%S', errors='coerce').dt.time
            #todo 夜勤便は+1が必要！！！！
            #todo 今の計算でいいか不明！！
            filtered_df['納入日_補正'] = filtered_df.apply(lambda row: (pd.to_datetime(row['納入日']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                                                        if row['納入予定時間'] and 0 <= row['納入予定時間'].hour < 6 else row['納入日'], axis=1)

            # ②-4 "納入日_補正"列と"納入予定時間"列を統合し"納入予定日時"列を作成
            filtered_df['納入予定日時'] = pd.to_datetime(filtered_df['納入日_補正']) + pd.to_timedelta(filtered_df['納入予定時間'].astype(str))

            #st.write(len(filtered_df))

            # ②-5 "納入予定日時"列で集計し、新しいデータフレームに格納
            nonyu_yotei_df = filtered_df.groupby('納入予定日時').agg(
                納入予定かんばん数=('納入予定日時', 'size'),
                納入予定便一覧=('納入便', lambda x: list(x)),
                納入予定かんばん一覧=('かんばんシリアル', lambda x: list(x)),
                納入予定便=('納入便', lambda x: list(set(x))[0] if len(set(x)) == 1 else list(set(x)))  # ユニークな納入便をスカラーに変換
            ).reset_index()
            
            nonyu_yotei_df['納入予定日時_raw'] = nonyu_yotei_df['納入予定日時']
            # "納入予定日時"列の分以降を0に設定
            nonyu_yotei_df['納入予定日時'] = nonyu_yotei_df['納入予定日時'].apply(lambda x: x.replace(minute=0, second=0) if pd.notna(x) else x)

            nonyu_yotei_df = nonyu_yotei_df.rename(columns={'納入予定かんばん数': '納入予定かんばん数（t）'})# コラム名変更
            nonyu_yotei_df = nonyu_yotei_df.rename(columns={'納入予定日時': '日時'})# コラム名変更

            #todo 検収日時2時56分だと、2時になるな。
            kensyu_df = filtered_df.groupby('検収日時').agg(
                検収かんばん数=('検収日時', 'size'),
                検収かんばん一覧=('かんばんシリアル', lambda x: list(x))
            ).reset_index()

            kensyu_df['検収日時_raw'] = kensyu_df['検収日時']
            kensyu_df['検収日時'] = kensyu_df['検収日時'].apply(lambda x: x.replace(minute=0, second=0) if pd.notna(x) else x)
            kensyu_df = kensyu_df.rename(columns={'検収日時': '日時'})# コラム名変更

            return nonyu_yotei_df, kensyu_df
        
        def calculate_disruption(df):
            """
            納入予定日時と検収日時の乱れを計算し、新しい列を作成する関数。

            Args:
                df (pd.DataFrame): 入力データフレーム。

            Returns:
                pd.DataFrame: 新しい列"仕入先到着or検収乱れ"を追加したデータフレーム。
            """
            # 新しい列を初期化
            df['仕入先到着or検収乱れ'] = None

            for idx, row in df.iterrows():
                if row['納入予定日時_raw'] != 0:
                    # base行を取得
                    base_time = pd.to_datetime(row['納入予定日時_raw'])

                    # 前2行と後ろ2行を取得
                    prev_rows = df.iloc[max(0, idx - 2):idx]
                    next_rows = df.iloc[idx + 1:idx + 3]

                    # early行とdelay行を探す
                    early_row = prev_rows[prev_rows['検収日時_raw'] != 0].tail(1)
                    delay_row = next_rows[next_rows['検収日時_raw'] != 0].head(1)

                    if not early_row.empty:
                        # early行がある場合
                        early_time = pd.to_datetime(early_row['検収日時_raw'].values[0])
                        time_diff = (base_time - early_time).total_seconds() / 3600  # 時間単位の差分を計算
                        df.at[idx, '仕入先到着or検収乱れ'] = time_diff
                    elif not delay_row.empty:
                        # earlyがなく、delay行がある場合
                        delay_time = pd.to_datetime(delay_row['検収日時_raw'].values[0])
                        time_diff = (delay_time - base_time).total_seconds() / 3600  # 時間単位の差分を計算
                        df.at[idx, '仕入先到着or検収乱れ'] = time_diff

            return df
        
        #! 納入予定かんばん数（t）の計算
        nonyu_yotei_df, kensyu_df = calculate_scheduled_nouyu_kanban(Timestamp_filtered_df, start_date, end_date)
        #! 日時でデータフレームを結合
        lagged_features = pd.merge(lagged_features, nonyu_yotei_df, on='日時', how='left')
        lagged_features = pd.merge(lagged_features, kensyu_df, on='日時', how='left')
        #! すべてのNone値を0に置き換え
        # lagged_featuresに統合する際、nonyu_yotei_dfに存在しない日時はNoneになるため
        lagged_features = lagged_features.fillna(0)

        # 実行結果の確認
        display_message(f"**納入予定かんばん数を計算しました。**")
        st.dataframe(nonyu_yotei_df)
        display_message(f"**参考）検収かんばん数を計算しました。**")
        st.dataframe(kensyu_df)
        display_message(f"**納入予定かんばん数、検収かんばん数を追加しました。**")
        st.dataframe(lagged_features)

        #! 仕入先便の到着乱れ
        display_message(f"**参考）検収タイムスタンプの乱れを確認します。**")
        columns_to_display = ['日時','納入予定かんばん一覧','検収かんばん一覧']
        st.dataframe(lagged_features[columns_to_display])

        #! Activedataの統合
        file_path = 'temp/activedata.csv'#ステップ１,2で併用しているため、変数ではなく一時フォルダーに格納して使用
        Activedata = pd.read_csv(file_path, encoding='shift_jis')
        # 日付列をdatetime型に変換
        Activedata['日付'] = pd.to_datetime(Activedata['日付'], errors='coerce')
        #! 品番、整備室情報読み込み
        #seibishitsu = product.split('_')[1]#整備室のみ
        product = part_number#product.split('_')[0]#品番のみ
        #! 同品番、同整備室のデータを抽出
        Activedata = Activedata[(Activedata['品番'] == product) & (Activedata['整備室'] == seibishitsu)]
        #! 1時間ごとに変換
        Activedata = Activedata.set_index('日付').resample('H').ffill().reset_index()
        filtered_Activedata = Activedata[Activedata['日付'].isin(lagged_features['日時'])].copy()
        filtered_Activedata = filtered_Activedata.reset_index(drop=True)
        filtered_Activedata = filtered_Activedata.rename(columns={'日付': '日時'})
        # 日時の形式が同じか確認し、必要ならば変換
        lagged_features['日付'] = pd.to_datetime(lagged_features['日時'])
        filtered_Activedata['日時'] = pd.to_datetime(filtered_Activedata['日時'])
        #! 昼勤夜勤の考慮
        def adjust_datetime(x):
            if 0 <= x.hour < 8:
                # 日付を前日に変更し、時間はそのまま
                return x + pd.Timedelta(days=1)
            else:
                # そのままの日付を返す
                return x
        #! 昼勤夜勤の考慮
        filtered_Activedata['日時'] = filtered_Activedata['日時'].apply(adjust_datetime)
        #! 日時でデータフレームを結合
        lagged_features = pd.merge(lagged_features, filtered_Activedata, on='日時')
        #! かんばん回転日数計算
        lagged_features["かんばん回転日数"] = (lagged_features["サイクル間隔"] * (lagged_features["サイクル情報"] + 1)) / lagged_features["サイクル回数"]
        #! 選択された列のユニークな値を取得
        kanban_kaiten_nissu = int(lagged_features["かんばん回転日数"].unique()[0])
        #! ユニークな値を表示
        display_message(f"**かんばん回転日数（設計値）は{kanban_kaiten_nissu}になりました。**")
        # 実行確認
        # st.dataframe(filtered_Activedata)

        #! IT生産管理版のデータを使った稼働フラグの計算　⇒　休日に生産計画データが入っていたり、データが存在しない日などがあるためボツに
        #! What：ある時間が稼働時間なのか非稼働時間なのか計算
        #! Result：kado_dfの作成
        # #todo 推測ではなく実績の稼働データが欲しい
        # # データフレームをコピー
        # kado_df = kumitate_df.copy()
        # # 稼働フラグを設定。'計画生産台数_加重平均済'>0なら稼働1、0なら非稼働0とする
        # kado_df['稼働フラグ'] = kado_df['計画生産台数_加重平均済'].apply(lambda x: 1 if x != 0 else 0)
        # # 必要な列を抽出
        # kado_df = kado_df[kado_df['整備室コード'] == seibishitsu]
        # kado_df = kado_df[['日時', '稼働フラグ']]
        # # 処理追加。入庫かんばん数（t）>0の時間も稼働とする
        # kado_df2 = lagged_features.copy()
        # kado_df2['稼働フラグ_入庫かんばん数基準'] = kado_df2['入庫かんばん数（t）'].apply(lambda x: 1 if x != 0 else 0)
        # kado_df2 = kado_df2[['日時', '稼働フラグ_入庫かんばん数基準']]
        # kado_df = pd.merge(kado_df, kado_df2, on='日時', how='right')
        # # 条件を満たす場合に稼働フラグを1に設定
        # kado_df.loc[kado_df['稼働フラグ_入庫かんばん数基準'] > 0, '稼働フラグ'] = 1
        # st.dataframe(kado_df)

        #! lagged_featuresに変数追加
        #! Result：「稼働フラグ」列の追加
        lagged_features = pd.merge(lagged_features, kado_df, on='日時', how='left')

        #! 仕入先便到着の乱れ計算
        lagged_features = calculate_disruption(lagged_features)
        # 0以上2以下の値を0に置き換える
        lagged_features = lagged_features.fillna(0)
        lagged_features['仕入先到着or検収乱れ_補正'] = lagged_features['仕入先到着or検収乱れ'].apply(lambda x: 0 if 0 <= x <= 2 else x)
        columns_to_display = ['日時','稼働フラグ','納入予定日時_raw','検収日時_raw','仕入先到着or検収乱れ','仕入先到着or検収乱れ_補正']
        display_message(f"**仕入先便到着の乱れを確認します。**")
        st.dataframe(lagged_features[columns_to_display])

        #!日量箱数計算
        lagged_features['日量箱数'] = lagged_features['日量数']/lagged_features['収容数']

        def shift_with_leadtime(df, target_column, output_column, leadtime):
            """
            指定列の値をリードタイムを考慮して新しい列に格納する関数。

            Args:
                df (pd.DataFrame): 入力データフレーム。
                target_column (str): 処理対象の列名。
                output_column (str): 新しく作成する列名。
                leadtime (int): リードタイム（稼働フラグが1の行を進める数）。

            Returns:
                pd.DataFrame: 処理後のデータフレーム。
            """
            # 新しい列を初期化
            df[output_column] = 0

            # target_columnが0以外の行を処理
            for idx, row in df.iterrows():
                if row[target_column] != 0:
                    base_time = row['日時']
                    base_value = row[target_column]

                    # 稼働フラグが1の行のみをカウントして進む
                    subset = df[(df['日時'] > base_time)]
                    active_rows = subset[subset['稼働フラグ'] == 1]

                    if len(active_rows) >= leadtime:
                        target_row = active_rows.iloc[leadtime - 1]
                        df.at[target_row.name, output_column] = base_value

            return df
        
        #! 納入かんばん数_時間遅れ（t）（後の入庫予定かんばん数）を計算する関数
        #! What：入庫かんばん数と納入かんばん数の間で相関が生まれるように時間遅れを考慮した納入かんばん数を計算する関数
        #! Result：'納入かんばん数_時間遅れ（t）'列を追加
        def calculate_delivery_kanban_with_time_delay(row, df, delivery_column, target_column, lead_time=5):

            """
            納入かんばん数の時間遅れを計算する汎用関数。

            この関数は、指定されたリードタイムと稼働フラグを基に納入かんばん数を入庫予定かんばん数に変換し、
            データフレームの該当する行に値を加算する。

            Args:
                row (pd.Series): データフレームの行データ。
                df (pd.DataFrame): 全体のデータフレーム。
                delivery_column (str): 納入かんばん数を持つ列名。
                target_column (str): 入庫予定かんばん数を更新する列名。
                lead_time (int): 基本リードタイム（時間単位）。デフォルトは5時間。

            Returns:
                None: 入庫予定かんばん数が更新されるが、明示的な戻り値はありません。
            """

            current_time = row['日時']
            kanban_count = row[delivery_column]

            # 納入かんばん数が0の場合は計算せず終了
            if kanban_count == 0:
                return None
            
            # リードタイムを四捨五入し、整数に変換
            lead_time = int(round(lead_time))

            # 5時間分の稼働フラグを取得
            end_time = current_time + pd.Timedelta(hours=lead_time)
            subset = df[(df['日時'] > current_time) & (df['日時'] <= end_time)]
            
            # 稼働フラグが0の回数をカウント
            zero_flag_count = subset[subset['稼働フラグ'] == 0].shape[0]
            
            # 実際のリードタイムを計算
            adjusted_lead_time = lead_time + zero_flag_count
            delivery_time = current_time + pd.Timedelta(hours=adjusted_lead_time)
            
            # 納入時刻がデータフレームの範囲外なら納入かんばん数は更新しない
            if delivery_time in df['日時'].values:
                delivery_index = df[df['日時'] == delivery_time].index[0]
                df.at[delivery_index, target_column] += kanban_count

            return None
        
        #todo 臨時計算、発注入庫Ltを非稼働時間削除で計算するまでの間
        def calculate_best_kanban_with_delay(df):
            """
            入庫かんばん数が0でないとき、前後2時間の最大値を新しい列に格納します。

            Args:
                df (pd.DataFrame): 処理対象のデータフレーム。

            Returns:
                pd.DataFrame: 結果を格納したデータフレーム。
            """

            df['日時'] = pd.to_datetime(df['日時'])
            df.set_index('日時', inplace=True)

            # 新しい列を初期化
            df['発注かんばん数_時間遅れ（t）'] = 0
            #df['納入かんばん数_時間遅れ'] = 0

            # 各行をループして処理
            for idx, row in df.iterrows():
                if row['入庫かんばん数（t）'] != 0:
                    # 前後2時間のデータを抽出
                    start_time = idx - pd.Timedelta(hours=2)
                    end_time = idx + pd.Timedelta(hours=2)
                    window_df = df[(df.index >= start_time) & (df.index <= end_time)]
                    
                    # 「発注かんばん数best」という文字列を含む列の最大値を計算
                    order_cols = [col for col in df.columns if "発注かんばん数best" in col]
                    if order_cols:
                        max_order = window_df[order_cols].max().max()
                        df.at[idx, '発注かんばん数_時間遅れ（t）'] = max_order

                    # 「納入かんばん数best」という文字列を含む列の最大値を計算
                    #delivery_cols = [col for col in df.columns if "納入かんばん数best" in col]
                    #if delivery_cols:
                        #max_delivery = window_df[delivery_cols].max().max()
                        #df.at[idx, '納入かんばん数_時間遅れ'] = max_delivery

            return df.reset_index()

        #! 入庫予定かんばん数（t）の補正
        def adjust_time_based_on_scheduled_incoming_kanban(df):

            """
            入庫予定かんばん数（t）を基に特定の条件で補正を行う関数。

            Args:
                df (pd.DataFrame): 入力データフレーム。
                                必須列:
                                - "入庫予定かんばん数（t）"
                                - "入庫かんばん数（t）"
                                - "稼働フラグ"
                                - "時間"

            Returns:
                pd.DataFrame: 補正後のデータフレーム（新しい列 "入庫予定かんばん数（t）_補正済" を追加）。
            """

            # 補正結果を記録する新しい列を初期化
            df['入庫予定かんばん数（t）_補正済'] = 0

            # データフレームの各行を順番に処理
            for idx, row in df.iterrows():
                # 「入庫予定かんばん数（t）」が0でない行を対象とする
                if row['入庫予定かんばん数（t）'] != 0:
                    # 現在の行より前で「稼働フラグ」が1の行を取得
                    # todo　【重要！】なぜ3に設定しているか？
                    # todo 入庫かんばん数が入庫予定かんばん数より早いことがある
                    # todo 17時18個入庫、18時39個入庫、19時1個入庫みたいなケースがあり、tail(1)だと、19時入庫予定に補正される
                    # todo 例：品番351710LC010_1Z、24年5月9日17時
                    before_active = df[(df.index < idx) & (df['稼働フラグ'] == 1)].tail(4)

                    # 現在の行より後で「稼働フラグ」が1の行を取得
                    after_active = df[(df.index > idx) & (df['稼働フラグ'] == 1)].head(1)

                    # 現在の行を含めて前後の稼働行を結合
                    current_row = df.loc[[idx]]
                    active_rows = pd.concat([before_active, current_row, after_active])

                    # 「入庫かんばん数（t）」列の値を確認
                    if active_rows['入庫かんばん数（t）'].sum() == 0:
                        # すべての値が0の場合、現在の行の「入庫予定かんばん数（t）_補正済」に値を設定
                        df.at[idx, '入庫予定かんばん数（t）_補正済'] = row['入庫予定かんばん数（t）']
                    else:
                        # 0以外の値が含まれる場合、最初の非0行を見つけ、その行に値を設定
                        first_non_zero = active_rows[active_rows['入庫かんばん数（t）'] != 0].head(1)
                        if not first_non_zero.empty:
                            first_non_zero_idx = first_non_zero.index[0]
                            df.at[first_non_zero_idx, '入庫予定かんばん数（t）_補正済'] = row['入庫予定かんばん数（t）']

            return df
        
        #! 滞留かんばんの時間変化を考慮
        def update_tairyukanban_by_considering_time_changes(df):

            """
            滞留かんばん数を更新し、変化を追跡するために滞留かんばん数_beforeと滞留かんばん数_afterを出力します。
            滞留かんばん数が0未満の場合は、0にリセットします。
            """

            df["西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮"] = df["西尾東物流センターor部品置き場での滞留かんばん数（t）"].copy()
            
            for i in range(1, len(df)):
                df.loc[i, "西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮"] = max(
                    0,
                    df.loc[i - 1, "西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮"]
                    + df.loc[i, "西尾東物流センターor部品置き場での滞留かんばん数（t）"]
                    - df.loc[i, "工場到着後の入庫作業などではない予定外の入庫かんばん数（t）"]
                )
            return df

        #! 出庫数を考慮して生産台数_出庫数考慮を計算する
        def calculate_production_considering_shipment(df):
            """
            出庫数を考慮して生産台数_出庫数考慮を計算する関数。
            出庫数が0の場合、生産台数を蓄積するが、出庫数が1以上の場合はリセットする。

            引数:
            df : DataFrame
                データフレーム（日時列、出庫数、生産台数を含む）

            戻り値:
            DataFrame
                「生産台数_出庫数考慮」を追加したデータフレーム
            """
            # データフレームのコピーを作成
            df = df.copy()
            
            # 新しい列を初期化
            df["計画生産台数_加重平均済_出庫数考慮"] = 0
            df["計画達成率_加重平均済_出庫数考慮"] = 0.0
            
            # 累積変数を初期化
            cumulative_production = 0
            cumulative_utilization = 0

            #カウント用
            count = 0

            for idx in df.index:
                cumulative_production += df.loc[idx, "計画生産台数_加重平均済"]
                cumulative_utilization += df.loc[idx, "計画達成率_加重平均済"]
                count += 1

                df.loc[idx, "計画生産台数_加重平均済_出庫数考慮"] = cumulative_production
                df.loc[idx, "計画達成率_加重平均済_出庫数考慮"] = cumulative_utilization / count if count > 0 else 0

                if df.loc[idx, "出庫かんばん数（t）"] > 0 and idx + 1 in df.index:
                    # 出庫数が1以上の場合、リセット
                    cumulative_production = 0
                    cumulative_utilization = 0
                    count = 0

            return df
        
        #! 出庫かんばん数列が0でない行の間隔の中央値を計算する
        def calculate_median_interval(df):
            """
            出庫かんばん数列が0でない行の間隔の中央値を計算する関数。
            df: データフレーム（日時列と出庫かんばん数列を含む）
            """
            # 出庫かんばん数が0でない行番号を抽出
            non_zero_indices = df[df["出庫かんばん数（t）"] != 0].index
            
            # 行番号の差を計算
            row_intervals = non_zero_indices.to_series().diff().dropna()
            
            # 行番号の差の中央値を計算
            median_row_interval = row_intervals.median()
            
            return median_row_interval
        
        median_interval_syuko = calculate_median_interval(lagged_features)

        #st.header("出庫の間隔（行数）")
        #st.write(median_interval_syuko)

        #! lagged_featuresに変数追加
        #! Result：入庫かんばん数に合わせた「納入かんばん数_時間遅れ（t）」列の追加
        # 納入かんばん数_時間遅れ（t）列を初期化
        #lagged_features['納入かんばん数_時間遅れ（t）'] = 0
        #lagged_features['納入予定かんばん数_時間遅れ（t）'] = 0
        #st.dataframe(lagged_features)
        # 納入かんばん数_時間遅れ（t）列を更新
        #lagged_features.apply(lambda row: calculate_delivery_kanban_with_time_delay(row, lagged_features, delivery_column='納入かんばん数（t）', target_column='納入かんばん数_時間遅れ（t）', lead_time=5), axis=1)
        # 納入予定かんばん数_時間遅れ（t）列を更新
        #lagged_features.apply(lambda row: calculate_delivery_kanban_with_time_delay(row, lagged_features, delivery_column='納入予定かんばん数（t）', target_column='納入予定かんばん数_時間遅れ（t）', lead_time=5), axis=1)
        lagged_features = shift_with_leadtime(lagged_features, target_column='納入かんばん数（t）', output_column='納入かんばん数_時間遅れ（t）', leadtime=5)
        lagged_features = shift_with_leadtime(lagged_features, target_column='納入予定かんばん数（t）', output_column='納入予定かんばん数_時間遅れ（t）', leadtime=5)
        lagged_features = shift_with_leadtime(lagged_features, target_column='仕入先到着or検収乱れ_補正', output_column='仕入先到着or検収乱れ_補正_時間遅れ', leadtime=5)
        display_message(f"**納入かんばん数（t）、納入予定かんばん数（t）、仕入先到着or検収乱れ_補正の時間遅れを計算しました。**")
        st.dataframe(lagged_features)

        #! lagged_featuresに変数追加
        #! Result：「入庫予定かんばん数（t）」列、「入庫予定かんばん数（t）_補正」列の追加
        # # 入庫予定かんばん数列に納入かんばん数_時間遅れ（t）の値をコピー
        lagged_features['入庫予定かんばん数（t）'] = lagged_features['納入予定かんばん数_時間遅れ（t）']

        #! 入庫予定かんばん数（t）_補正済の計算
        lagged_features = adjust_time_based_on_scheduled_incoming_kanban(lagged_features)

        #!
        lagged_features['西尾東物流センターor部品置き場での滞留かんばん数（t）'] = lagged_features['入庫予定かんばん数（t）_補正済'] - lagged_features['入庫かんばん数（t）']
        # マイナス値を0に変更
        lagged_features.loc[lagged_features['西尾東物流センターor部品置き場での滞留かんばん数（t）'] < 0, '西尾東物流センターor部品置き場での滞留かんばん数（t）'] = 0

        #todo 発注かんばん数については稼働時間抜きの正確なリードタイムがわからないため今は前のやつを流用
        #lagged_features['発注かんばん数_時間遅れ（t）']=0
        #lagged_features.apply(lambda row: calculate_delivery_kanban_with_time_delay(row, lagged_features, delivery_column='発注かんばん数（t）', target_column='発注かんばん数_時間遅れ（t）', lead_time=median_lt_order), axis=1)
        #todo 発注かんばん数
        #lagged_features = adjust_time_based_on_incoming_kanban(lagged_features, target_column="発注かんばん数_時間遅れ（t）")

        #! lagged_featuresに変数追加
        #! Result：西尾東or部品置き場での滞留かんばん数（t）列を追加
        #! What；入庫予定かんばん数（t）- ★入庫かんばん数（t）の前後1時間の合計★※前後1時間の合計がポイント
        #lagged_features = calculate_tairyukanban(lagged_features, target_column="入庫かんばん数（t）")
        
        #todo 発注かんばん数_時間遅れ計算
        lagged_features = calculate_best_kanban_with_delay(lagged_features)

        #todo 納入フレ数を計算
        #lagged_features['納入フレ数（t）'] = lagged_features['入庫予定かんばん数（t）_補正'] - lagged_features['発注かんばん数_時間遅れ（t）']
        lagged_features['納入フレ数（t）'] = lagged_features['入庫予定かんばん数（t）_補正済'] - lagged_features['発注かんばん数_時間遅れ（t）']

        # 新しい列を作成し、予定外の入庫かんばん数を計算
        lagged_features['工場到着後の入庫作業などではない予定外の入庫かんばん数（t）'] = 0
        lagged_features.loc[lagged_features['入庫かんばん数（t）'] != 0, '工場到着後の入庫作業などではない予定外の入庫かんばん数（t）'] = (
            lagged_features['入庫かんばん数（t）'] - lagged_features['入庫予定かんばん数（t）_補正済']
        )
        # マイナス値を0に変更
        lagged_features.loc[lagged_features['工場到着後の入庫作業などではない予定外の入庫かんばん数（t）'] < 0, '工場到着後の入庫作業などではない予定外の入庫かんばん数（t）'] = 0
        
        #! 滞留かんばんの時間変化を考慮
        lagged_features = update_tairyukanban_by_considering_time_changes(lagged_features)
    
        #! lagged_featuresに変数追加
        #! "計画生産台数_加重平均済_出庫数考慮"列、"計画達成率_加重平均済_出庫数考慮"列を追加
        lagged_features = calculate_production_considering_shipment(lagged_features)

        # 実行結果の確認
        columns_to_display = ['日時','稼働フラグ','納入かんばん数（t）','納入予定かんばん数（t）','入庫かんばん数（t）','入庫予定かんばん数（t）','入庫予定かんばん数（t）_補正済',
                               '工場到着後の入庫作業などではない予定外の入庫かんばん数（t）','西尾東物流センターor部品置き場での滞留かんばん数（t）',
                               '西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮']
        display_message(f"**西尾東Bc～部品置き場の滞留かんばん数などを計算しました。**")
        st.dataframe(lagged_features[columns_to_display])

        #! 間口の充足率を計算
        lagged_features[f'間口A1の充足率'] = lagged_features['在庫数（箱）合計_A1']/2592
        lagged_features[f'間口A2の充足率'] = lagged_features['在庫数（箱）合計_A2']/1668
        lagged_features[f'間口B1の充足率'] = lagged_features['在庫数（箱）合計_B1']/827
        lagged_features[f'間口B2の充足率'] = lagged_features['在庫数（箱）合計_B2']/466
        lagged_features[f'間口B3の充足率'] = lagged_features['在庫数（箱）合計_B3']/330
        lagged_features[f'間口B4の充足率'] = lagged_features['在庫数（箱）合計_B4']/33
        lagged_features[f'全間口の平均充足率'] = (
            lagged_features[f'間口A1の充足率'] +
            lagged_features[f'間口A2の充足率'] + 
            lagged_features[f'間口B1の充足率'] + 
            lagged_features[f'間口B2の充足率'] + 
            lagged_features[f'間口B3の充足率'] + 
            lagged_features[f'間口B4の充足率'])/6
        #!いずれかが0.95を超えた場合に1に設定
        lagged_features['投入間口の渋滞判定フラグ'] = (
            (lagged_features['間口A1の充足率'] > 0.95) |
            (lagged_features['間口A2の充足率'] > 0.95) |
            (lagged_features['間口B1の充足率'] > 0.95) |
            (lagged_features['間口B2の充足率'] > 0.95) |
            (lagged_features['間口B3の充足率'] > 0.95) |
            (lagged_features['間口B4の充足率'] > 0.95)
        ).astype(int)
        display_message(f"**投入間口の渋滞判定を計算しました。**")
        st.dataframe(lagged_features)

        #todo---------------------------------------------------------------------------------------------------------------------------
    
        # 実行結果の確認
        # 各変数の相関を調べる
        #? 目的関数をどのように設計すべきか？
        
        # ＜結論＞
        # 設計案2の方が滞留かんばん数との相関が出るので、2を採用
        
        #* ＜目的関数の設計案１＞
        # 全データポイントの中央値に対するズレ
        #lagged_features['在庫数（箱）_中央値からのズレ'] = lagged_features['在庫数（箱）'] - lagged_features['在庫数（箱）'].median()
        # 実行結果の確認
        #st.write(lagged_features['在庫数（箱）'].median())

        #* ＜目的関数の設計案２＞
        #　各時点の在庫数（箱）から同じ時刻（例: 0時、1時）の中央値を引いた値

        #! 時間帯ごとの箱ひげ図
        def plot_box_by_hour(dataframe, value_col, time_col):
            """
            時間帯ごとの箱ひげ図を作成する関数。

            Parameters:
            - dataframe: pd.DataFrame - データフレーム
            - value_col: str - 箱ひげ図に使用する値の列名
            - time_col: str - 時間帯をグループ化する列名（日時列）
            """
            # '時間'列を作成
            dataframe['Hour'] = pd.to_datetime(dataframe[time_col]).dt.hour

            # 箱ひげ図を作成
            fig = px.box(
                dataframe,
                x='Hour',
                y=value_col,
                title=f"{value_col}の時間帯別箱ひげ図",
                labels={'Hour': '時間', value_col: value_col},
                points="all"  # データポイントも表示
            )

            # Streamlitで表示
            st.plotly_chart(fig, use_container_width=True)

        #! 'Hour'列を作成
        lagged_features['Hour'] = pd.to_datetime(lagged_features['日時']).dt.hour

        #! 各時間帯の中央値を引いた新しい列を作成
        lagged_features['在庫数（箱）_中央値からのズレ'] = (
            lagged_features['在庫数（箱）'] - 
            lagged_features.groupby('Hour')['在庫数（箱）'].transform('median')
        )
        st.header("test")
        st.dataframe(lagged_features.groupby('Hour')['在庫数（箱）'].transform('median'))
        lagged_features['いつもの在庫数（箱）'] = lagged_features.groupby('Hour')['在庫数（箱）'].transform('median')
        column = ['日時','いつもの在庫数（箱）']
        basezaiko_data = lagged_features[column]
        basezaiko_data.to_csv('temp/いつもの在庫数.csv', index=False, encoding='shift_jis')
        st.dataframe(basezaiko_data)

        # 実行結果の確認
        # 在庫数の箱ひげ図確認
        plot_box_by_hour(dataframe=lagged_features, value_col='在庫数（箱）', time_col='日時')
        # 各時間帯の中央値を計算
        hourly_median = lagged_features.groupby('Hour')['在庫数（箱）'].median()
        # 各時間の中央値を確認
        st.write(hourly_median)

        #todo 上に一つ移動する処理
        # 16時に在庫10個、入庫かんばん数が20個だから、17時の在庫が30個というデータになっている
        lagged_features['在庫数（箱）_中央値からのズレ'] = lagged_features['在庫数（箱）_中央値からのズレ'].shift(-1)

        # ここまでで作成した変数
        # '西尾東or部品置き場での滞留かんばん数（t）_時間変化考慮'
        # '計画生産台数_加重平均済'

        lagged_features['計画生産台数_加重平均済_中央値からのズレ'] = lagged_features['計画生産台数_加重平均済'] - lagged_features['計画生産台数_加重平均済'].median()

        lagged_features['計画生産台数_加重平均済_中央値からのズレ'] = (
            lagged_features['計画生産台数_加重平均済'] - 
            lagged_features.groupby('Hour')['計画生産台数_加重平均済'].transform('median')
        )

        lagged_features['出庫かんばん数（t）_中央値からのズレ'] = (
            lagged_features['出庫かんばん数（t）'] - 
            lagged_features.groupby('Hour')['出庫かんばん数（t）'].transform('median')
        )

        #lag_end = 24*5*2 #〇前から
        #lag_start = 24*5 #△行個
        # lagged_features['計画生産台数_加重平均済_長期過去要因'] = (
        #     lagged_features['計画生産台数_加重平均済']
        #     .shift(lag_end)
        #     .rolling(window=lag_start, min_periods=lag_start)  # 6行分（5行目から10行目まで）を対象に合計
        #     .sum()
        # )

        def calculate_flag_based_sum(df, target_col,output_column, flag_col, lag_past_count, lag_future_count):
            """
            稼働フラグ列を基準として過去にlag_past_count行、未来にlag_future_count行を探し、
            指定された列の合計を計算する関数。
            
            Args:
                df (pd.DataFrame): 処理対象のデータフレーム。
                target_col (str): 合計を計算する対象列名。
                flag_col (str): 稼働フラグ列の列名（1または0の値）。
                lag_past_count (int): 過去方向に探す稼働フラグ1の行数。
                lag_future_count (int): 現在の行方向に探す稼働フラグ1の行数。

            Returns:
                pd.DataFrame: 処理後のデータフレーム（新しい列を追加）。
            """
            result = []  # 計算結果を格納するリスト

            # データフレームの全行を対象に、lag_past_count行まで遡る処理を行う
            for i in range(lag_past_count, len(df)):
                # 1. 現在の行`i`のlag_past_count行前までを取得（過去方向）
                past_rows = df.iloc[:i][flag_col]  # 現在の行から過去の行の稼働フラグ列を取得

                # 2. 稼働フラグが1の行を過去方向にlag_past_count個見つける
                past_indices = past_rows[past_rows == 1].index[-lag_past_count:]  # 稼働フラグ1の行インデックス

                if len(past_indices) < lag_past_count:
                    # 稼働フラグ1が指定数見つからない場合は計算せず次へ
                    result.append(None)
                    continue

                # 3. 過去方向のlag_past_count行見つかったスタート地点を取得
                start_index = past_indices[0]

                # 4. 未来方向にlag_future_count行の稼働フラグ1を探す
                future_rows = df.iloc[start_index:i][flag_col]  # スタート地点から現在行方向へ稼働フラグを探索
                future_indices = future_rows[future_rows == 1].index[:lag_future_count]  # 未来方向に稼働フラグ1のインデックスを取得

                if len(future_indices) < lag_future_count:
                    # 稼働フラグ1が指定数見つからない場合は計算せず次へ
                    result.append(None)
                    continue

                # 5. 未来方向のlag_future_count行分の範囲で合計を計算
                sum_value = df.loc[future_indices, target_col].sum()  # 合計を計算
                result.append(sum_value)  # 計算結果をリストに追加

            # 6. 結果リストの先頭に`NaN`をlag_past_count個追加し、全体の行数を揃える
            result = [None] * lag_past_count + result  # 過去方向に遡る数分だけ`None`を追加

            df[output_column] = result
            return df
        
        #! 計画生産台数_長期過去要因
        lag_past_count_value = (int(WWWW)+kanban_kaiten_nissu)*24 #24*kanban_kaiten_nissu*3
        lag_future_count_value = (int(WWWW))*24#24*kanban_kaiten_nissu*2
        st.write(lag_past_count_value)
        lagged_features = calculate_flag_based_sum(lagged_features,
                                                   target_col='日量数',output_column='計画生産台数_加重平均済_長期過去要因',
                                                     flag_col='稼働フラグ',lag_past_count = lag_past_count_value, lag_future_count = lag_future_count_value)

        # lagged_features['計画生産台数_加重平均済_長期過去要因'] = (
        #     lagged_features['日量数']
        #     .shift(lag_past_count_value)
        #     .rolling(window=lag_future_count_value, min_periods=lag_future_count_value)  # 6行分（5行目から10行目まで）を対象に合計
        #     .sum()
        # )

        fig = px.line(lagged_features, x='日時', y='日量数', title='日付ごとの数量推移', markers=True)
        st.plotly_chart(fig)
        st.dataframe(lagged_features)
        
        # 日付ごとの平均値計算
        lagged_features['日付'] = lagged_features['日時'].dt.date  # 日付列を追加
        mean_values = lagged_features.groupby('日付')['計画生産台数_加重平均済_長期過去要因'].transform('mean')  # 同じ日付ごとの平均値
        # 値を日付ごとの平均値に置き換え
        lagged_features['計画生産台数_加重平均済_長期過去要因'] = mean_values
        
        
        #計画生産台数は在庫増、減両方に関係する要因

        from plotly.subplots import make_subplots

        #!　散布図と相関係数を表示する関数
        def scatter_and_correlation(dataframe, x_col, y_col):

            """
            散布図を作成し、相関係数を計算して表示する関数。

            Parameters:
            - dataframe: pd.DataFrame - 対象のデータフレーム
            - x_col: str - 散布図のx軸に使用する列名
            - y_col: str - 散布図のy軸に使用する列名
            """

            # 相関係数を計算
            correlation = dataframe[x_col].corr(dataframe[y_col])

            # 散布図を作成
            fig = px.scatter(
                dataframe,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}",
                labels={x_col: x_col, y_col: y_col}
                #trendline="ols"  # 回帰直線を追加
            )

            # 2. 時系列プロット
            time_col = '日時'
            fig_time_series = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                            subplot_titles=(f"{x_col} Over Time", f"{y_col} Over Time"))

            # 上段プロット (X)
            fig_time_series.add_trace(
                go.Scatter(x=dataframe[time_col], y=dataframe[x_col], mode='lines+markers', name=x_col),
                row=1, col=1
            )

            # 下段プロット (Y)
            fig_time_series.add_trace(
                go.Scatter(x=dataframe[time_col], y=dataframe[y_col], mode='lines+markers', name=y_col),
                row=2, col=1
            )


            # グラフサイズを調整
            fig.update_layout(
                width=1200,  # 幅
                height=600  # 高さ
            )

            # グラフサイズを調整
            fig_time_series.update_layout(
                width=1200,  # 幅
                height=600  # 高さ
            )

            # Streamlitアプリケーション内で表示
            st.plotly_chart(fig)
            st.write(f"相関係数: {correlation:.2f}")

            st.plotly_chart(fig_time_series, use_container_width=True)

        # 関数を呼び出して結果を表示
        # 西尾東or部品置き場での滞留かんばん数（t）_時間変化考慮'の場合
        # scatter_and_correlation(
        #     dataframe=lagged_features,
        #     x_col='在庫数（箱）_中央値からのズレ',
        #     y_col='西尾東or部品置き場での滞留かんばん数（t）_時間変化考慮')
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col="西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮")
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col='入庫予定かんばん数（t）')
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col='入庫予定かんばん数（t）_補正済')
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col="工場到着後の入庫作業などではない予定外の入庫かんばん数（t）")
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col='計画生産台数_加重平均済')
        
        lagged_features = lagged_features[lagged_features['日時'].dt.weekday < 5]
        
        scatter_and_correlation(
            dataframe=lagged_features,
            x_col='在庫数（箱）_中央値からのズレ',
            y_col='計画生産台数_加重平均済_長期過去要因')
        
        #! 箱ひげ図確認
        plot_box_by_hour(dataframe=lagged_features, value_col='出庫かんばん数（t）', time_col='日時')

        # やること
        # 発注して取り消したものがどれだけあるか？

        #todo---------------------------------------------------------------------------------------------------------------------------
        
        #!定期便
        lagged_features = pd.merge(lagged_features, teikibin_df[['日時', '荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)']], on='日時', how='left')
        #!特徴量エンジニアリング
        lagged_features = feature_engineering(lagged_features)

        #!解析窓
        timelag = 48#best_range_order
        end_hours_ago = 0

        #!ローリング特徴量
        #! 解析窓で計算
        lagged_features = calculate_window_width(lagged_features, timelag, best_range_order, best_range_reception)

        #! NaN値を処理する（例: 0で埋める）
        lagged_features = lagged_features.fillna(0)
        
        #    ##todo 全部終わった後に非稼動日時間のデータ追加。上まで遅れ計算で土日などを除外しているので。
        #    # 補完する時間範囲を決定
        #    full_range = pd.date_range(start=start_date, end=end_date, freq='H')
        #    # full_rangeをデータフレームに変換
        #    full_df = pd.DataFrame(full_range, columns=['日時'])
        #    # 元のデータフレームとマージして欠損値を補完
        #    lagged_features = pd.merge(full_df, lagged_features, on='日時', how='left')
        #    # 欠損値を0で補完
        #    lagged_features.fillna(0, inplace=True)
        #    #
        #    lagged_features = lagged_features.drop(columns=['在庫数（箱）'])
        #    lagged_features['品番']=part_number
        #    lagged_features = pd.merge(lagged_features, df2[['日時', '品番','在庫数（箱）']], on=['品番', '日時'], how='left')#自動ラック在庫結合

        
        #todo 長期休暇分削除
        def delete_holiday(lagged_features):

            #todo 稼働日フラグ欲しい
        
            #! 夏休み
            start = '2024-08-12'
            end = '2024-08-16'
            #! 日付範囲に基づいてフィルタリングして削除
            lagged_features= lagged_features[~((lagged_features['日時'] >= start) & (lagged_features['日時'] <= end))]

            #! GW
            start = '2024-05-06'
            end = '2024-05-10'
            #! 日付範囲に基づいてフィルタリングして削除
            lagged_features= lagged_features[~((lagged_features['日時'] >= start) & (lagged_features['日時'] <= end))]

            return lagged_features

        #todo 長期休暇分削除
        lagged_features = delete_holiday(lagged_features)

        #!遅れ分削除
        data = lagged_features.iloc[300:]

        data['定期便出発状況（t-4~t-6）']=data['荷役時間(t-4)']/50+data['荷役時間(t-5)']/50+data['荷役時間(t-6)']/50

        # 実行結果確認
        display_message(f"**前処理が完了しました。**")
        st.dataframe(lagged_features)

        #todo ここまでで土日は消えている

        temp_data = data

        # モデルを格納するためのリストを作成
        rf_models = []

        # 3つのRFモデルを作成する
        for i in range(3):

            if i == 0:
                # 全データを活用
                data = temp_data
            elif i == 1:
                #one_and_half_months_ago = pd.to_datetime(end_date) - pd.Timedelta(days=45)
                # 1か月半前のデータを抽出
                #data = temp_data[temp_data['日時'] >= one_and_half_months_ago]
                data = temp_data
            elif i == 2:
                #three_and_half_months_ago_manual = pd.to_datetime(end_date) - pd.Timedelta(days=105)
                # 3か月半前のデータを抽出
                #data = temp_data[temp_data['日時'] >= three_and_half_months_ago_manual]
                data = temp_data

            #! 番号を割り当てる
            delay_No1 = best_range_order
            timelag_No1 = timelag
            data[f'No1_発注かんばん数（t-{delay_No1}~t-{delay_No1+timelag_No1}）'] = data[f'発注かんばん数（t-{delay_No1}~t-{delay_No1+timelag_No1}）']

            delay_No2 = end_hours_ago
            timelag_No2 = timelag
            data[f'No2_計画組立生産台数_加重平均（t-{delay_No2}~t-{delay_No2+timelag_No2}）'] = data[f'計画組立生産台数_加重平均（t-{delay_No2}~t-{delay_No2+timelag_No2}）']
            
            delay_No3 = end_hours_ago
            timelag_No3 = timelag
            data[f'No3_計画達成率_加重平均（t-{delay_No3}~t-{delay_No3+timelag_No3}）'] = data[f'計画達成率_加重平均（t-{delay_No3}~t-{delay_No3+timelag_No3}）']
            
            #delay_No4 = best_range_reception
            #timelag_No4 = timelag
            #data[f'No4_納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）'] = data[f'納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）']
            
            delay_No5 = best_range_reception
            timelag_No5 = 2
            data[f'No5_仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）'] = data[f'仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）']
            
            #data['No6_定期便出発状況（t-4~t-6）'] = data['定期便出発状況（t-4~t-6）']
            
            delay_No7 = end_hours_ago
            timelag_No7 = timelag
            data[f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）'] = data[f'間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）']
            # 充足率が1より小さい場合、0に更新
            data.loc[data[f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）'] < 1, f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）'] = 0
            
            delay_No8 = end_hours_ago
            timelag_No8 = timelag
            data[f'No8_部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）'] = data[f'部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）']
            
            #delay_No9 = end_hours_ago
            #timelag_No9 = timelag
            #data[f'No9_定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）'] = data[f'定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）']


            #! 特徴量選択-----------------------------------------------------------------------------------------------------------------------------------------------
            
            #! ----------------------------
            #! 今のもの
            #! ----------------------------

            # 瞬間要因はラグ0
            delay_No12 = 0
            timelag_No12 = 0
            data[f'No12_発注かんばん数（t-{delay_No12}~t-{delay_No12+timelag_No12}）'] = data['入庫予定かんばん数（t）_補正済']

            delay_No13 = 0
            timelag_No13 = 0
            data[f'No13_西尾東物流センターor部品置き場での滞留かんばん数（t-{delay_No13}~t-{delay_No13+timelag_No13}）'] = data["西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮"]

            delay_No14 = 0
            timelag_No14 = 0
            data[f'No14_入庫遅れなどによる予定外の入庫かんばん数（t-{delay_No14}~t-{delay_No14+timelag_No14}）'] = data['工場到着後の入庫作業などではない予定外の入庫かんばん数（t）']

            delay_No15 = 0
            timelag_No15 = 0
            data[f'No15_間口の平均充足率（t-{delay_No15}~t-{delay_No15+timelag_No15}）'] = data['投入間口の渋滞判定フラグ']

            #仕入先便otu
            delay_No16 = 0
            timelag_No16 = 0
            data[f'No16_仕入先到着or検収乱れ（t-{delay_No16}~t-{delay_No16+timelag_No16}）'] = data['仕入先到着or検収乱れ_補正_時間遅れ']

            #計画生産台数
            #lag_past_count_value = 24*5*2
            #lag_future_count_value =24*5
            delay_No17 = lag_past_count_value - lag_future_count_value
            timelag_No17 = lag_past_count_value - delay_No17
            data[f'No17_計画生産台数（t-{delay_No17}~t-{delay_No17+timelag_No17}）'] = data['計画生産台数_加重平均済_長期過去要因']

            #納入フレ

            #todo-------------------------------------------------------------------------------------------------------------

            #! old

            # 発注フラグ_時間遅れ（t）を設定
            data[f"No10_発注フラグ_時間遅れ（t-{delay_No1}~t-{delay_No1+timelag_No1}）"] = data["発注かんばん数_時間遅れ（t）"].apply(lambda x: 1 if x > 0 else 0)

            data['発注かんばん数_時間遅れ（t）'] = data['発注かんばん数_時間遅れ（t）'] #- data['便Ave']#todo 便Aveどうする？
            # 発注フラグが0の行に対して、発注かんばん数_時間遅れ（t）を0に更新
            data.loc[data[f"No10_発注フラグ_時間遅れ（t-{delay_No1}~t-{delay_No1+timelag_No1}）"] == 0, '発注かんばん数_時間遅れ（t）'] = 0
            data[f'No1_発注かんばん数（t-{delay_No1}~t-{delay_No1+timelag_No1}）'] = data['入庫予定かんばん数（t）_補正済']

            data[f'No2_計画組立生産台数_加重平均（t-{delay_No2}~t-{delay_No2+timelag_No2}）'] = data["計画生産台数_加重平均済_出庫数考慮"]

            data["計画達成率_加重平均済"] = data["計画達成率_加重平均済"].replace([np.inf, -np.inf], 1.5, inplace=True)
            data[f'No3_計画達成率_加重平均（t-{delay_No3}~t-{delay_No3+timelag_No3}）'] = data["計画達成率_加重平均済"]

            data[f'No8_部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）'] =data["西尾東物流センターor部品置き場での滞留かんばん数（t）_時間変化考慮"]

            data[f"No11_予定外の入庫かんばん数"] = data['工場到着後の入庫作業などではない予定外の入庫かんばん数（t）']

            display_message(f"**説明変数候補の計算が完了しました。**")
            st.dataframe(data)
            
            data.fillna(0,inplace=True)
            

            #todo-------------------------------------------------------------------------------------------------------------

            #! 説明変数の設定
            # X = data[[f'No1_発注かんばん数（t-{delay_No1}~t-{delay_No1+timelag_No1}）',
            #         f'No2_計画組立生産台数_加重平均（t-{delay_No2}~t-{delay_No2+timelag_No2}）',
            #         f'No3_計画達成率_加重平均（t-{delay_No3}~t-{delay_No3+timelag_No3}）',
            #         #f'No4_納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）',
            #         f'No5_仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）',
            #         #'No6_定期便出発状況（t-4~t-6）',#'荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)',
            #         f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）',#f'間口_A1の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_A2の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B1の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B2の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_B3の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B4の充足率（t-{end_hours_ago}~t-{best_range_order}）',
            #         f'No8_部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）',#f'部品置き場からの入庫（t-{end_hours_ago}~t-{best_range_order}）',f'部品置き場で滞留（t-{end_hours_ago}~t-{best_range_order}）',
            #         #f'No9_定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）']
            #         f"No10_発注フラグ_時間遅れ（t-{delay_No1}~t-{delay_No1+timelag_No1}）",
            #         f"No11_予定外の入庫かんばん数"
            #         ]]
            
            #! 説明変数の設定
            X = data[[f'No12_発注かんばん数（t-{delay_No12}~t-{delay_No12+timelag_No12}）',
                      f'No13_西尾東物流センターor部品置き場での滞留かんばん数（t-{delay_No13}~t-{delay_No13+timelag_No13}）',
                      f'No14_入庫遅れなどによる予定外の入庫かんばん数（t-{delay_No14}~t-{delay_No14+timelag_No14}）',
                      f'No15_間口の平均充足率（t-{delay_No15}~t-{delay_No15+timelag_No15}）',
                      f'No16_仕入先到着or検収乱れ（t-{delay_No16}~t-{delay_No16+timelag_No16}）',
                      f'No17_計画生産台数（t-{delay_No17}~t-{delay_No17+timelag_No17}）'
                    ]]
                      
            
            #確認：実行結果
            display_message(f"**目的変数と説明変数の設定が完了しました。**")
            st.dataframe(X.head(300))

            #! 目的変数の定義
            #! 中央値ズレのアプローチを採用
            y = data['在庫数（箱）_中央値からのズレ']

            #! データを学習データとテストデータに分割
            #todo 学習データの割合
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

            #! Lasso回帰モデルの作成
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

            #! ランダムフォレストモデルの訓練
            if i == 2:
                rf_model = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=42)
                rf_model.fit(X_train, y_train)
            elif i == 1:
                rf_model2 = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=42)
                rf_model2.fit(X_train, y_train)
            elif i == 0:
                rf_model3 = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=42)
                rf_model3.fit(X_train, y_train)

            # テストデータで予測し、MSEを計算
            #y_pred = rf_model.predict(X_test)
            #mse = mean_squared_error(y_test, y_pred)
            #print(f'ランダムフォレスト - テストデータのMSE: {mse}')
            # 最大誤差を計算
            #max_err = max_error(y_test, y_pred)
            #print(f'ランダムフォレスト - テストデータの最大誤差: {max_err}')
            # マイナス方向の最大誤差を計算
            #min_err = np.min(y_test - y_pred)
            #print(f'ランダムフォレスト - テストデータのマイナス方向の最大誤差: {min_err}')
            # 平均誤差を計算
            #mae2 = mean_absolute_error(y_test, y_pred)
            #st.header(mae2)
            #print(f'ランダムフォレスト - テストデータの平均誤差: {mae2}')
            #--------------------------------------------------------------------------------------------------------
            
            unique_hinban_list = lagged_features['仕入先名'].unique()
            supply = str(unique_hinban_list[0])
            zaikozaiko = lagged_features['在庫数（箱）'].mean()
            
            #appendメソッドはpandasの最新バージョンでは廃止
            # 結果をデータフレームに追加
            #results_df = results_df.append({'品番': part_number,'仕入先名':supply,'平均在庫':zaikozaiko,'Ridge回帰の平均誤差': mae, 'Ridge回帰のマイナス方向の最大誤差': min_error_test, 'Ridge回帰のプラス方向の最大誤差': max_error_test,
                                            #'ランダムフォレストの平均誤差': mae2, 'ランダムフォレストのマイナス方向の最大誤差': min_err, 'ランダムフォレストのプラス方向の最大誤差': max_err}, ignore_index=True)

            #! 実行結果収集
            new_row = pd.DataFrame([{'品番': part_number,'仕入先名':supply,'平均在庫':zaikozaiko,'Ridge回帰の平均誤差': mae, 'Ridge回帰のマイナス方向の最大誤差': min_error_test, 'Ridge回帰のプラス方向の最大誤差': max_error_test}])
            results_df = pd.concat([results_df, new_row], ignore_index=True)

            #! 終了通知
            print("終了")
            
            #! CSVファイルに保存
            with open("temp/一時保存データ.csv", mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
                data.to_csv(f)

            #! CSVファイルに保存
            with open("temp/全品番テスト結果.csv", mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
                results_df.to_csv(f)
        
    return data, rf_model, rf_model2, rf_model3, X

        ###全部終わった後に非稼動日時間のデータ追加。上まで遅れ計算で土日などを除外しているので。
        ## 補完する時間範囲を決定
        #full_range = pd.date_range(start=start_date, end=end_date, freq='H')
        ## full_rangeをデータフレームに変換
        #full_df = pd.DataFrame(full_range, columns=['日時'])
        ## 元のデータフレームとマージして欠損値を補完
        #lagged_features = pd.merge(full_df, lagged_features, on='日時', how='left')
        ## 欠損値を0で補完
        #lagged_features.fillna(0, inplace=True)
        ##
        #lagged_features = lagged_features.drop(columns=['在庫数（箱）'])
        #lagged_features['品番']=part_number
        #lagged_features = pd.merge(lagged_features, df2[['日時', '品番','在庫数（箱）']], on=['品番', '日時'], how='left')#自動ラック在庫結合

#---------------------------------------------------------------------------------------------------------------------------------

#! ステップ２の処理
def step2(data, rf_model, X, start_index, end_index, step3_flag, highlight_time=None):

    #Todo 品番名を取り出すために実行、きれいじゃないから要修正
    with open('temp/model_and_data.pkl', 'rb') as file:
        rf_model, rf_model2, rf_model3, X, data, product = pickle.load(file)

    #! Activeデータをダウンロード
    #start_date2 = '202405'
    #end_date2 = '202408'
    #ver = '00'
    #Activedata = read_activedata_from_IBMDB2(start_date2, end_date2, ver)#process_Activedata()
    #st.header(product)
    file_path = 'temp/activedata.csv'
    Activedata = pd.read_csv(file_path, encoding='shift_jis')
    # 日付列をdatetime型に変換
    Activedata['日付'] = pd.to_datetime(Activedata['日付'], errors='coerce')
    #! 品番、整備室情報読み込み
    seibishitsu = product.split('_')[1]#整備室のみ
    product = product.split('_')[0]#品番のみ
    #! 同品番、同整備室のデータを抽出
    Activedata = Activedata[(Activedata['品番'] == product) & (Activedata['整備室'] == seibishitsu)]

    #実行結果の確認
    #st.header(start_index)
    #st.header(end_index)

    # 在庫データに合わせて時間粒度を1時間ごとにリサンプリング
    # 内示データを日付ごとに集約して重複を排除
    #Activedata = Activedata.groupby('日付').mean(numeric_only=True).reset_index()
    #st.dataframe(Activedata)
    Activedata = Activedata.set_index('日付').resample('H').ffill().reset_index()

    #st.dataframe(Activedata.head(300))

    #折り返し線を追加
    st.markdown("---")

    #インデックスが300スタートなのでインデックスをリセット
    #遅れ時間の計算のため
    data = data.reset_index(drop=True)
    #st.dataframe(data.head(300))

    # SHAP計算
    #before
    #explainer = shap.TreeExplainer(rf_model, feature_dependence='tree_path_dependent', model_output='margin')

    #after
    explainer = shap.TreeExplainer(rf_model, model_output='raw')
    shap_values = explainer.shap_values(X)

    explainer = shap.TreeExplainer(rf_model2, model_output='raw')
    shap_values2 = explainer.shap_values(X)

    explainer = shap.TreeExplainer(rf_model3, model_output='raw')
    shap_values3 = explainer.shap_values(X)

    #アンサンブル試験
    shap_values = shap_values# + shap_values2 + shap_values3
    #shap_values = shap_values
    #shap_values = shap_values2
    #shap_values = shap_values3

    #first_datetime_df = data['日時'].iloc[0]
    #print(f"dataの日時列の最初の値: {first_datetime_df}")

    # リストから整数に変換
    start_index_int = start_index[0]#-300
    end_index_int = end_index[0]+1#-300

    #在庫データフレーム
    df = data.iloc[start_index_int:end_index_int]
    #st.dataframe(df)

    #st.dataframe(df.head(300))

    #first_datetime_df = df.iloc[0]
    #print(f"dfの日時列の最初の値: {first_datetime_df}")


    #! -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    #! ＜目的関数に在庫増減数を設定する場合＞
    #! Results：start_index_int、end_index_int、y_pred_subset、y_base_subset
    
    # #! 16時時点の在庫数は、15時時点の在庫数と15時の在庫増減数で決定されるため、時刻を1つ前に参照する必要がある
    # start_index_int = start_index_int - 1
    # end_index_int = end_index_int -1 
    # X_subset = X.iloc[start_index_int:end_index_int]

    # #! 学習済モデルを活用して、X_subsetから予測値を計算
    # #! y_pred_subsetは在庫増減数の予測値を表す
    # y_pred_subset = rf_model.predict(X_subset)

    # #! 在庫データ準備
    df['日時'] = pd.to_datetime(df['日時'])
    df.set_index('日時', inplace=True)
    #df2 = df['在庫増減数（t-52~t-0）']
    df2 = df['在庫数（箱）']

    # #! 1つ前の在庫数データが欲しい
    # #? 昔のやつ
    # #best_range_order = find_columns_with_word_in_name(df, '在庫数（箱）（t-')
    # #yyyy = df[f'{best_range_order}']
    # # yyyyを1時間前の在庫数（箱）に設定
    # #yyyy = df2.shift(1)
    # #? 今のやつ
    # y_base_subset = data['在庫数（箱）'].iloc[start_index_int:end_index_int]
    # #st.dataframe(y_base_subset)

    # # 比較
    # #st.write(x.equals(y))  # Trueであれば一致

    #! ＜目的関数に在庫数_中央値ズレを設定する場合＞
    start_index_int = start_index_int-1
    end_index_int = end_index_int-1
    X_subset = X.iloc[start_index_int:end_index_int]

    #! 学習済モデルを活用して、X_subsetから予測値を計算
    #! y_pred_subsetは在庫増減数の予測値を表す
    y_pred_subset = rf_model.predict(X_subset)

    # 時刻を抽出
    data['時刻'] = pd.to_datetime(data['日時']).dt.hour

    # 時刻ごとの在庫数中央値を計算
    median_by_hour = data.groupby('時刻')['在庫数（箱）'].transform('median')

    # 新しい列を作成
    data['在庫数_中央値'] = median_by_hour

    y_base_subset = data['在庫数_中央値'].iloc[start_index_int:end_index_int]

    #実行結果の確認
    #st.dataframe(y_pred_subset)
    #st.dataframe(y_base_subset)

    #! -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

    #st.dataframe(y_base_subset.head(300))

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
    #cmap = 'RdBu_r'  # 青から赤に変化するカラーマップ

    #df2_subset.index = df2_subset.index.strftime('%Y-%m-%d-%H')
    df.columns = df2.index.strftime('%Y-%m-%d-%H')

    #行の並びを反転
    #df_reversed = df.iloc[::-1]

    # インデックスをリセット
    df2_subset_df = df2.to_frame().reset_index()

    # データフレームの行と列を入れ替え
    df_transposed = df.transpose()
    # インデックスをリセットして日時列を作成
    df_transposed.reset_index(inplace=True)
    # インデックス列の名前を '日時' に変更
    df_transposed.rename(columns={'index': '日時'}, inplace=True)

    #説明変数
    #todo-------------------------------------------------------------------------
    #説明変数もずらす
    zzz = X.iloc[start_index_int:end_index_int]#[start_idx:end_idx]
    #todo-------------------------------------------------------------------------
    # インデックスをリセット
    zzz = zzz.reset_index(drop=True)
    #日時列
    temp_time = df_transposed.reset_index(drop=True)

    #確認用
    #first_datetime_df1 = data['日時'].iloc[0]
    #first_datetime_df2 = temp_time['日時'].iloc[0]
    #first_datetime_df3 = df_transposed['日時'].iloc[0]
    #print(f"dataの日時列の最初の値: {first_datetime_df1}")
    #print(f"df_transposedの日時列の最初の値: {first_datetime_df3}")
    #print(f"temp_timeの日時列の最初の値: {first_datetime_df2}")

    #! 日時列と説明変数を結合
    merged_df = pd.concat([temp_time[['日時']], zzz], axis=1)
    
    #! 変数名を変更する
    line_data = df2_subset_df #在庫データ
    bar_data = df_transposed #SHAP値
    df2 = merged_df #元データ
    
    #! 在庫データのデータフレーム化
    line_df = pd.DataFrame(line_data)
    line_df['日時'] = pd.to_datetime(line_df['日時'], format='%Y%m%d%H')

    #! SHAP値のデータフレーム化
    bar_df = pd.DataFrame(bar_data)
    bar_df['日時'] = pd.to_datetime(bar_df['日時'])
    
    #! 元データのデータフレーム化
    df2 = pd.DataFrame(df2)
    df2['日時'] = pd.to_datetime(df2['日時'])

    #確認
    #st.dataframe(line_df.head(300))
    #print("増減")
    #print(y_pred_subset)
    #st.dataframe(y_pred_subset)
    #print("ベース")
    #print(y_base_subset)
    #st.dataframe(y_base_subset)

    #! 開示時間と終了時間を計算
    #start_datetime = bar_df['日時'].min().to_pydatetime()
    #end_datetime = bar_df['日時'].max().to_pydatetime()

    #Activedata = Activedata[(Activedata['日付'] >= start_datetime) & 
                                     # (Activedata['日付'] <= end_datetime)]
    
    # bar_dfの時間帯を抽出
    bar_times = bar_df['日時']
    
    #st.dataframe(bar_times)
    #st.dataframe(bar_df['日時'])
    #st.dataframe(Activedata['日付'])

    # Activedataの時間帯を抽出し、bar_dfの時間帯と一致するものをフィルタリング
    Activedata = Activedata[Activedata['日付'].isin(bar_times)]

    #! ヘッダーを表示
    st.header('在庫情報')

    #! 結果を可視化
    if step3_flag == 0:
        plot_inventory_graph(line_df, y_pred_subset, y_base_subset, Activedata)
    elif step3_flag == 1:
        plot_inventory_graph2(line_df, y_pred_subset, y_base_subset, Activedata, highlight_time)

    #実行結果の確認
    #st.dataframe(line_df)
    
    #実行結果の確認；開始時刻と終了時刻
    #print(strat_datetime,end_datetime)

    #!実行結果の確認：全体SHAPプロットの生成
    # fig, ax = plt.subplots()
    # shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    # #プロットをStreamlitで表示
    # st.pyplot(fig)
    
    #! STEP3の要因分析結果の可視化のために、開始日時（strat_datetime）と終了日時（end_datetime）、
    #! SHAP値（bar_df）、元データ値（df2）を出力する
    return bar_df, df2, line_df

#! ステップ３の処理
def step3(bar_df, df2, selected_datetime, line_df):

    #st.dataframe(df2)
    #st.dataframe(line_df)
    #st.dataframe(bar_df)

    #! 折り返し線を追加
    st.markdown("---")

    #! ヘッダー表示
    st.header('要因分析')

    #! Activeデータの準備
    #Todo 品番名を取り出すために実行、きれいじゃないから要修正
    with open('temp/model_and_data.pkl', 'rb') as file:
        rf_model,rf_model2,rf_model3, X, data, product = pickle.load(file)
    #　Activeデータをダウンロード
    #start_date2 = '202405'
    #end_date2 = '202408'
    #ver = '00'
    file_path = 'temp/activedata.csv'
    Activedata = pd.read_csv(file_path, encoding='shift_jis')
    # 日付列をdatetime型に変換
    Activedata['日付'] = pd.to_datetime(Activedata['日付'], errors='coerce')
    #　品番、整備室情報読み込み
    seibishitsu = product.split('_')[1]#整備室のみ
    product = product.split('_')[0]#品番のみ
    #　同品番、同整備室のデータを抽出
    Activedata = Activedata[(Activedata['品番'] == product) & (Activedata['整備室'] == seibishitsu)]
    #　1時間単位に変換
    Activedata = Activedata.set_index('日付').resample('H').ffill().reset_index()

    #! SHAP値（bar_df）、元データ値（df2）の日時をdatetime型にする　
    bar_df['日時'] = pd.to_datetime(bar_df['日時'])
    df2['日時'] = pd.to_datetime(df2['日時'])
    #st.dataframe(bar_df)

    #! selected_datetime を1時間前に変更
    #selected_datetime = pd.Timestamp(selected_datetime) - pd.Timedelta(hours=1)

    #! 選択された日時のデータを抽出
    filtered_df1 = bar_df[bar_df['日時'] == pd.Timestamp(selected_datetime)]
    filtered_df2 = df2[df2['日時'] == pd.Timestamp(selected_datetime)]

    #todo----------------------------------------------------------------------------------------

    #! 在庫増減のやつ

    # # selected_datetime2を計算
    # selected_datetime2 = pd.Timestamp(selected_datetime) - pd.Timedelta(hours=15)

    # # 指定した時間範囲でデータを抽出
    # filtered_df1_width = bar_df[(bar_df['日時'] >= pd.Timestamp(selected_datetime2)) & 
    #                     (bar_df['日時'] <= pd.Timestamp(selected_datetime))]
    
    # filtered_df2_width = df2[(df2['日時'] >= pd.Timestamp(selected_datetime2)) & 
    #                     (df2['日時'] <= pd.Timestamp(selected_datetime))]
    
    # #st.dataframe(filtered_df1_width)
    # #st.dataframe(filtered_df2_width)

    # # '日時'列は除外した上で各列の合計を計算
    # filtered_df1 = filtered_df1_width.drop(columns=['日時']).sum().to_frame().T
    # # '日時'列は除外した上で各列の合計を計算
    # filtered_df2 = filtered_df2_width.drop(columns=['日時']).mean().to_frame().T

    # #st.dataframe(filtered_df1)

    # #日時列は最後の値を使用
    # filtered_df1['日時'] = filtered_df1_width['日時'].iloc[-1]
    # filtered_df2['日時'] = filtered_df2_width['日時'].iloc[-1]

    # #st.dataframe(filtered_df1)

    #todo------------------------------------------------------------------------------------------

    # #st.dataframe(df2)
    # st.dataframe(filtered_df1)

    #! selected_datetime を1時間前に変更
    #filtered_df1 = bar_df[bar_df['日時'] == (pd.Timestamp(selected_datetime) - pd.Timedelta(hours=1))]
    #filtered_df2 = df2[df2['日時'] == (pd.Timestamp(selected_datetime) - pd.Timedelta(hours=1))]
    
    #! 
    if not filtered_df1.empty:

        zaikosu = line_df.loc[line_df['日時'] == selected_datetime, '在庫数（箱）'].values[0]

        #! いつもの値の推移を追加
        file_path = 'temp/いつもの在庫数.csv'
        basezaiko_df = pd.read_csv(file_path, encoding='shift_jis')
        basezaiko_df['日時'] = pd.to_datetime(basezaiko_df['日時'])
        # line_df の日時範囲に合わせる
        basezaiko_df = basezaiko_df[basezaiko_df['日時'].isin(line_df['日時'])]
        basezaiko = basezaiko_df.loc[basezaiko_df['日時'] == selected_datetime, 'いつもの在庫数（箱）'].values[0]

        #! 2つのmetricを作成
        col1, col2, col3 = st.columns(3)
        col1.metric(label="選択された日時", value=selected_datetime)#, delta="1 mph")
        col2.metric(label="いつもの在庫数（箱）", value=int(basezaiko))
        col3.metric(label="在庫数（箱）", value=int(zaikosu), delta=f"{int(zaikosu)-int(basezaiko)} 箱（いつもの在庫数との差分）")

        #! データを長い形式に変換
        df1_long = filtered_df1.melt(id_vars=['日時'], var_name='変数', value_name='寄与度（SHAP値）')
        #! データフレームを値の降順にソート
        df1_long = df1_long.sort_values(by='寄与度（SHAP値）', ascending=True)

        # ホバーデータに追加の情報を含める
        hover_data = {}
        for i, row in filtered_df2.iterrows():
            for idx, value in row.items():#iteritemsは、pandasのSeriesではitemsに名称が変更
            #for idx, value in row.iteritems():
                if idx != '日時':
                    hover_data[idx] = f"<b>日時:</b> {row['日時']}<br><b>{idx}:</b> {value:.2f}<br>"

        # 横棒グラフ
        fig_bar = px.bar(df1_long,
                         x='寄与度（SHAP値）', y='変数',
                         orientation='h',
                         labels={'寄与度（SHAP値）': '寄与度（SHAP値）', '変数': '変数', '日時': '日時'},
                         title=f"{selected_datetime}のデータ")

        
        # 色の設定
        colors = ['red' if v >= 0 else 'blue' for v in df1_long['寄与度（SHAP値）']]
        # ホバーテンプレートの設定
        # SHAP値ではないものを表示用
        fig_bar.update_traces(
            marker_color=colors,
            hovertemplate=[hover_data[v] for v in df1_long['変数']]
        )

        fig_bar.update_layout(
            #title="要因分析",
            height=500,  # 高さを調整
            width=100,   # 幅を調整
            margin=dict(l=0, r=0, t=30, b=0)
        )

        #! タブの作成
        tab1, tab2 = st.tabs(["ランキング表示", "棒グラフ表示"])

        #! ランキング表示
        with tab1:

            #! もし 'Unnamed: 0' や '日時' が存在する場合にのみ削除するよう変数を作成
            columns_to_drop = []
            if 'Unnamed: 0' in df2.columns:
                columns_to_drop.append('Unnamed: 0')
            if '日時' in df2.columns:
                columns_to_drop.append('日時')
            #!  'Unnamed: 0' や '日時' を削除する
            df2_cleaned = df2.drop(columns=columns_to_drop)

            #! 各要因の値の平均値と中央値を計算
            average_values = df2_cleaned.mean()
            median_values = df2_cleaned.median()

            # 実行結果の確認
            #st.dataframe(median_values)
            #st.header(type(median_values))
            #st.header(median_values.index)

            #! "発注かんばん"と名前が付く要因の開始時間と終了時間を抽出
            def extract_kanban_t_values_from_index(df):

                # 正規表現パターン: 発注かんばん数 + t-X~t-Y
                pattern = r'発注かんばん数.*\（t-(\d+)~t-(\d+)\）'
                
                # インデックスを走査
                for index in df.index:
                    index_value = str(index)  # インデックスの値を文字列として取得
                    match = re.search(pattern, index_value)  # 正規表現でマッチング
                    if match:
                        X = match.group(1)  # Xの値
                        Y = match.group(2)  # Yの値
                        return X, Y  # 一つのセルが見つかれば終了
                return None, None
            
            #! "発注かんばん"と名前が付く要因の開始遅れ時間と終了遅れ時間を抽出
            hacchu_start, hacchu_end = extract_kanban_t_values_from_index(median_values)
            # 実行結果の確認
            #st.write(hacchu_start, hacchu_end)

            #! 抽出した開始遅れ時間と終了遅れ時間もとに開始時間と終了時間を計算
            def calculate_hacchu_times(hacchu_start, hacchu_end, time_str):
                # hacchu_startとhacchu_endが文字列の場合、整数に変換
                hacchu_start = int(hacchu_start)
                hacchu_end = int(hacchu_end)

                #hacchu_start = 2
                #hacchu_end = 24

                # 時間の文字列をdatetimeオブジェクトに変換
                base_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                
                # hacchu_start時間とhacchu_end時間を引く
                time_start = base_time - timedelta(hours=hacchu_start)
                time_end = base_time - timedelta(hours=hacchu_end)
                
                # 結果を表示
                #st.write(f"{time_str} - {hacchu_start} 時間 = {time_start}")
                #st.write(f"{time_str} - {hacchu_end} 時間 = {time_end}")

                return time_start, time_end

            #! 開始時間と終了時間を計
            hacchu_start_time, hacchu_end_time = calculate_hacchu_times(hacchu_start, hacchu_end, selected_datetime)

            #! 特定の時間帯のデータを抽出
            filtered_data = Activedata[(Activedata['日付'] >= hacchu_end_time) & (Activedata['日付'] <= hacchu_start_time)]

            # 実行結果の確認
            #st.dataframe(filtered_data)

            #増減のとき
            #total_ave = filtered_data['便Ave'].sum()/24*filtered_data['サイクル回数'].median()
            total_ave = filtered_data['便Ave'].iloc[0]

            # 実行結果の確認
            #st.write(total_ave)

            # DataFrameに変換
            average_df = pd.DataFrame(average_values, columns=["平均値"])
            average_df.index.name = '変数'
            median_df = pd.DataFrame(median_values, columns=["基準値"])
            median_df.index.name = '変数'

            def update_values_for_kanban(df,total_ave):
                # インデックスを走査
                for index in df.index:
                    # インデックスに「発注かんばん数」を含むかチェック
                    if "発注かんばん数" in str(index):
                        # 該当する行のすべての値を 10 に設定
                        df.loc[index] = total_ave
                return df

            #! 中央値を更新
            median_df = update_values_for_kanban(median_df,total_ave)

            #! 平均値を統合
            df1_long = pd.merge(df1_long, average_df, left_on="変数", right_on="変数", how="left")
            #! 中央値を統合
            df1_long = pd.merge(df1_long, median_df, left_on="変数", right_on="変数", how="left")

            #! SHAPデータフレームを繰り返し処理し、対応する元要因データフレームの値を追加
            for index, row in df1_long.iterrows():
                variable = row['変数']  # SHAPデータフレームの「変数」列を取得
                if variable in filtered_df2.columns:  # 変数名が元要因データフレームの列名に存在する場合
                    # SHAPデータフレームの現在の行に元要因の値を追加
                    df1_long.at[index, '要因の値'] = filtered_df2.loc[filtered_df2['日時'] == row['日時'], variable].values[0]

            #st.dataframe(df1_long)

            #! 順位表を表示
            #* df1_long一例
            #*　	日時	変数	寄与度（SHAP値）	平均値	基準値	要因の値
            #*    0	2024-08-23T04:00:00.000	No1_発注かんばん数（t-40~t-88）	-0.091102726	0.083333333	8.166666667	0
            #*    1	2024-08-23T04:00:00.000	No8_部品置き場の入庫滞留状況（t-0~t-48）	-0.025413021	3.166666667	3	5
            #*    2	2024-08-23T04:00:00.000	No7_間口の平均充足率（t-0~t-48）	-0.018948366	0.563765326	0.564081148	0.570461665
            #*    3	2024-08-23T04:00:00.000	No6_定期便出発状況（t-4~t-6）	-0.009560572	0.514873722	0.7467865	0.780578667
            #*    4	2024-08-23T04:00:00.000	No5_仕入先便到着状況（t-3~t-5）	-0.006888544	3.375	4	1
            #*    5	2024-08-23T04:00:00.000	No3_計画達成率_加重平均（t-0~t-48）	0	0	0	0
            #*    6	2024-08-23T04:00:00.000	No2_計画組立生産台数_加重平均（t-0~t-48）	0.150128654	96.35416667	97.25	125
            display_shap_contributions(df1_long)
          
            # 背景を青くして、情報ボックスのように見せる
            st.markdown("""
            <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;">
            📌 <strong>基準値についての説明（要因の値が大きいか小さいか、正常なのか異常なのかを判断するための指標）</strong><br>
            <ul>
            <li><strong>発注フラグ</strong>：Activeの日量数（箱数）× 対象期間</li>
            <li><strong>発注かんばん数の基準値</strong>：Activeの日量数（箱数）× 対象期間</li>
            <li><strong>計画組立生産台数の基準値</strong>：過去半年の中央値</li>
            <li><strong>組立ラインの稼働率の基準値</strong>：過去半年の中央値</li>
            <li><strong>間口の充足率の基準値</strong>：過去半年の中央値</li>
            <li><strong>西尾東か部品置き場で滞留しているの基準値</strong>：過去半年の中央値</li>
            <li><strong>仕入先便の到着フラグの基準値</strong>：過去半年の中央値</li>
            <li><strong>定期便の出発フラグの基準値</strong>：過去半年の中央値</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            memo_text = st.text_area("メモ（気づいたことをご記入ください）", height=200)
            # 提出ボタン
            if st.button("登録内容"):
                if memo_text.strip():
                    st.success("提出が完了しました！")
                    st.write("登録内容：")
                    st.write(memo_text)
                else:
                    st.warning("メモ内容が空です。入力してください。")

        with tab2:

            #! 棒グラフ表示
            st.plotly_chart(fig_bar, use_container_width=True)


    else:
        st.write("在庫データがありません")
