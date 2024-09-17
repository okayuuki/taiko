#ライブラリのimport
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


# フォント設定の変更（日本語対応のため）
mpl.rcParams['font.family'] = 'MS Gothic'

#データ読み取り用
from read_v2 import read_data, process_Activedata
#データ前処理用
from functions_v2 import display_corr_matrix, calculate_hourly_counts,calculate_business_time_base,calculate_business_time_order, \
    calculate_business_time_reception,calculate_median_lt,find_best_lag_range,create_lagged_features,add_part_supplier_info, \
        find_columns_with_word_in_name,calculate_elapsed_time_since_last_dispatch,timedelta_to_hhmmss,set_arrival_flag, \
            drop_columns_with_word,calculate_window_width,process_shiresakibin_flag,feature_engineering, \
                plot_inventory_graph, display_shap_contributions,plot_inventory_graph2
    
def show_analysis(product):

    #!学習期間（解析期間）任意に設定できるように。直近1年とかで
    start_date = '2023-10-01'
    end_date = '2024-03-31'

    #! 前処理済みのデータをダウンロード
    AutomatedRack_Details_df, arrival_times_df, kumitate_df, teikibin_df, Timestamp_df, zaiko_df = read_data()

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
    

    #! 品番の数だけループを回す
    #! 今は1品番で
    count = 0
    for part_number in [product]:
        
        # 確認用：実行時の条件確認
        # filtered_Timestamp_df = Timestamp_df[Timestamp_df['品番'] == part_number]#特定品番のデータを抽出
        # suppliers = filtered_Timestamp_df['仕入先名'].unique()#該当仕入先名を抽出
        # supplier = str(suppliers[0])
        # count = count + 1
        # print("品番：", part_number)
        # print("仕入先名：", supplier)
        # print("ユニークな品番の数：", len(Timestamp_df['品番'].unique()))
        # print("ループ：", count)

        #! 内容：関所毎のかんばん数（1時間単位）を計算
        #! Args：関所毎のタイムスタンプデータ、開始時間、終了時間
        #! Return：関所毎のかんばん数（1時間単位）
        hourly_counts_of_order, _ , _ = calculate_hourly_counts(Timestamp_df, part_number, order_time_col, start_date, end_date)#発注
        hourly_counts_of_reception, delivery_info, reception_times = calculate_hourly_counts(Timestamp_df, part_number, reception_time_col, start_date, end_date)#検収
        hourly_counts_of_in, _ , _ = calculate_hourly_counts(Timestamp_df, part_number, target_time_col, start_date, end_date)#入庫
        hourly_counts_of_out, _ , _ = calculate_hourly_counts(Timestamp_df, part_number, leave_time_col, start_date, end_date)#出庫

        #! 内容：時間遅れを計算。発注から入庫までの時間、検収から入庫までの時間を計算（非稼動日時間をの取り除いて）
        #! Args：品番、関所毎のタイムスタンプデータ
        #! Return：発注〜入庫LT、検収〜入庫LT（日単位）の中央値
        median_lt_order, median_lt_reception = calculate_median_lt(part_number,Timestamp_df)
        
        # Todo：発注日時は2山ある。発注して4日後に納入せよとかある、土日の影響？
        #! 内容：発注かんばん数の最適な影響時間範囲を見つける
        #! Args：1時間ごとの発注かんばん数、1時間ごとの入庫かんばん数、探索時間範囲
        #! Return：最適相関値、最適開始遅れ、終了範囲遅れ
        min_lag =int(median_lt_order * 24)-4  # LT中央値を基準に最小遅れ時間を設定
        max_lag =int(median_lt_order * 24)+4  # LT中央値を基準に最大遅れ時間を設定
        best_corr_order, best_range_start_order, best_range_end_order = find_best_lag_range(hourly_counts_of_order, hourly_counts_of_in, min_lag, max_lag, '発注かんばん数')

        #! 内容：発注かんばん数の最適な影響時間範囲を見つける
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

        #確認：実行結果
        #st.dataframe(lagged_features.head(300))

        #! 内容：各種情報を追加
        #! lagged_featuresに情報追加
        lagged_features['在庫増減数（t）'] = lagged_features['入庫かんばん数（t）'] - lagged_features['出庫かんばん数（t）']#在庫増減数を計算
        lagged_features['発注かんばん数（t）'] = hourly_counts_of_order#! 発注かんばん数(t)を計算
        lagged_features['納入かんばん数（t）'] = hourly_counts_of_reception#! 納入かんばん数(t)を計算
        lagged_features = add_part_supplier_info(Timestamp_df, lagged_features, part_number)#! 品番と仕入先名を追加
        lagged_features = lagged_features.rename(columns={'仕入先工場名': '発送場所名'})#! コラム名変更
        lagged_features, median_interval = calculate_elapsed_time_since_last_dispatch(lagged_features)#! 過去の出庫からの経過時間を計算
        lagged_features = pd.merge(lagged_features, zaiko_df[['日時', '品番','在庫数（箱）']], on=['品番', '日時'], how='left')#! 自動ラック在庫結合
        lagged_features = pd.merge(lagged_features, AutomatedRack_Details_df, on=['日時'], how='left')#! 1時間ああたりの間口別在庫の計算
        for col in lagged_features.columns:
            if pd.api.types.is_timedelta64_dtype(lagged_features[col]):
                lagged_features[col] = lagged_features[col].fillna(pd.Timedelta(0))
            else:
                lagged_features[col] = lagged_features[col].fillna(0)
        lagged_features = process_shiresakibin_flag(lagged_features, arrival_times_df)#! 仕入先便到着フラグ計算
        lagged_features = pd.merge(lagged_features,kumitate_df[['日時','生産台数_加重平均済','計画生産台数_加重平均済','計画達成率_加重平均済']], on='日時', how='left')# lagged_features と kumitate_df を日時でマージ
        
        #確認：実行結果
        #st.dataframe(lagged_features.head(300))
        
        #! 最適な遅れ時間を計算
        best_range_order = int((best_range_start_order + best_range_end_order)/2)#最適な発注かんばん数の幅
        best_range_reception = int((best_range_start_reception + best_range_end_reception)/2)#最適な納入かんばん数の幅
        
        #!定期便
        lagged_features = pd.merge(lagged_features, teikibin_df[['日時', '荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)']], on='日時', how='left')
        #!特徴量エンジニアリング
        lagged_features = feature_engineering(lagged_features)

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

        #------------------------------------------------------------------------------------------------------------------
        #todo 削除、今は24年度のデータがないから
        start = '2023-12-30'
        end = '2024-03-31'
        #! 日付範囲に基づいてフィルタリングして削除
        lagged_features= lagged_features[~((lagged_features['日時'] >= start) & (lagged_features['日時'] <= end))]
        #------------------------------------------------------------------------------------------------------------------

        data = lagged_features.iloc[300:]#遅れ分削除

        #確認用
        #st.dataframe(lagged_features.head(300))

        #data = data.rename(columns={'仕入先便到着フラグ': f'仕入先便到着状況（t-{best_range_reception}~t-{best_range_reception + timelag}）'})#コラム名変更
        data['定期便出発状況（t-4~t-6）']=data['荷役時間(t-4)']/50+data['荷役時間(t-5)']/50+data['荷役時間(t-6)']/50

        #確認：実行結果
        #st.dataframe(lagged_features.head(300))

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
        
        delay_No4 = best_range_reception
        timelag_No4 = timelag
        data[f'No4_納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）'] = data[f'納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）']
        
        delay_No5 = best_range_reception
        timelag_No5 = 2
        data[f'No5_仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）'] = data[f'仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）']
        
        data['No6_定期便出発状況（t-4~t-6）'] = data['定期便出発状況（t-4~t-6）']
        
        delay_No7 = end_hours_ago
        timelag_No7 = timelag
        data[f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）'] = data[f'間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）']
        
        delay_No8 = end_hours_ago
        timelag_No8 = timelag
        data[f'No8_部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）'] = data[f'部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）']
        
        delay_No9 = end_hours_ago
        timelag_No9 = timelag
        data[f'No9_定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）'] = data[f'定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）']

        #! 説明変数の設定
        X = data[[f'No1_発注かんばん数（t-{delay_No1}~t-{delay_No1+timelag_No1}）',
                  f'No2_計画組立生産台数_加重平均（t-{delay_No2}~t-{delay_No2+timelag_No2}）',
                  f'No3_計画達成率_加重平均（t-{delay_No3}~t-{delay_No3+timelag_No3}）',
                  f'No4_納入フレ（t-{delay_No4}~t-{delay_No4+timelag_No4}）',
                  f'No5_仕入先便到着状況（t-{delay_No5}~t-{delay_No5+timelag_No5}）',
                  'No6_定期便出発状況（t-4~t-6）',#'荷役時間(t-4)','荷役時間(t-5)','荷役時間(t-6)',
                  f'No7_間口の平均充足率（t-{delay_No7}~t-{delay_No7+timelag_No7}）',#f'間口_A1の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_A2の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B1の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B2の充足率（t-{end_hours_ago}~t-{best_range_order}）',f'間口_B3の充足率（t-{end_hours_ago}~t-{best_range_order}）', f'間口_B4の充足率（t-{end_hours_ago}~t-{best_range_order}）',
                  f'No8_部品置き場の入庫滞留状況（t-{delay_No8}~t-{delay_No8+timelag_No8}）',#f'部品置き場からの入庫（t-{end_hours_ago}~t-{best_range_order}）',f'部品置き場で滞留（t-{end_hours_ago}~t-{best_range_order}）',
                  f'No9_定期便にモノ無し（t-{delay_No9}~t-{delay_No9+timelag_No9}）']]
        
        #確認：実行結果
        st.dataframe(X.head(300))

        #! 目的変数の定義
        y = data[f'在庫増減数（t-0~t-{timelag}）']
        #y = data[f'在庫増減数(t)']

        # DataFrame に変換（列名を指定する）
        #y = pd.DataFrame(y, columns=[f'在庫増減数（t-0~t-{best_range_order}）'])

        # StandardScalerを使用して標準化
        #scaler = StandardScaler()
        #y_scaled = pd.DataFrame(scaler.fit_transform(y), columns=y.columns)

        #st.dataframe(X)

        #! データを学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        rf_model = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42)
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
        #st.header(mae2)
        print(f'ランダムフォレスト - テストデータの平均誤差: {mae2}')
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
        
        #todo データをCSVファイルに保存
        with open("一時保存データ.csv", mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
            data.to_csv(f)
        
        return data, rf_model, X

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

def step2(data, rf_model, X, start_index, end_index, step3_flag, highlight_time=None):

    #Todo 品番名を取り出すために実行、きれいじゃないから要修正
    with open('model_and_data.pkl', 'rb') as file:
        rf_model, X, data, product = pickle.load(file)

    Activedata = process_Activedata()
    #st.header(product)
    Activedata = Activedata[Activedata['品番'] == product]

    #st.header(start_index)
    #st.header(end_index)

    # 在庫データに合わせて時間粒度を1時間ごとにリサンプリング
    # 内示データを日付ごとに集約して重複を排除
    #Activedata = Activedata.groupby('日付').mean(numeric_only=True).reset_index()
    Activedata = Activedata.set_index('日付').resample('H').ffill().reset_index()

    #st.dataframe(Activedata.head(300))

    #折り返し線を追加
    st.markdown("---")

    #インデックスが300スタートなのでリセット
    #遅れ時間の計算のため
    data = data.reset_index(drop=True)
    #st.dataframe(data.head(300))

    # SHAP計算
    #explainer = shap.TreeExplainer(rf_model, feature_dependence='tree_path_dependent', model_output='margin')
    explainer = shap.TreeExplainer(rf_model, model_output='raw')
    shap_values = explainer.shap_values(X)

    first_datetime_df = data['日時'].iloc[0]
    print(f"dataの日時列の最初の値: {first_datetime_df}")

    # リストから整数に変換
    start_index_int = start_index[0]#-300
    end_index_int = end_index[0]+1#-300

    #在庫データフレーム
    df = data.iloc[start_index_int:end_index_int]
    print(df.head())

    #st.dataframe(df.head(300))

    first_datetime_df = df.iloc[0]
    print(f"dfの日時列の最初の値: {first_datetime_df}")

    X_subset = X.iloc[start_index_int:end_index_int]
    # モデルを使ってX_subsetから予測値を計算
    y_pred_subset = rf_model.predict(X_subset)

    df['日時'] = pd.to_datetime(df['日時'])
    df.set_index('日時', inplace=True)

    #df2 = df['在庫増減数（t-52~t-0）']
    df2 = df['在庫数（箱）']
    print(df2.head())

    #在庫数（箱）を計算する
    best_range_order = find_columns_with_word_in_name(df, '在庫数（箱）（t-')
    yyyy = df[f'{best_range_order}']
    y_base_subset = yyyy

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
    zzz = X.iloc[start_index_int:end_index_int]#[start_idx:end_idx]
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

    if step3_flag == 0:
        #! 在庫可視化
        plot_inventory_graph(line_df, y_pred_subset, y_base_subset, Activedata)
    elif step3_flag == 1:
        plot_inventory_graph2(line_df, y_pred_subset, y_base_subset, Activedata, highlight_time)

    #st.dataframe(line_df)
    
    #確認；開始時刻と終了時刻
    #print(strat_datetime,end_datetime)

    #確認：全体SHAPプロットの生成
    #fig, ax = plt.subplots()
    #shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    #プロットをStreamlitで表示
    #st.pyplot(fig)
    
    #! STEP3の要因分析結果の可視化のために、開始日時（strat_datetime）と終了日時（end_datetime）、
    #! SHAP値（bar_df）、元データ値（df2）を出力する
    return bar_df, df2, line_df
    
def step3(bar_df, df2, selected_datetime, line_df):

    #! 折り返し線を追加
    st.markdown("---")

    st.header('要因分析')

    bar_df['日時'] = pd.to_datetime(bar_df['日時'])
    df2['日時'] = pd.to_datetime(df2['日時'])

    # 選択された日時のデータを抽出
    filtered_df1 = bar_df[bar_df['日時'] == pd.Timestamp(selected_datetime)]
    filtered_df2 = df2[df2['日時'] == pd.Timestamp(selected_datetime)]
    
    if not filtered_df1.empty:
        zaikosu = line_df.loc[line_df['日時'] == selected_datetime, '在庫数（箱）'].values[0]
        #st.write(f"##### 選択された日時: {selected_datetime}")
        #st.write(f"##### 在庫数（箱）: {int(zaikosu)}")

        # 3つの列を作成
        col1, col2 = st.columns(2)
        col1.metric(label="選択された日時", value=selected_datetime)#, delta="1 mph")
        col2.metric(label="在庫数（箱）", value=int(zaikosu))

        # データを長い形式に変換
        df1_long = filtered_df1.melt(id_vars=['日時'], var_name='変数', value_name='寄与度（SHAP値）')
        # データフレームを値の降順にソート
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

        # タブの作成
        tab1, tab2 = st.tabs(["ランキング表示", "棒グラフ表示"])

        with tab1:

            # もし 'Unnamed: 0' や '日時' が存在する場合にのみ削除する
            columns_to_drop = []
            if 'Unnamed: 0' in df2.columns:
                columns_to_drop.append('Unnamed: 0')
            if '日時' in df2.columns:
                columns_to_drop.append('日時')

            # 不要な列を削除します
            df2_cleaned = df2.drop(columns=columns_to_drop)

            # 平均値と中央値を計算
            average_values = df2_cleaned.mean()
            median_values = df2_cleaned.median()

            # DataFrameに変換
            average_df = pd.DataFrame(average_values, columns=["平均値"])
            average_df.index.name = '変数'
            median_df = pd.DataFrame(median_values, columns=["いつもの値（中央値）"])
            median_df.index.name = '変数'

            #統合
            df1_long = pd.merge(df1_long, average_df, left_on="変数", right_on="変数", how="left")
            df1_long = pd.merge(df1_long, median_df, left_on="変数", right_on="変数", how="left")

            # SHAPデータフレームを繰り返し処理し、対応する元要因データフレームの値を追加
            for index, row in df1_long.iterrows():
                variable = row['変数']  # SHAPデータフレームの「変数」列を取得
                if variable in filtered_df2.columns:  # 変数名が元要因データフレームの列名に存在する場合
                    # SHAPデータフレームの現在の行に元要因の値を追加
                    df1_long.at[index, '要因の値'] = filtered_df2.loc[filtered_df2['日時'] == row['日時'], variable].values[0]

            display_shap_contributions(df1_long)

        with tab2:
            # グラフ表示
            st.plotly_chart(fig_bar, use_container_width=True)


    else:
        st.write("在庫データがありません")
