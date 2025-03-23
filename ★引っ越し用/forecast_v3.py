#予測用

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
import streamlit.components.v1 as components
import base64

#! 自作ライブラリのimport
from read_v3 import read_data, process_Activedata, read_syozailt_by_using_archive_data, read_activedata_by_using_archive_data,read_zaiko_by_using_archive_data, calculate_supplier_truck_arrival_types2

from functions_v3 import process_shiresakibin_flag

#! リミット計算
def show_forecast( unique_product, start_datetime, selected_zaiko, past_hours, type):

    #start_date = '2024-05-01-00'
    #end_date = '2024-08-31-00'

    start_date = start_datetime.strftime('%Y-%m-%d-%H')
    end_datetime = start_datetime + relativedelta(months=1)
    end_date = end_datetime.strftime('%Y-%m-%d-%H')

    selected_datetime = start_datetime

    #! 品番、整備室コードを抽出
    product = unique_product.split('_')[0]
    seibishitsu = unique_product.split('_')[1]

    #! パラメータ設定
    prediction_hours = 12#何時間先まで予測するのか
    #past_hours = 5
    lookback_hours = past_hours+2

    # タイトル表示
    st.header('予測結果')

    #!----------------------------------------------------------------------- 
    #! 自動ラックの在庫データの読み込みと処理
    #!-----------------------------------------------------------------------
    # todo 引数関係なく全データ読み込みしてる
    zaiko_df = read_zaiko_by_using_archive_data(selected_datetime.strftime('%Y-%m-%d-%H'), selected_datetime.strftime('%Y-%m-%d-%H'))
    # todo
    #! 品番列を昇順にソート
    zaiko_df = zaiko_df.sort_values(by='品番', ascending=True)
    #! 無効な値を NaN に変換
    zaiko_df['拠点所番地'] = pd.to_numeric(zaiko_df['拠点所番地'], errors='coerce')
    #! 品番ごとに欠損値（NaN）を埋める(前方埋め後方埋め)
    zaiko_df['拠点所番地'] = zaiko_df.groupby('品番')['拠点所番地'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    #! それでも置換できないものはNaN を 0 で埋める
    zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].fillna(0).astype(int).astype(str)
    #! str型に変換
    zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].astype(int).astype(str)
    #! 受入場所情報準備
    file_path = 'temp/マスター_品番&仕入先名&仕入先工場名.csv'
    syozaikyotenchi_data = pd.read_csv(file_path, encoding='shift_jis')
    #! 空白文字列や非数値データをNaNに変換
    syozaikyotenchi_data['拠点所番地'] = pd.to_numeric(syozaikyotenchi_data['拠点所番地'], errors='coerce')
    #! str型に変換
    syozaikyotenchi_data['拠点所番地'] = syozaikyotenchi_data['拠点所番地'].fillna(0).astype(int).astype(str)
    #! 受入場所追加
    zaiko_df = pd.merge(zaiko_df, syozaikyotenchi_data[['品番','拠点所番地','受入場所']], on=['品番', '拠点所番地'], how='left')
    #! 日付列を作成
    zaiko_df['日付'] = zaiko_df['日時'].dt.date
    #! 品番_受入番号作成
    zaiko_df['品番_受入場所'] = zaiko_df['品番'].astype(str) + "_" + zaiko_df['受入場所'].astype(str)
    zaiko_df = zaiko_df[(zaiko_df['品番'] == product) & (zaiko_df['受入場所'] == seibishitsu)]
    # 特定の日時のデータを抽出
    zaiko_df = zaiko_df[zaiko_df['日時'] == start_datetime]
    # 日時を再度datetime型に変換（念のため）
    zaiko_df['日時'] = pd.to_datetime(zaiko_df['日時'])
    zaiko_extracted = zaiko_df[['日時', '在庫数（箱）']]
    #st.dataframe(zaiko_extracted)

    #!-----------------------------------------------------------------------
    #! 所在管理リードタイムのデータ
    #!-----------------------------------------------------------------------
    #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
    Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
    # '更新日時'列に無効な日時データがある行を削除する
    data_cleaned = Timestamp_df.dropna(subset=['検収日時'])
    #st.dataframe(data_cleaned.head(50000))
    # 特定の品番の商品データを抽出
    data_cleaned = data_cleaned[(data_cleaned['品番'] == product) & (data_cleaned['整備室コード'] == seibishitsu)]
    # 時間ごとにグループ化し、各時間でのかんばん数をカウントする
    data_cleaned['日時'] = data_cleaned['検収日時'].dt.floor('H')  # 時間単位に丸める
    hourly_kanban_count = data_cleaned.groupby('日時').size().reset_index(name='納入予定かんばん数')
    #st.dataframe(hourly_kanban_count)

    # 時間の範囲を決定し、欠損時間帯を補完する
    full_time_range = pd.date_range(start=hourly_kanban_count['日時'].min(),end=hourly_kanban_count['日時'].max(),freq='H')

    # 全ての時間を含むデータフレームを作成し、欠損値を0で埋める
    hourly_kanban_count_full = pd.DataFrame(full_time_range, columns=['日時']).merge(hourly_kanban_count, on='日時', how='left').fillna(0)

    # かんばん数を整数に戻す
    hourly_kanban_count_full['納入予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].astype(int)

    # '予測入庫時間'列として、5時間前のかんばん数を追加する
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].shift(past_hours)

    # 欠損値（最初の5時間分）を0で埋める
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['工場到着予定かんばん数'].fillna(0).astype(int)
    #st.dataframe(hourly_kanban_count_full)

    #!-----------------------------------------------------------------------
    #! Activedataの処理
    #!-----------------------------------------------------------------------
    activedata = read_activedata_by_using_archive_data(start_date, end_date, 0)
    # 特定の品番の商品データを抽出
    activedata = activedata[(activedata['品番'] == product) & (activedata['受入場所'] == seibishitsu)]
    #st.dataframe(activedata)
    #! 稼働時間で割る処理 (休憩時間の考慮が必要か？)
    activedata['日量数（箱数）'] = activedata['日量数']/activedata['収容数']
    activedata['日量数（箱数）/稼働時間'] = activedata['日量数（箱数）'] / 16.5
    activedata['日付'] = pd.to_datetime(activedata['日付'])#これしないと次の.resample('H')でエラーが出る
    # 日付を基準に1時間ごとのデータに変換
    activedata = activedata.set_index('日付').resample('H').ffill().reset_index()
    # '日付' をdatetime型に変換
    activedata['日付'] = pd.to_datetime(activedata['日付'])
    activedata = activedata.rename(columns={'日付': '日時'})

    # 必要な列とデータ型を整備
    activedata['日時'] = pd.to_datetime(activedata['日時'])
    activedata = activedata.sort_values('日時').reset_index(drop=True)

    # 処理対象列を決定
    target_column = '日量数（箱数）'

    # 年月情報を追加
    activedata['年'] = activedata['日時'].dt.year
    activedata['月'] = activedata['日時'].dt.month

    # 新しい列「月末までの最大日量数（箱数）」を初期化
    activedata['月末までの最大日量数（箱数）'] = None

    # 年・月ごとにグループ化して処理
    for (year, month), group in activedata.groupby(['年', '月']):
        for idx, row in group.iterrows():
            current_time = row['日時']
            # 現在時刻以降、同じ月内のデータをフィルタ
            subset = group[group['日時'] >= current_time]
            # 対象の「日量数（箱数）」の最大値を取得
            max_value = subset[target_column].max()
            # 値をセット
            activedata.loc[idx, '月末までの最大日量数（箱数）'] = max_value

    if type == 1:
        activedata['日量数（箱数）/稼働時間'] = activedata['月末までの最大日量数（箱数）']/(16.5)

    #st.dataframe(activedata)

    # '日付' と '日量数（箱数）' の列のみを抽出
    activedata_extracted = activedata[['日時', '日量数（箱数）/稼働時間']]
    #st.dataframe(activedata_extracted)

    # 在庫データの開始時刻を取得
    start_time = zaiko_extracted.iloc[0]['日時']
    # 開始時刻から20時間後までのデータを抽出
    end_time = start_time + pd.Timedelta(hours=prediction_hours)
    # 稼働有無設定-------------------------------------------------------------------
    # 日時リストを1時間ごとに作成
    date_range = pd.date_range(start=start_time, end=end_time, freq='H')
    # DataFrame を作成し、稼働フラグ列を 0 で初期化
    kado_df = pd.DataFrame({
        '日時': date_range,
        '稼働フラグ': 0
    })
    for idx, row in kado_df.iterrows():
        hour = row['日時'].hour  # 時間だけ取得
        if hour in [12, 17, 18, 19, 20, 21, 2, 6, 7]:
            kado_df.at[idx, '稼働フラグ'] = 0
        else:
            kado_df.at[idx, '稼働フラグ'] = 1

    #------------------------------------------------------------------------------------
    #
    #-----------------------------------------------------------------------------------

    # グループAとグループBの時間帯
    group_a_hours = [17, 18, 19, 20, 21]
    group_b_hours = [6, 7]

    # グループごとにフィルタ
    filtered_kado_a = kado_df[kado_df['日時'].dt.hour.isin(group_a_hours)]
    filtered_kado_b = kado_df[kado_df['日時'].dt.hour.isin(group_b_hours)]

    # 年月日だけ取り出してユニーク化
    dates_group_a = filtered_kado_a['日時'].dt.date.unique()
    dates_group_b = filtered_kado_b['日時'].dt.date.unique()

    # hiru_time = None
    # yoru_time = None
    # if len(dates_group_a) != 0:
    #     hiru_time = filtered_kado_a['日時'].dt.date.unique()[0]
    # if len((dates_group_a)) !=0:
    #     yoru_time = filtered_kado_b['日時'].dt.date.unique()[0]


    # if hiru_time != None:

    # 出力
    #st.write("【Group A】[17, 18, 19, 20, 21] の時間帯を含む日付一覧")
    #st.write(dates_group_a)

    #その日の残業時間をとる
    #時間に応じて稼働フラグを更新

    #st.write("\n【Group B】[6, 7] の時間帯を含む日付一覧")
    #st.write(dates_group_b)

    #その日の残業時間をとる
    #時間に応じて稼働フラグを更新
    
    #稼働有無設定---------------------------------------------------------------------
    filtered_activedata = activedata_extracted[(activedata_extracted['日時'] >= start_time) & (activedata_extracted['日時'] < end_time)]
    #st.write(start_time,end_time)
    #st.dataframe(filtered_activedata)

    # 各時間後の消費量および入庫量を考慮した在庫数を計算
    inventory_after_adjustments = []
    # 現在の在庫数を初期値として設定
    current_inventory = selected_zaiko#zaiko_extracted.iloc[0]['在庫数（箱）']

    # 3つの列を作成
    col1, col2 = st.columns(2)
    col1.metric(label="選択された日時", value=str(start_datetime))#, delta="1 mph")
    col2.metric(label="入力された組立ラインの在庫数（箱）", value=int(current_inventory))

    # 時間ごとの在庫数を更新しながらリストに追加
    for i, row in filtered_activedata.iterrows():
        kanban_row = hourly_kanban_count_full[hourly_kanban_count_full['日時'] == row['日時']]
        incoming_kanban = kanban_row['工場到着予定かんばん数'].values[0] if not kanban_row.empty else 0
        inventory_after_adjustments.append({
            '日時': row['日時'],
            '在庫数（箱）': current_inventory
        })
        # 最初のタイムスタンプでは消費を引かないが、以降は消費量と入庫量を調整
        if i != 0:
            # 該当日時の稼働フラグを取得
            kado_flag = kado_df.loc[kado_df['日時'] == row['日時'], '稼働フラグ'].values[0]
            #st.write(kado_flag)
            if kado_flag == 1:
                current_inventory = current_inventory - row['日量数（箱数）/稼働時間']  # 消費量を引く
            current_inventory = current_inventory + incoming_kanban  # 入庫量を足す
            

    # 計算結果をDataFrameに変換
    inventory_df_adjusted = pd.DataFrame(inventory_after_adjustments)
    #st.dataframe(inventory_df_adjusted)

    # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
    actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
    forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

    # 時間軸を統一するため、全時間の範囲を作成
    #full_time_range = pd.date_range(start=actual_data['日時'].min(), end=forecast_data['日時'].max(), freq='H')

    # データフレームをそれぞれこの時間軸に合わせて再構築し、欠損値を埋める
    #actual_data = actual_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})
    #forecast_data = forecast_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})

    # 欠損値はそれぞれ0に置き換える（必要に応じて）
    #actual_data['在庫数（箱）'].fillna(0, inplace=True)
    #forecast_data['在庫数（箱）'].fillna(0, inplace=True)

    # グラフの作成
    fig = go.Figure()

    # 実際のデータを青色で描画
    fig.add_trace(go.Bar(
        x=actual_data['日時'], 
        y=actual_data['在庫数（箱）'], 
        name='実績', 
        marker_color='blue', 
        opacity=0.3
    ))

    # 予測データをオレンジ色で追加描画
    fig.add_trace(go.Bar(
        x=forecast_data['日時'], 
        y=forecast_data['在庫数（箱）'], 
        name='予測', 
        marker_color='orange', 
        opacity=0.3
    ))

    # x軸を1時間ごとに表示する設定
    fig.update_layout(
        title='予測結果',  # ここでタイトルを設定
        xaxis_title='日時',  # x軸タイトル
        yaxis_title='在庫数（箱）',  # y軸タイトル
        xaxis=dict(
            tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
            dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
        ),
        barmode='group'  # 複数のバーをグループ化
    )

    # グラフをStreamlitで表示
    st.plotly_chart(fig)

    # 5時間前の日時を計算
    hours_before = start_time - pd.Timedelta(hours=lookback_hours)

    # ユーザーに結果を表示する
    hourly_kanban_count_full = hourly_kanban_count_full[(hourly_kanban_count_full['日時'] >= hours_before) & (hourly_kanban_count_full['日時'] < end_time)]

    # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
    hourly_kanban_count_full['※注釈                                                                               '] = hourly_kanban_count_full['日時'].apply(
        lambda x: 'あなたはこの時間を選択しました' if x == start_time else ('過去' if x < start_time else '未来')
    )

    # '日時'列でstart_timeに一致する行をハイライト
    def highlight_start_time(row):
        return ['background-color: yellow' if row['日時'] == start_time else '' for _ in row]
    
    st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

    # 注釈を追加（例としてstart_timeを表示）
    st.markdown(f"")
    st.markdown(f"")
    st.markdown(f"**下の表で予測の内容を確認できます。**")
    #st.code(f"計算式：在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

    # 'hourly_kanban_count_full' と 'inventory_df_adjusted' を '日時' をキーに結合
    merged_df = pd.merge(hourly_kanban_count_full, inventory_df_adjusted, on='日時', how='outer')
    activedata_extracted = activedata_extracted[(activedata_extracted['日時'] >= hours_before) & (activedata_extracted['日時'] < end_time)]
    merged_df = pd.merge(merged_df, activedata_extracted, on='日時', how='outer')

    # 必要に応じてNaNを0に置き換える（在庫数やかんばん数に関して）
    merged_df.fillna(0, inplace=True)

    # Streamlitで表示
    # データフレームの列の順番を指定
    new_column_order = ['日時', '納入予定かんばん数', '工場到着予定かんばん数', '日量数（箱数）/稼働時間', '在庫数（箱）','※注釈                                                                               ']
    # 列の順番を変更
    merged_df = merged_df[new_column_order]

    # 条件に該当する行の在庫数を "-" にする
    merged_df.loc[
        (merged_df['日時'] >= hours_before) & 
        (merged_df['日時'] < start_time), 
        '在庫数（箱）'
    ] = "-"

    # '日時'列でstart_timeに一致する行をハイライトして表示
    st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))

# リミット計算
# def show_forecast2( unique_product, start_datetime, selected_zaiko):

#     start_date = '2024-05-01-00'
#     end_date = '2024-08-31-00'

#     #! 品番、整備室コードを抽出
#     product = unique_product.split('_')[0]
#     seibishitsu = unique_product.split('_')[1]

#     #! パラメータ設定
#     prediction_hours = 24#何時間先まで予測するのか
#     past_hours = 5
#     lookback_hours = past_hours+2

#     # タイトル表示
#     st.header('予測結果')

#     #!----------------------------------------------------------------------- 
#     #! 自動ラックの在庫データの読み込みと処理
#     #!-----------------------------------------------------------------------
#     zaiko_df = read_zaiko_by_using_archive_data(start_date, end_date)
#     # 品番列の空白を削除
#     zaiko_df['品番'] = zaiko_df['品番'].str.strip()
#     # '計測日時'をdatetime型に変換
#     #zaiko_df['計測日時'] = pd.to_datetime(zaiko_df['計測日時'], errors='coerce')
#     # 列名 '計測日時' を '日時' に変更
#     #zaiko_df = zaiko_df.rename(columns={'計測日時': '日時'})
#     # 特定の品番の商品データを抽出
#     zaiko_df = zaiko_df[zaiko_df['品番'] == product]
#     # 特定の日時のデータを抽出
#     zaiko_df = zaiko_df[zaiko_df['日時'] == start_datetime]
#     # 日時を再度datetime型に変換（念のため）
#     zaiko_df['日時'] = pd.to_datetime(zaiko_df['日時'])
#     # '日時' と '在庫数（箱）' の列のみを抽出
#     zaiko_extracted = zaiko_df[['日時', '在庫数（箱）']]

#     #!-----------------------------------------------------------------------
#     #! 所在管理リードタイムのデータ
#     #!-----------------------------------------------------------------------
#     #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
#     Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
#     # '更新日時'列に無効な日時データがある行を削除する
#     data_cleaned = Timestamp_df.dropna(subset=['検収日時'])
#     #st.dataframe(data_cleaned.head(50000))
#     # 特定の品番の商品データを抽出
#     data_cleaned = data_cleaned[(data_cleaned['品番'] == product) & (data_cleaned['整備室コード'] == seibishitsu)]
#     # 時間ごとにグループ化し、各時間でのかんばん数をカウントする
#     data_cleaned['日時'] = data_cleaned['検収日時'].dt.floor('H')  # 時間単位に丸める
#     hourly_kanban_count = data_cleaned.groupby('日時').size().reset_index(name='納入予定かんばん数')
#     #st.dataframe(hourly_kanban_count)

#     # 時間の範囲を決定し、欠損時間帯を補完する
#     full_time_range = pd.date_range(start=hourly_kanban_count['日時'].min(),end=hourly_kanban_count['日時'].max(),freq='H')

#     # 全ての時間を含むデータフレームを作成し、欠損値を0で埋める
#     hourly_kanban_count_full = pd.DataFrame(full_time_range, columns=['日時']).merge(hourly_kanban_count, on='日時', how='left').fillna(0)

#     # かんばん数を整数に戻す
#     hourly_kanban_count_full['納入予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].astype(int)

#     # '予測入庫時間'列として、5時間前のかんばん数を追加する
#     hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].shift(past_hours)

#     # 欠損値（最初の5時間分）を0で埋める
#     hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['工場到着予定かんばん数'].fillna(0).astype(int)

#     #!-----------------------------------------------------------------------
#     #! Activedataの処理
#     #!-----------------------------------------------------------------------
#     activedata = read_activedata_by_using_archive_data(start_date, end_date, 0)
#     # 特定の品番の商品データを抽出
#     activedata = activedata[activedata['品番'] == product]
#     #st.dataframe(activedata)
#     #! 稼働時間で割る処理 (休憩時間の考慮が必要か？)
#     activedata['日量数（箱数）'] = activedata['日量数']/activedata['収容数']
#     activedata['日量数（箱数）/稼働時間'] = activedata['日量数（箱数）'] / 16.5
#     activedata['日付'] = pd.to_datetime(activedata['日付'])#これしないと次の.resample('H')でエラーが出る
#     # 日付を基準に1時間ごとのデータに変換
#     activedata = activedata.set_index('日付').resample('H').ffill().reset_index()
#     # '日付' をdatetime型に変換
#     activedata['日付'] = pd.to_datetime(activedata['日付'])
#     activedata = activedata.rename(columns={'日付': '日時'})
#     # '日付' と '日量数（箱数）' の列のみを抽出
#     activedata_extracted = activedata[['日時', '日量数（箱数）/稼働時間']]

#     # 在庫データの開始時刻を取得
#     start_time = zaiko_extracted.iloc[0]['日時']
#     # 開始時刻から20時間後までのデータを抽出
#     end_time = start_time + pd.Timedelta(hours=prediction_hours)
#     filtered_activedata = activedata_extracted[(activedata_extracted['日時'] >= start_time) & (activedata_extracted['日時'] < end_time)]

#     # 各時間後の消費量および入庫量を考慮した在庫数を計算
#     inventory_after_adjustments = []
#     # 現在の在庫数を初期値として設定
#     current_inventory = selected_zaiko#zaiko_extracted.iloc[0]['在庫数（箱）']

#     # 3つの列を作成
#     col1, col2 = st.columns(2)
#     col1.metric(label="選択された日時", value=str(start_datetime))#, delta="1 mph")
#     col2.metric(label="入力された組立ラインの在庫数（箱）", value=int(current_inventory))

#     # 時間ごとの在庫数を更新しながらリストに追加
#     for i, row in filtered_activedata.iterrows():
#         kanban_row = hourly_kanban_count_full[hourly_kanban_count_full['日時'] == row['日時']]
#         incoming_kanban = kanban_row['工場到着予定かんばん数'].values[0] if not kanban_row.empty else 0
#         inventory_after_adjustments.append({
#             '日時': row['日時'],
#             '在庫数（箱）': current_inventory
#         })
#         # 最初のタイムスタンプでは消費を引かないが、以降は消費量と入庫量を調整
#         if i != 0:
#             current_inventory = current_inventory - row['日量数（箱数）/稼働時間']  # 消費量を引く
#             current_inventory = current_inventory + incoming_kanban  # 入庫量を足す
            

#     # 計算結果をDataFrameに変換
#     inventory_df_adjusted = pd.DataFrame(inventory_after_adjustments)

#     # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
#     actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
#     forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

#     # 時間軸を統一するため、全時間の範囲を作成
#     #full_time_range = pd.date_range(start=actual_data['日時'].min(), end=forecast_data['日時'].max(), freq='H')

#     # データフレームをそれぞれこの時間軸に合わせて再構築し、欠損値を埋める
#     #actual_data = actual_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})
#     #forecast_data = forecast_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})

#     # 欠損値はそれぞれ0に置き換える（必要に応じて）
#     #actual_data['在庫数（箱）'].fillna(0, inplace=True)
#     #forecast_data['在庫数（箱）'].fillna(0, inplace=True)

#     # グラフの作成
#     fig = go.Figure()

#     # 実際のデータを青色で描画
#     fig.add_trace(go.Bar(
#         x=actual_data['日時'], 
#         y=actual_data['在庫数（箱）'], 
#         name='実績', 
#         marker_color='blue', 
#         opacity=0.3
#     ))

#     # 予測データをオレンジ色で追加描画
#     fig.add_trace(go.Bar(
#         x=forecast_data['日時'], 
#         y=forecast_data['在庫数（箱）'], 
#         name='予測', 
#         marker_color='orange', 
#         opacity=0.3
#     ))

#     # x軸を1時間ごとに表示する設定
#     fig.update_layout(
#         title='予測結果',  # ここでタイトルを設定
#         xaxis_title='日時',  # x軸タイトル
#         yaxis_title='在庫数（箱）',  # y軸タイトル
#         xaxis=dict(
#             tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
#             dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
#         ),
#         barmode='group'  # 複数のバーをグループ化
#     )

#     # グラフをStreamlitで表示
#     st.plotly_chart(fig)

#     # 5時間前の日時を計算
#     hours_before = start_time - pd.Timedelta(hours=lookback_hours)

#     # ユーザーに結果を表示する
#     hourly_kanban_count_full = hourly_kanban_count_full[(hourly_kanban_count_full['日時'] >= hours_before) & (hourly_kanban_count_full['日時'] < end_time)]

#     # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
#     hourly_kanban_count_full['※注釈                                                                               '] = hourly_kanban_count_full['日時'].apply(
#         lambda x: 'あなたはこの時間を選択しました' if x == start_time else ('過去' if x < start_time else '未来')
#     )

#     # '日時'列でstart_timeに一致する行をハイライト
#     def highlight_start_time(row):
#         return ['background-color: yellow' if row['日時'] == start_time else '' for _ in row]
    
#     st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

#     # 注釈を追加（例としてstart_timeを表示）
#     st.markdown(f"")
#     st.markdown(f"")
#     st.markdown(f"**下の表で予測の内容を確認できます。**")
#     #st.code(f"計算式：在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

#     # 'hourly_kanban_count_full' と 'inventory_df_adjusted' を '日時' をキーに結合
#     merged_df = pd.merge(hourly_kanban_count_full, inventory_df_adjusted, on='日時', how='outer')
#     activedata_extracted = activedata_extracted[(activedata_extracted['日時'] >= hours_before) & (activedata_extracted['日時'] < end_time)]
#     merged_df = pd.merge(merged_df, activedata_extracted, on='日時', how='outer')

#     # 必要に応じてNaNを0に置き換える（在庫数やかんばん数に関して）
#     merged_df.fillna(0, inplace=True)

#     # Streamlitで表示
#     # データフレームの列の順番を指定
#     new_column_order = ['日時', '納入予定かんばん数', '工場到着予定かんばん数', '日量数（箱数）/稼働時間', '在庫数（箱）','※注釈                                                                               ']
#     # 列の順番を変更
#     merged_df = merged_df[new_column_order]

#     # 条件に該当する行の在庫数を "-" にする
#     merged_df.loc[
#         (merged_df['日時'] >= hours_before) & 
#         (merged_df['日時'] < start_time), 
#         '在庫数（箱）'
#     ] = "-"

#     # '日時'列でstart_timeに一致する行をハイライトして表示
#     st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))

#! 在庫シミュレーション
def show_zaiko_simulation( selected_datetime, change_rate):

    #! 日量箱数を時間単位にするために
    #todo 稼働時間などを考えるなら、16.5で割る必要があるかもだが、その場合はどの時間帯が稼働時間分かる必要がある
    kado_time = 25
    selected_datetime_end = selected_datetime + timedelta(hours=kado_time-1)

    #! 選択情報表示
    col1, col2 = st.columns(2)
    col1.metric(label="シミュレーション開始時間", value=selected_datetime.strftime("%Y-%m-%d %H:%M"))
    col2.metric(label="シミュレーション終了時間", value=selected_datetime_end.strftime("%Y-%m-%d %H:%M"))
    #col2.metric(label="選択変動率", value=change_rate)

    # 1時間ごとの時間列（24時間分）を作成
    time_series = pd.date_range(start=selected_datetime, periods=24, freq="H")
    # データフレームを作成
    time_df = pd.DataFrame({"日時": time_series})
    #st.dataframe(time_df)

    #! 自動ラックの在庫データを読み込み
    # todo 引数関係なく全データ読み込みしてる
    zaiko_df = read_zaiko_by_using_archive_data(selected_datetime.strftime('%Y-%m-%d-%H'), selected_datetime.strftime('%Y-%m-%d-%H'))
    # todo
    #! 品番列を昇順にソート
    zaiko_df = zaiko_df.sort_values(by='品番', ascending=True)
    #! 無効な値を NaN に変換
    zaiko_df['拠点所番地'] = pd.to_numeric(zaiko_df['拠点所番地'], errors='coerce')
    #! 品番ごとに欠損値（NaN）を埋める(前方埋め後方埋め)
    zaiko_df['拠点所番地'] = zaiko_df.groupby('品番')['拠点所番地'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    #! それでも置換できないものはNaN を 0 で埋める
    zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].fillna(0).astype(int).astype(str)
    #! str型に変換
    zaiko_df['拠点所番地'] = zaiko_df['拠点所番地'].astype(int).astype(str)
    #! 受入場所情報準備
    file_path = 'temp/マスター_品番&仕入先名&仕入先工場名.csv'
    syozaikyotenchi_data = pd.read_csv(file_path, encoding='shift_jis')
    #! 空白文字列や非数値データをNaNに変換
    syozaikyotenchi_data['拠点所番地'] = pd.to_numeric(syozaikyotenchi_data['拠点所番地'], errors='coerce')
    #! str型に変換
    syozaikyotenchi_data['拠点所番地'] = syozaikyotenchi_data['拠点所番地'].fillna(0).astype(int).astype(str)
    #! 受入場所追加
    zaiko_df = pd.merge(zaiko_df, syozaikyotenchi_data[['品番','拠点所番地','受入場所']], on=['品番', '拠点所番地'], how='left')
    #! 日付列を作成
    zaiko_df['日付'] = zaiko_df['日時'].dt.date
    #! 品番_受入番号作成
    zaiko_df['品番_受入場所'] = zaiko_df['品番'].astype(str) + "_" + zaiko_df['受入場所'].astype(str)
    # product列のユニークな値を取得
    #unique_hinbans = zaiko_df['品番_受入場所'].unique()
    #st.dataframe(zaiko_df.head(10000))

    # 24時間後
    start_datetime = selected_datetime - timedelta(hours=6)
    end_datetime = selected_datetime + timedelta(days=1)
    #st.write(start_datetime,end_datetime)
    # todo
    Timestamp_df = read_syozailt_by_using_archive_data(start_datetime.strftime('%Y-%m-%d-%H'), end_datetime.strftime('%Y-%m-%d-%H'))
    # todo
    Timestamp_df['仕入先工場名'] = Timestamp_df['仕入先工場名'].apply(lambda x: '< NULL >' if pd.isna(x) else x)
    Timestamp_df = Timestamp_df.rename(columns={'仕入先工場名': '発送場所名'})# コラム名変更

    #! Activedataの統合
    # todo
    flag_DB_active = 0
    #todo 日単位のため、00にしないと、その日の納入日が入らない
    #start_date = datetime.strptime(start_datetime, '%Y-%m-%d-%H')
    start_date = start_datetime.replace(hour=0, minute=0, second=0)
    start_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    end_date = pd.to_datetime(end_datetime) + pd.Timedelta(days=1)
    active_df = read_activedata_by_using_archive_data(start_date, end_date, flag_DB_active)
    # todo
    file_path = 'temp/activedata.csv'#ステップ１,2で併用しているため、変数ではなく一時フォルダーに格納して使用
    Activedata = pd.read_csv(file_path, encoding='shift_jis')
    # 日付列をdatetime型に変換
    Activedata['日付'] = pd.to_datetime(Activedata['日付'], errors='coerce')
    Activedata['品番_受入場所'] = Activedata['品番'].astype(str) + "_" + Activedata['受入場所'].astype(str)
    Activedata['日量箱数'] = Activedata['日量数']/Activedata['収容数']
    Activedata['出庫予定かんばん数（t）'] = Activedata['日量箱数']/kado_time
    # product列のユニークな値を取得
    unique_hinbans = Activedata['品番_受入場所'].unique()
    #st.dataframe(Activedata)

    # test用
    unique_hinbans = Activedata['品番_受入場所'].unique()[:20]
    
    # 空のリストを作成
    hinban_list = []
    data_list = []

    #! ユニークな品番の組み合わせの数だけ処理を行う
    for unique_hinban in unique_hinbans:

        # 最初の _ で 2 つに分割
        part_number, seibishitsu = unique_hinban.split("_", 1)
        #st.write(part_number, seibishitsu)

        # test用
        #part_number = "9036340085"
        #seibishitsu = "1Y"

        #! ---------------------------在庫データの準備------------------------------
        #! 全データ　⇒　品番、受入場所抽出　⇒　selected_datetimeのみ抽出
        #! ------------------------------------------------------------------------
        filtered_zaiko_df = zaiko_df[(zaiko_df['品番'] == part_number) & (zaiko_df['受入場所'] == seibishitsu)]
        # 条件に一致する行を取得
        filtered_zaiko_df = filtered_zaiko_df[filtered_zaiko_df["日時"] == selected_datetime]
        #! 在庫データないならその品番はスキップ
        if len(filtered_zaiko_df) == 0:
            continue
        # 実行結果の確認
        #st.dataframe(filtered_zaiko_df)

        #! -----------------------------Activedataの準備----------------------------
        #! 全データ ⇒ 品番、整備室抽出 ⇒ 指定期間抽出
        #! -------------------------------------------------------------------------
        #! 同品番、同整備室のデータを抽出
        filtered_Activedata = Activedata[(Activedata['品番'] == part_number) & (Activedata['整備室'] == seibishitsu)]
        #st.dataframe(filtered_Activedata)
        # todo（ダブり消す、設計値違うなどでダブりがある）
        before_rows = len(filtered_Activedata)# 適用前の行数を記録
        filtered_Activedata = filtered_Activedata.drop_duplicates(subset=["日付"], keep="first")  # 最初の行を採用
        after_rows = len(filtered_Activedata)# 適用後の行数を記録
        # もし行数が変わったら、削除が機能したと判定してメッセージを出力
        # if before_rows != after_rows:
        #     st.write(f"{part_number}, {seibishitsu}重複削除が適用されました: {before_rows - after_rows} 行が削除されました。")
        # todo
        #! 1時間ごとに変換
        filtered_Activedata = filtered_Activedata.set_index('日付').resample('H').ffill().reset_index()
        filtered_Activedata = filtered_Activedata.reset_index(drop=True)
        filtered_Activedata = filtered_Activedata.rename(columns={'日付': '日時'})
        filtered_Activedata['日時'] = pd.to_datetime(filtered_Activedata['日時'])
        #st.dataframe(filtered_Activedata)
        #! 昼勤夜勤の考慮関数
        def adjust_datetime(x):
            if 0 <= x.hour < 8:
                # 日付を前日に変更し、時間はそのまま
                return x + pd.Timedelta(days=1)
            else:
                # そのままの日付を返す
                return x
        #! 昼勤夜勤の考慮
        filtered_Activedata['日時'] = filtered_Activedata['日時'].apply(adjust_datetime)
        #! 指定期間のみ抽出
        filtered_Activedata = filtered_Activedata[filtered_Activedata['日時'].isin(time_df['日時'])].copy()
        #! Activeデータないならその品番はスキップ
        if len(filtered_Activedata) == 0:
            continue
        #st.write(part_number, seibishitsu, len(filtered_Activedata))
        #st.dataframe(filtered_Activedata)

        #! ---------------------------Activeと在庫データの統合----------------------
        basedata = pd.merge(filtered_Activedata[['日時','品番_受入場所','品名','日量数','収容数','設計値MIN','設計値MAX','日量箱数','出庫予定かんばん数（t）']], filtered_zaiko_df[['日時', '品番_受入場所', '在庫数（箱）']], on=['品番_受入場所', '日時'], how='left')#! 自動ラック在庫結合
        # 実行結果の確認
        #st.dataframe(basedata)

        #! ---------------------------納入予定かんばん数の計算----------------------
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
            #todo 日単位のため、00にしないと、その日の納入日が入らない
            start_date = datetime.strptime(start_date, '%Y-%m-%d-%H')
            start_date = start_date.replace(hour=0, minute=0, second=0)
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            #st.write(start_date,end_date)

            # ① "納入日"列が期間内に該当する行を抽出
            filtered_df = df[(pd.to_datetime(df['納入日']) >= start_date) & (pd.to_datetime(df['納入日']) < end_date)]
            #st.dataframe(filtered_df)

            if len(filtered_df) != 0:

                #st.header("定刻便確認")
                #st.dataframe(filtered_df)

                # ② 抽出したデータに対して処理
                # ②-1 "納入便"列から数値を取得
                filtered_df['B'] = filtered_df['納入便'].astype(int)

                # ②-2 "B便_定刻"列の値を取得して新しい列"納入予定時間"に格納
                filtered_df['納入予定時間'] = filtered_df.apply(lambda row: row[f"{row['B']}便_定刻"] if f"{row['B']}便_定刻" in df.columns else None, axis=1)

                # ②-3 "納入予定時間"列が0時～8時の場合に"納入日_補正"列を1日後に設定
                filtered_df['納入予定時間'] = pd.to_datetime(filtered_df['納入予定時間'], format='%H:%M:%S', errors='coerce').dt.time
                #st.dataframe(filtered_df)
                #todo 夜勤便は+1が必要！！！！
                #todo 今の計算でいいか不明！！\
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

            else:
                nonyu_yotei_df = pd.DataFrame(columns=["日時", "納入予定かんばん数（t）", "納入予定便一覧",
                                                       "納入予定かんばん一覧","納入予定便","納入予定日時_raw"])
                kensyu_df = pd.DataFrame(columns=["日時", "検収日時_raw", "検収かんばん数",
                                                       "検収かんばん一覧"])

            return nonyu_yotei_df, kensyu_df
        
        #! 所在管理MBのテーブルデータ
        #! 同品番、同整備室のデータを抽出
        filtered_Timestamp_df = Timestamp_df[(Timestamp_df['品番'] == part_number) & (Timestamp_df['整備室コード'] == seibishitsu)]
        #! 仕入先名、仕入先工場名抽出
        unique_shiresaki = filtered_Timestamp_df['仕入先名'].unique()[0]
        unique_shiresaki_kojo = filtered_Timestamp_df['発送場所名'].unique()[0]
        #st.write(unique_shiresaki,unique_shiresaki_kojo)
        #st.dataframe(filtered_Timestamp_df)
        #! 仕入先便情報抽出
        arrival_times_df = calculate_supplier_truck_arrival_types2()
        #! 一致する仕入れ先フラグが見つからない場合、エラーを出す
        #! 3つの列（仕入先名、発送場所名、整備室コード）で条件を満たす行をarrival_times_dfから抽出し、新しいデータフレームmatched_arrival_times_dfを作成
        # 条件は、lagged_featuresと同じ仕入先名、発送場所名、整備室コードを持つもの
        matched_arrival_times_df = arrival_times_df[
            (arrival_times_df['仕入先名'].isin([unique_shiresaki])) &
            (arrival_times_df['発送場所名'].isin([unique_shiresaki_kojo])) &
            (arrival_times_df['受入'].isin([seibishitsu]))
        ]
        matched_arrival_times_df = matched_arrival_times_df.rename(columns={'受入': '整備室コード'})# コラム名変更
        #st.dataframe(matched_arrival_times_df)
        # 統合する列の選別
        columns_to_extract_t = ['かんばんシリアル','品名','納入日', '納入便','検収日時','仕入先名', '発送場所名', '整備室コード']
        columns_to_extract_l = matched_arrival_times_df.filter(regex='便_定刻').columns.tolist() + ['仕入先名', '発送場所名', '整備室コード','納入先']
        # 統合
        filtered_Timestamp_df = pd.merge(filtered_Timestamp_df[columns_to_extract_t], matched_arrival_times_df[columns_to_extract_l], on=['仕入先名', '発送場所名', '整備室コード'], how='left')
        #st.dataframe(filtered_Timestamp_df)
        #! 納入タイプ抽出
        unique_nonyu_type = filtered_Timestamp_df['納入先'].unique()[0]
        #st.write(unique_nonyu_type)
        #! 納入予定かんばん数（t）の計算
        #st.write(start_datetime,end_datetime)
        nonyu_yotei_df, kensyu_df = calculate_scheduled_nouyu_kanban(filtered_Timestamp_df, start_datetime.strftime('%Y-%m-%d-%H'), end_datetime.strftime('%Y-%m-%d-%H'))
        #st.dataframe(nonyu_yotei_df)
        #st.write(part_number, seibishitsu, nonyu_yotei_df["納入予定かんばん数（t）"].mean(), filtered_Activedata['日量箱数'].mean()/filtered_Activedata['サイクル回数'].mean(), before_rows - after_rows)

        # 1時間ごとの時間列（24時間分）を作成
        time_series = pd.date_range(start=start_datetime, periods=24+5, freq="H")
        # データフレームを作成
        nonyu_data_df = pd.DataFrame({"日時": time_series})
        #st.dataframe(nonyu_data_df)

        #! 日時でデータフレームを結合
        nonyu_data_df = pd.merge(nonyu_data_df, nonyu_yotei_df, on='日時', how='left')
        nonyu_data_df = pd.merge(nonyu_data_df, kensyu_df, on='日時', how='left')
        #! すべてのNone値を0に置き換え
        # basedataに統合する際、nonyu_yotei_dfに存在しない日時はNoneになるため
        nonyu_data_df = nonyu_data_df.fillna(0)
        #st.dataframe(nonyu_data_df)

        if unique_nonyu_type == "西尾東":
            nonyu_lt = 5
        else:
            nonyu_lt = 1

        # nonyu_lt時間後にシフト
        # 昇順に並び替え
        nonyu_data_df = nonyu_data_df.sort_values(by="日時")
        nonyu_data_df["入庫予定かんばん数（t）"] = nonyu_data_df["納入予定かんばん数（t）"].shift(nonyu_lt)
        #st.dataframe(nonyu_data_df)

        #! ---------------------------------------納入予定かんばん数データの統合--------------------------------------------------------
        #! 日時でデータフレームを結合
        basedata = basedata.sort_values(by="日時")
        basedata = basedata.fillna(0)
        basedata = pd.merge(basedata, nonyu_data_df, on='日時', how='left')
        #st.dataframe(basedata)

        #! 在庫シミュレーション
        # 在庫数を計算（累積計算）
        basedata["在庫数（箱）_予測値"]=basedata["在庫数（箱）"]
        for i in range(1, len(basedata)):
            basedata.loc[i, "在庫数（箱）_予測値"] = (
                basedata.loc[i - 1, "在庫数（箱）_予測値"]  # 1つ上の行の在庫数
                + basedata.loc[i, "入庫予定かんばん数（t）"]  # 納入分を加算
                - basedata.loc[i, "出庫予定かんばん数（t）"]  # 出庫分を減算
            )
        #st.dataframe(basedata)

        #! 判定
        basedata["下限割れ"] = (basedata["在庫数（箱）_予測値"] < basedata["設計値MIN"]).astype(int)
        basedata["上限越え"] = (basedata["在庫数（箱）_予測値"] > basedata["設計値MAX"]).astype(int)
        basedata["在庫0"] = (basedata["在庫数（箱）_予測値"] < 0).astype(int)

        #! 各項目の合計を計算
        total_lower_limit = basedata["下限割れ"].sum()
        total_upper_exceed = basedata["上限越え"].sum()
        total_stock_zero = basedata["在庫0"].sum()
        # 条件分岐でOK/NGに変換
        total_lower_limit = "NG" if total_lower_limit > 0 else "OK"
        total_upper_exceed = "NG" if total_upper_exceed > 0 else "OK"
        total_stock_zero = "NG" if total_stock_zero > 0 else "OK"

        #st.dataframe(basedata)

        # ---- 必要な列を抽出 ----
        basedata_filtered = basedata[["日時", "在庫数（箱）_予測値", "設計値MIN", "設計値MAX"]]

        # Matplotlibでプロット作成
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(basedata_filtered["日時"], basedata_filtered["在庫数（箱）_予測値"], label="在庫数（箱）_予測値", marker="o")
        ax.fill_between(basedata_filtered["日時"], basedata_filtered["設計値MIN"], basedata_filtered["設計値MAX"], 
                        color="lightgray", alpha=0.5, label="設計値範囲 (MIN-MAX)")
        #これはいらないかも
        #ax.axhline(y=basedata_filtered["設計値MIN"].iloc[0], color="blue", linestyle="--", label="設計値MIN")
        #ax.axhline(y=basedata_filtered["設計値MAX"].iloc[0], color="red", linestyle="--", label="設計値MAX")

        # ---- グラフの装飾 ----
        ax.set_title("在庫数と設計値の比較", fontsize=14)
        ax.set_xlabel("日時", fontsize=12)
        ax.set_ylabel("在庫数", fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        # 画面に表示
        #st.pyplot(fig)

        # 日時列をdatetime型に変換
        basedata_filtered['日時'] = pd.to_datetime(basedata_filtered['日時'])

        # 一番古い時刻を基準時刻（現在時刻）とする
        base_time = basedata_filtered['日時'].min()

        # 基準時刻からの経過時間 (時間単位) を計算
        basedata_filtered['経過時間(時間)'] = (basedata_filtered['日時'] - base_time).dt.total_seconds() / 3600

        # 設計値MINを割る最初の時間
        time_min = basedata_filtered.loc[basedata_filtered['在庫数（箱）_予測値'] < basedata_filtered['設計値MIN'], '経過時間(時間)'].min()

        # 設計値MAXを割る最初の時間
        time_max = basedata_filtered.loc[basedata_filtered['在庫数（箱）_予測値'] > basedata_filtered['設計値MAX'], '経過時間(時間)'].min()

        # 在庫が0より小さくなる最初の時間
        time_zero = basedata_filtered.loc[basedata_filtered['在庫数（箱）_予測値'] < 0, '経過時間(時間)'].min()

        # st.dataframe(basedata_filtered)
        # st.write(time_min,time_max,time_zero)

        # ---- PNGファイルとして保存 ----
        save_dir = "temp/在庫シミュレーション"
        os.makedirs(save_dir, exist_ok=True)
        output_file = f"{save_dir}/{unique_hinban}.png"
        fig.savefig(output_file, format="png", dpi=300, bbox_inches="tight")

        #! 必要データだけ準備
        hinban_list.append(output_file)
        unique_hinmei = filtered_Timestamp_df['品名'].unique()[0]
        data_list.append({"品番_整備室": unique_hinban, "品名": unique_hinmei,
                           "仕入先名": unique_shiresaki, "発送工場名": unique_shiresaki_kojo,
                           "下限割れ":total_lower_limit,"上限越え":total_upper_exceed,"欠品":total_stock_zero,
                           "下限割れまでの時間":time_min,"上限越えまでの時間":time_max,"欠品までの時間":time_zero})

    # ローカルの PNG ファイルを Base64 エンコードする関数
    def img_to_base64(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        # Base64 エンコードして文字列に変換
        return base64.b64encode(data).decode("utf-8")

    # DataFrame に変換
    df_A = pd.DataFrame(data_list)
    # 画像を Base64 変換
    base64_images = [img_to_base64(p) for p in hinban_list]
    # DataFrame に変換
    df_B = pd.DataFrame(base64_images, columns=["画像base64"])

    #edited_df = st.data_editor(df_A, num_rows="dynamic")

    # DataFrame を統合（横方向に結合）
    data = pd.concat([df_A, df_B], axis=1)

    df = pd.DataFrame(data)

    #st.dataframe(df)

    #import csv

    st.divider()

    # 最後の列を除く
    df_excluded_last = df.iloc[:, :-1]

    # CSV文字列に変換（Shift_JISでエンコード）
    csv_data = df_excluded_last.to_csv(index=False, encoding='cp932')#, quoting=csv.QUOTE_ALL)

    # CSVをバイナリに変換 → Base64エンコード
    b64_encoded = base64.b64encode(csv_data.encode('cp932')).decode()

    # data URI スキームの文字列を作成
    # Shift_JIS で解釈されるよう charset=shift_jis も付与
    csv_uri = f"data:text/csv;charset=shift_jis;base64,{b64_encoded}"

    # カスタムHTMLボタン（例）
    custom_button = f"""
        <a download="在庫予測結果.csv" href="{csv_uri}">
            <button style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
            ">
            📥 結果をダウンロードする
            </button>
        </a>
    """

    st.markdown(custom_button, unsafe_allow_html=True)

    def style_ok_ng(value):
        """OK/NG文字列を色付きバッジHTMLに変換"""
        if value == "OK":
            return """<span class="badge-ok">OK</span>"""
        elif value == "NG":
            return """<span class="badge-ng">NG</span>"""
        else:
            return str(value)

    #HTML 組み立て
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>DataTables with Default Filters and Sorting</title>

        <!-- ▼ DataTables用CSS (CDN) ▼ -->
        <link rel="stylesheet" type="text/css"
            href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.css"/>
        <script type="text/javascript"
                src="https://cdn.jsdelivr.net/npm/jquery@3.6.4/dist/jquery.min.js"></script>
        <script type="text/javascript"
                src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.js"></script>

        <style>
        /* テーブルの基本スタイル */
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
        }

        /* OK/NGバッジのスタイル */
        .badge-ok {
            display: inline-block;
            padding: 5px 10px;
            color: #fff;
            background-color: #00aaff;
            border-radius: 5px;
        }
        .badge-ng {
            display: inline-block;
            padding: 5px 10px;
            color: #fff;
            background-color: #ff4444;
            border-radius: 5px;
        }

        /* トグルボタン */
        .toggle-button {
            padding: 8px 16px;
            font-size: 16px;
            background-color: #008CBA;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .toggle-button:hover {
            background-color: #006F9A;
        }

        /* トグル表示部分（画像）は初期非表示 */
        .hidden-content {
            display: none;
            margin-top: 8px;
        }

        /* ソートセレクトボックスのスタイル */
        .filter-select {
            margin: 10px 0;
            padding: 6px 12px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        </style>

        <script>
        $(document).ready(function() {
            // DataTables の初期化
            var table = $('#myTable').DataTable({
                paging: true,
                searching: true,
                ordering: true,
                pageLength: 10,
                lengthMenu: [10, 20, 30],
                order: [[5, 'asc']],  // デフォルトで「欠品までの時間」を降順にソート
                columnDefs: [
                    {
                        targets: [7, 9, 5], // 数値列（下限割れ、上限越え、欠品までの時間）
                        render: function(data, type, row) {
                            if (type === 'sort' || type === 'type') {
                                // NaN を -Infinity に変換してソート可能に
                                return data === null || data === '' ? Infinity : parseFloat(data);
                            }
                            return data; // 表示時はそのまま
                        }
                    }
                ]
            });

            // デフォルトフィルタ適用
            table.column(4).search('NG').draw();  // 欠品列を NG でフィルタ
            table.column(6).search('NG').draw();  // 下限割れ列を NG でフィルタ
            table.column(8).search('').draw();   // 上限越え列はフィルタなし（すべて）

            // 列単位フィルタの処理
            function doFilter() {
                var valKekin = $('#filter_kekin').val();
                var valKagen = $('#filter_kagen').val();
                var valJougen = $('#filter_jougen').val();
                table.column(4).search(valKekin).draw();
                table.column(6).search(valKagen).draw();
                table.column(8).search(valJougen).draw();
            }

            // フィルタが変更されたら doFilter を実行
            $('#filter_kekin').on('change', doFilter);
            $('#filter_kagen').on('change', doFilter);
            $('#filter_jougen').on('change', doFilter);

            // ソート条件の変更イベント
            $('#sort-order').on('change', function() {
                var columnIndex = $(this).val();  // セレクトボックスの値を取得
                table.order([columnIndex, 'asc']).draw();  // 指定列で降順ソート
            });
        });

        // トグル機能の実装
        function toggleImage(id) {
            var elem = document.getElementById(id);
            if (elem.style.display === 'none' || elem.style.display === '') {
                elem.style.display = 'block';
            } else {
                elem.style.display = 'none';
            }
        }
        </script>
    </head>
    <body>
        <!-- ▼ 列単位フィルタUI: 欠品, 下限割れ, 上限越え ▼ -->
        <div class="filter-boxes">
            <label>欠品:
                <select id="filter_kekin" class="filter-select">
                    <option value="">(すべて)</option>
                    <option value="OK">OKのみ</option>
                    <option value="NG" selected>NGのみ</option>
                </select>
            </label>

            <label>下限割れ:
                <select id="filter_kagen" class="filter-select">
                    <option value="">(すべて)</option>
                    <option value="OK">OKのみ</option>
                    <option value="NG" selected>NGのみ</option>
                </select>
            </label>

            <label>上限越え:
                <select id="filter_jougen" class="filter-select">
                    <option value="" selected>(すべて)</option>
                    <option value="OK">OKのみ</option>
                    <option value="NG">NGのみ</option>
                </select>
            </label>
        </div>

        <!-- ▼ ソート用セレクトボックスを追加 -->
        <div>
            <label for="sort-order">並び替え条件:</label>
            <select id="sort-order" class="filter-select">
                <option value="6">下限割れまでの時間（大きい順）</option>
                <option value="8">上限越えまでの時間（大きい順）</option>
                <option value="4" selected>欠品までの時間（大きい順）</option>
            </select>
        </div>

        <!-- ▼ DataTables 対応テーブル -->
        <table id="myTable" class="display">
            <thead>
                <tr>
                    <th>品番_整備室</th>
                    <th>品名</th>
                    <th>仕入先名</th>
                    <th>仕入先工場名</th>
                    <th>欠品</th>
                    <th>欠品までの時間</th>
                    <th>下限割れ</th>
                    <th>下限割れまでの時間</th>
                    <th>上限越え</th>
                    <th>上限越えまでの時間</th>
                    <th>グラフ</th>
                </tr>
            </thead>
            <tbody>
    """

    # DataFrame の行をループして HTML に変換
    # for i, row in df.iterrows():
    #     html_code += f"""
    #     <tr>
    #         <td>{row['品番_整備室']}</td>
    #         <td>{row['品名']}</td>
    #         <td>{row['仕入先名']}</td>
    #         <td>{row['発送工場名']}</td>
    #         <td>{row['下限割れまでの時間']}</td>
    #         <td>{row['上限越えまでの時間']}</td>
    #         <td>{row['欠品までの時間']}</td>
    #         <td>{style_ok_ng(row['欠品'])}</td>
    #         <td>{style_ok_ng(row['下限割れ'])}</td>
    #         <td>{style_ok_ng(row['上限越え'])}</td>
    #         <td>
    #             <button class="toggle-button" onclick="toggleImage('hidden-content-{i}')">表示</button>
    #             <div id="hidden-content-{i}" class="hidden-content">
    #                 <img src="data:image/png;base64,{row['画像base64']}" style="max-width: 200px;">
    #             </div>
    #         </td>
    #     </tr>
    #     """
    for i, row in df.iterrows():
        html_code += f"""
        <tr>
            <td>{row['品番_整備室']}</td>
            <td>{row['品名']}</td>
            <td>{row['仕入先名']}</td>
            <td>{row['発送工場名']}</td>
            <td>{style_ok_ng(row['欠品'])}</td>
            <td>{row['欠品までの時間']}</td>
            <td>{style_ok_ng(row['下限割れ'])}</td>
             <td>{row['下限割れまでの時間']}</td>
            <td>{style_ok_ng(row['上限越え'])}</td>
            <td>{row['上限越えまでの時間']}</td>
            <td>
                <button class="toggle-button" onclick="toggleImage('hidden-content-{i}')">表示</button>
                <div id="hidden-content-{i}" class="hidden-content">
                    <img src="data:image/png;base64,{row['画像base64']}" style="max-width: 200px;">
                </div>
            </td>
        </tr>
        """

    html_code += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # 4) Streamlit で表示
    components.html(html_code, height=1000, scrolling=True)

    #st.dataframe(df)

    return df_excluded_last


#! 在庫予測
def show_forecast2( unique_product, start_datetime, selected_zaiko):

    start_date = '2024-05-01-00'
    end_date = '2024-08-31-00'

    #! 品番、整備室コードを抽出
    product = unique_product.split('_')[0]
    seibishitsu = unique_product.split('_')[1]

    #! パラメータ設定
    prediction_hours = 24#何時間先まで予測するのか
    past_hours = 5
    lookback_hours = past_hours+2

    # タイトル表示
    st.header('予測結果')

    #!----------------------------------------------------------------------- 
    #! 自動ラックの在庫データの読み込みと処理
    #!-----------------------------------------------------------------------
    zaiko_df = read_zaiko_by_using_archive_data(start_date, end_date)
    # 品番列の空白を削除
    zaiko_df['品番'] = zaiko_df['品番'].str.strip()
    # '計測日時'をdatetime型に変換
    #zaiko_df['計測日時'] = pd.to_datetime(zaiko_df['計測日時'], errors='coerce')
    # 列名 '計測日時' を '日時' に変更
    #zaiko_df = zaiko_df.rename(columns={'計測日時': '日時'})
    # 特定の品番の商品データを抽出
    zaiko_df = zaiko_df[zaiko_df['品番'] == product]
    # 特定の日時のデータを抽出
    zaiko_df = zaiko_df[zaiko_df['日時'] == start_datetime]
    # 日時を再度datetime型に変換（念のため）
    zaiko_df['日時'] = pd.to_datetime(zaiko_df['日時'])
    # '日時' と '在庫数（箱）' の列のみを抽出
    zaiko_extracted = zaiko_df[['日時', '在庫数（箱）']]

    #!-----------------------------------------------------------------------
    #! 所在管理リードタイムのデータ
    #!-----------------------------------------------------------------------
    #file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
    Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
    # '更新日時'列に無効な日時データがある行を削除する
    data_cleaned = Timestamp_df.dropna(subset=['検収日時'])
    st.dataframe(data_cleaned.head(50000))
    # 特定の品番の商品データを抽出
    data_cleaned = data_cleaned[(data_cleaned['品番'] == product) & (data_cleaned['整備室コード'] == seibishitsu)]
    # 時間ごとにグループ化し、各時間でのかんばん数をカウントする
    data_cleaned['日時'] = data_cleaned['検収日時'].dt.floor('H')  # 時間単位に丸める
    hourly_kanban_count = data_cleaned.groupby('日時').size().reset_index(name='納入予定かんばん数')
    #st.dataframe(hourly_kanban_count)

    # 時間の範囲を決定し、欠損時間帯を補完する
    full_time_range = pd.date_range(start=hourly_kanban_count['日時'].min(),end=hourly_kanban_count['日時'].max(),freq='H')

    # 全ての時間を含むデータフレームを作成し、欠損値を0で埋める
    hourly_kanban_count_full = pd.DataFrame(full_time_range, columns=['日時']).merge(hourly_kanban_count, on='日時', how='left').fillna(0)

    # かんばん数を整数に戻す
    hourly_kanban_count_full['納入予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].astype(int)

    # '予測入庫時間'列として、5時間前のかんばん数を追加する
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['納入予定かんばん数'].shift(past_hours)

    # 欠損値（最初の5時間分）を0で埋める
    hourly_kanban_count_full['工場到着予定かんばん数'] = hourly_kanban_count_full['工場到着予定かんばん数'].fillna(0).astype(int)

    #!-----------------------------------------------------------------------
    #! Activedataの処理
    #!-----------------------------------------------------------------------
    activedata = read_activedata_by_using_archive_data(start_date, end_date, 0)
    # 特定の品番の商品データを抽出
    activedata = activedata[activedata['品番'] == product]
    #st.dataframe(activedata)
    #! 稼働時間で割る処理 (休憩時間の考慮が必要か？)
    activedata['日量数（箱数）'] = activedata['日量数']/activedata['収容数']
    activedata['日量数（箱数）/稼働時間'] = activedata['日量数（箱数）'] / 16.5
    activedata['日付'] = pd.to_datetime(activedata['日付'])#これしないと次の.resample('H')でエラーが出る
    # 日付を基準に1時間ごとのデータに変換
    activedata = activedata.set_index('日付').resample('H').ffill().reset_index()
    # '日付' をdatetime型に変換
    activedata['日付'] = pd.to_datetime(activedata['日付'])
    activedata = activedata.rename(columns={'日付': '日時'})
    # '日付' と '日量数（箱数）' の列のみを抽出
    activedata_extracted = activedata[['日時', '日量数（箱数）/稼働時間']]

    # 在庫データの開始時刻を取得
    start_time = zaiko_extracted.iloc[0]['日時']
    # 開始時刻から20時間後までのデータを抽出
    end_time = start_time + pd.Timedelta(hours=prediction_hours)
    filtered_activedata = activedata_extracted[(activedata_extracted['日時'] >= start_time) & (activedata_extracted['日時'] < end_time)]

    # 各時間後の消費量および入庫量を考慮した在庫数を計算
    inventory_after_adjustments = []
    # 現在の在庫数を初期値として設定
    current_inventory = zaiko_extracted.iloc[0]['在庫数（箱）']

    # 3つの列を作成
    col1, col2 = st.columns(2)
    col1.metric(label="選択された日時", value=str(start_datetime))#, delta="1 mph")
    col2.metric(label="入力された組立ラインの在庫数（箱）", value=int(current_inventory))

    # 時間ごとの在庫数を更新しながらリストに追加
    for i, row in filtered_activedata.iterrows():
        kanban_row = hourly_kanban_count_full[hourly_kanban_count_full['日時'] == row['日時']]
        incoming_kanban = kanban_row['工場到着予定かんばん数'].values[0] if not kanban_row.empty else 0
        inventory_after_adjustments.append({
            '日時': row['日時'],
            '在庫数（箱）': current_inventory
        })
        # 最初のタイムスタンプでは消費を引かないが、以降は消費量と入庫量を調整
        if i != 0:
            current_inventory = current_inventory - row['日量数（箱数）/稼働時間']  # 消費量を引く
            current_inventory = current_inventory + incoming_kanban  # 入庫量を足す
            

    # 計算結果をDataFrameに変換
    inventory_df_adjusted = pd.DataFrame(inventory_after_adjustments)

    # 最初の時間のデータ（実際のデータ）とそれ以降の予測データに分割
    actual_data = inventory_df_adjusted.iloc[0:1]  # 最初の1時間分は実際のデータ
    forecast_data = inventory_df_adjusted.iloc[1:]  # それ以降は予測データ

    # 時間軸を統一するため、全時間の範囲を作成
    #full_time_range = pd.date_range(start=actual_data['日時'].min(), end=forecast_data['日時'].max(), freq='H')

    # データフレームをそれぞれこの時間軸に合わせて再構築し、欠損値を埋める
    #actual_data = actual_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})
    #forecast_data = forecast_data.set_index('日時').reindex(full_time_range).reset_index().rename(columns={'index': '日時'})

    # 欠損値はそれぞれ0に置き換える（必要に応じて）
    #actual_data['在庫数（箱）'].fillna(0, inplace=True)
    #forecast_data['在庫数（箱）'].fillna(0, inplace=True)

    # グラフの作成
    fig = go.Figure()

    # 実際のデータを青色で描画
    fig.add_trace(go.Bar(
        x=actual_data['日時'], 
        y=actual_data['在庫数（箱）'], 
        name='実績', 
        marker_color='blue', 
        opacity=0.3
    ))

    # 予測データをオレンジ色で追加描画
    fig.add_trace(go.Bar(
        x=forecast_data['日時'], 
        y=forecast_data['在庫数（箱）'], 
        name='予測', 
        marker_color='orange', 
        opacity=0.3
    ))

    # x軸を1時間ごとに表示する設定
    fig.update_layout(
        title='予測結果',  # ここでタイトルを設定
        xaxis_title='日時',  # x軸タイトル
        yaxis_title='在庫数（箱）',  # y軸タイトル
        xaxis=dict(
            tickformat="%Y-%m-%d %H:%M",  # 日時のフォーマットを指定
            dtick=3600000  # 1時間ごとに表示 (3600000ミリ秒 = 1時間)
        ),
        barmode='group'  # 複数のバーをグループ化
    )

    # グラフをStreamlitで表示
    st.plotly_chart(fig)

    # 5時間前の日時を計算
    hours_before = start_time - pd.Timedelta(hours=lookback_hours)

    # ユーザーに結果を表示する
    hourly_kanban_count_full = hourly_kanban_count_full[(hourly_kanban_count_full['日時'] >= hours_before) & (hourly_kanban_count_full['日時'] < end_time)]

    # 新しい列「備考」を追加し、start_timeに基づいて「過去」「未来」と表示
    hourly_kanban_count_full['※注釈                                                                               '] = hourly_kanban_count_full['日時'].apply(
        lambda x: 'あなたはこの時間を選択しました' if x == start_time else ('過去' if x < start_time else '未来')
    )

    # '日時'列でstart_timeに一致する行をハイライト
    def highlight_start_time(row):
        return ['background-color: yellow' if row['日時'] == start_time else '' for _ in row]
    
    st.code(f"📝 計算式：未来の在庫数 = 在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

    # 注釈を追加（例としてstart_timeを表示）
    st.markdown(f"")
    st.markdown(f"")
    st.markdown(f"**下の表で予測の内容を確認できます。**")
    #st.code(f"計算式：在庫数 + 工場到着予定かんばん数 - 日量箱数/稼働時間")

    # 'hourly_kanban_count_full' と 'inventory_df_adjusted' を '日時' をキーに結合
    merged_df = pd.merge(hourly_kanban_count_full, inventory_df_adjusted, on='日時', how='outer')
    activedata_extracted = activedata_extracted[(activedata_extracted['日時'] >= hours_before) & (activedata_extracted['日時'] < end_time)]
    merged_df = pd.merge(merged_df, activedata_extracted, on='日時', how='outer')

    # 必要に応じてNaNを0に置き換える（在庫数やかんばん数に関して）
    merged_df.fillna(0, inplace=True)

    # Streamlitで表示
    # データフレームの列の順番を指定
    new_column_order = ['日時', '納入予定かんばん数', '工場到着予定かんばん数', '日量数（箱数）/稼働時間', '在庫数（箱）','※注釈                                                                               ']
    # 列の順番を変更
    merged_df = merged_df[new_column_order]

    # 条件に該当する行の在庫数を "-" にする
    merged_df.loc[
        (merged_df['日時'] >= hours_before) & 
        (merged_df['日時'] < start_time), 
        '在庫数（箱）'
    ] = "-"

    # '日時'列でstart_timeに一致する行をハイライトして表示
    st.dataframe(merged_df.style.apply(highlight_start_time, axis=1))








