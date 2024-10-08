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

#! 自作ライブラリのimport
from read_v3 import read_data, process_Activedata, read_syozailt_by_using_archive_data, read_activedata_by_using_archive_data,read_zaiko__by_using_archive_data

def show_forecast( unique_product, start_datetime, selected_zaiko):

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
    zaiko_df = read_zaiko__by_using_archive_data(start_date, end_date)
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
    # 時間ごとにグループ化し、各時間でのかんばん数をカウントする
    data_cleaned['日時'] = data_cleaned['検収日時'].dt.floor('H')  # 時間単位に丸める
    hourly_kanban_count = data_cleaned.groupby('日時').size().reset_index(name='納入予定かんばん数')

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
    activedata = read_activedata_by_using_archive_data(start_date, end_date,0)
    # 特定の品番の商品データを抽出
    activedata = activedata[activedata['品番'] == product]
    st.dataframe(activedata)
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







