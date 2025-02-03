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

#! 在庫シミュレーション
def show_zaiko_simulation( selected_datetime, change_rate):

    #! 日量箱数を時間単位にするために
    #todo 稼働時間などを考えるなら、16.5で割る必要があるかもだが、その場合はどの時間帯が稼働時間分かる必要がある
    kado_time = 24

    #! 選択情報表示
    col1, col2 = st.columns(2)
    col1.metric(label="選択日時", value=selected_datetime.strftime("%Y-%m-%d %H:%M"))
    col2.metric(label="選択変動率", value=change_rate)

    # 1時間ごとの時間列（24時間分）を作成
    time_series = pd.date_range(start=selected_datetime, periods=24, freq="H")
    # データフレームを作成
    time_df = pd.DataFrame({"日時": time_series})
    #st.dataframe(time_df)

    #! 自動ラックの在庫データを読み込み
    # todo 引数関係なく全データ読み込みしてる
    zaiko_df = read_zaiko_by_using_archive_data(selected_datetime.strftime('%Y-%m-%d-%H'), selected_datetime.strftime('%Y-%m-%d-%H'))
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
    Timestamp_df = read_syozailt_by_using_archive_data(start_datetime.strftime('%Y-%m-%d-%H'), end_datetime.strftime('%Y-%m-%d-%H'))
    Timestamp_df['仕入先工場名'] = Timestamp_df['仕入先工場名'].apply(lambda x: '< NULL >' if pd.isna(x) else x)
    Timestamp_df = Timestamp_df.rename(columns={'仕入先工場名': '発送場所名'})# コラム名変更

    #! Activedataの統合
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
    #unique_hinbans = Activedata['品番_受入場所'].unique()[:20]\
    
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
        ax.axhline(y=basedata_filtered["設計値MIN"].iloc[0], color="blue", linestyle="--", label="設計値MIN")
        ax.axhline(y=basedata_filtered["設計値MAX"].iloc[0], color="red", linestyle="--", label="設計値MAX")

        # ---- グラフの装飾 ----
        ax.set_title("在庫数と設計値の比較", fontsize=14)
        ax.set_xlabel("日時", fontsize=12)
        ax.set_ylabel("在庫数", fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        # 画面に表示
        #st.pyplot(fig)

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
                           "下限割れ":total_lower_limit,"上限越え":total_upper_exceed,"在庫0":total_stock_zero})

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

    # DataFrame を統合（横方向に結合）
    data = pd.concat([df_A, df_B], axis=1)

    df = pd.DataFrame(data)

    #st.dataframe(df)

    # ---- HTMLを組み立てる ----
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 1em;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f7f7f7;
            }
            /* 折りたたまれている要素を非表示 */
            .hidden-content {
                display: none;
            }
            .toggle-button {
                padding: 6px 12px;
                background-color: #008CBA;
                color: white;
                border: none;
                cursor: pointer;
                border-radius: 4px;
            }
            .toggle-button:hover {
                background-color: #006F9A;
            }
        </style>
    </head>
    <body>

    <script>
        // ボタン押下時に表示・非表示を切り替える関数
        function toggleImage(id) {
            var elem = document.getElementById(id);
            if (elem.style.display === 'none' || elem.style.display === '') {
                elem.style.display = 'block';
            } else {
                elem.style.display = 'none';
            }
        }
    </script>

    <table>
        <thead>
            <tr>
                <th>品番_整備室</th>
                <th>品名</th>
                <th>仕入先名</th>
                <th>仕入先工場名</th>
                <th>下限割れ</th>
                <th>在庫0</th>
                <th>上限越え</th>
                <th>グラフ</th>
            </tr>
        </thead>
        <tbody>
    """

    # DataFrameの各行をループしてテーブルHTML作成
    for i, row in df.iterrows():

        name = row["品番_整備室"]
        price = row["品名"]
        stock = row["仕入先名"]
        stock1 = row["発送工場名"]
        stock2 = row["下限割れ"]
        stock3 = row["在庫0"]
        stock4 = row["上限越え"]
        img_b64 = row["画像base64"]
        
        # PNGの場合 => data:image/png;base64, ...
        data_url = f"data:image/png;base64,{img_b64}"
        
        html_code += f"""
        <tr>
        <td>{name}</td>
        <td>{price}</td>
        <td>{stock}</td>
        <td>{stock1}</td>
        <td>{stock2}</td>
        <td>{stock3}</td>
        <td>{stock4}</td>
        <td>
            <!-- onclick で toggleImage() を呼び出し、ID指定で要素を表示/非表示 -->
            <button class="toggle-button" onclick="toggleImage('hidden-content-{i}')">表示</button>
            <!-- 最初は hidden-content クラスで非表示状態 -->
            <div id="hidden-content-{i}" class="hidden-content">
                <img src="{data_url}" style="max-width: 200px; margin-top: 8px;">
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

    # Streamlit で HTML を描画 (高さやスクロールは必要に応じて変更)
    components.html(html_code, height=600, scrolling=True)


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








