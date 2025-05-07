import pandas as pd
import datetime as dt
import numpy as np
import streamlit as st
from pathlib import Path
import json
import pyodbc
import glob
import os
os.add_dll_directory('C:/Program Files/IBM/IBM DATA SERVER DRIVER/bin')
import ibm_db
from dateutil.relativedelta import relativedelta
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
import warnings
import shutil
from pathlib import Path
import filecmp
from scipy.stats import gaussian_kde

#! 設定用ファイル.jsonのパス定義
# ★相対パスで読み込み（EXE時不可）
# CONFIG_PATH = '../../configs/settings.json'
# ★絶対パスで読み込み
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 設定ファイルの絶対パスを作成
CONFIG_PATH = os.path.join(current_dir, "..", "..", "configs", "settings.json")

#! 中間成果物確認用ディレクトリパス定義
# ★相対パスで読み込み（EXE時不可）
# TEMP_OUTPUTS_PATH = 'outputs/temp'
# ★絶対パスで読み込み
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 設定ファイルの絶対パスを作成
TEMP_OUTPUTS_PATH = os.path.join(current_dir, "outputs", "temp")

#! 最終成果物ディレクトリー
# ★相対パスで読み込み（EXE時不可）
#FINAL_OUTPUTS_PATH = 'outputs'
# ★絶対パスで読み込み
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 目的のファイルの絶対パスを作成
FINAL_OUTPUTS_PATH = os.path.join(current_dir,"outputs")

#! 手配データファイルのパス定義
# 日本語があるため、UTF-8でファイルを読み込む
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)
#　対象工場の読み込み
selected_data = config['selected_data']
selecte_kojo = selected_data["kojo"]
# 仕入先ダイヤファイルの読み込み
active_data_paths= config['active_data_path']
# ファイル名格納
#TEHAI_DATA_NAME = '手配必要数&手配運用情報テーブル.csv'#old
TEHAI_DATA_NAME =  active_data_paths[selecte_kojo]
# 手配系データのフルパス
TEHAI_DATA_PATH = os.path.join( FINAL_OUTPUTS_PATH, TEHAI_DATA_NAME)

#! 生物着工データのパス定義
# old
# SEIBUTSU_BASE_PATH = "C:\\Users\\1082794-Z100\\Documents\\Model\\zaiko\\生データ\\202501-02_P8 1"
# ★絶対パスで読み込み
# 現在のファイルの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 目的のファイルの絶対パスを作成
SEIBUTSU_BASE_PATH = os.path.join(current_dir, "..", "..", "data", "生産物流システム")
# 生物最終実行日
SEIBUTSU_ZIKKOU_TIME = os.path.join(current_dir, "..", "..", "data", "生産物流システム","最終実行日.txt")

# MARK:データ読み込み関数定義 ----------------------------------------------------------------------------------------

# MARK: 仕入先ダイヤを読み込む、不等ピッチ等を計算する
@st.cache_data
def get_shiiresaki_bin_data(kojo):

    # 不等ピッチなどを計算する
    def calculate_time_differences_adjacent_efficient(times):

        # 有効な時刻とそのインデックスを取得
        valid_times = [t for t in times if pd.notna(t)]
        valid_indices = [i + 1 for i in range(len(times)) if pd.notna(times[i])]  # 便番号を記録
        
        # 有効な時刻が2つ未満なら結果をゼロで返す
        if len(valid_times) < 2:
            return 0, '', 0
        
        # 時刻変換用関数
        def convert_time(t):
            if isinstance(t, dt.datetime):
                return t.time()
            elif isinstance(t, str):
                try:
                    return dt.datetime.strptime(t, '%H:%M:%S').time()
                except ValueError:
                    return dt.datetime.strptime(t, '%H:%M').time()
            return t

        # 時間差計算（日付をまたぐ場合考慮）
        def calculate_time_difference(time1, time2):
            dt_today = dt.datetime.today()
            datetime1 = dt.datetime.combine(dt_today, convert_time(time1))
            datetime2 = dt.datetime.combine(dt_today, convert_time(time2))
            if datetime2 < datetime1:
                datetime2 += dt.timedelta(days=1)
            return (datetime2 - datetime1).total_seconds() / 60  # 分単位

        # 隣合う便と最後の便から1便の時間差を計算
        time_diffs = [calculate_time_difference(valid_times[i], valid_times[i + 1])
                    for i in range(len(valid_times) - 1)]
        time_diffs.append(calculate_time_difference(valid_times[-1], valid_times[0]))  # 最後の便と1便

        # 時間差の詳細
        time_diff_details = [f"{valid_indices[i]}便と{valid_indices[i + 1]}便" for i in range(len(valid_indices) - 1)]
        time_diff_details.append(f"{valid_indices[-1]}便と{valid_indices[0]}便")

        # 最大時間差を取得し、詳細と時間単位も返す
        max_diff = max(time_diffs)
        max_diff_detail = time_diff_details[time_diffs.index(max_diff)]

        return max_diff, max_diff_detail, max_diff / 60

    # 日本語があるため、UTF-8でファイルを読み込む
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 仕入先ダイヤファイルの読み込み
    shiiresaki_bin_paths= config['shiiresaki_bin_path']
    shiiresaki_bin_path =  shiiresaki_bin_paths[kojo]
    # ★相対パスで読み込み
    shiiresaki_bin_path = os.path.join(current_dir,shiiresaki_bin_path)

    # openpyxlエンジンを使用してExcelファイルを読み込む
    df = pd.read_excel(
        shiiresaki_bin_path,
        engine='openpyxl',
        skiprows=4,
        sheet_name='納入時間一覧'  # シート名を指定
    )
    #print(df)
    #st.write("仕入先ダイヤの確認")
    #st.dataframe(df)

    # 列名をリセット
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

    # 抽出したい列名を指定
    # 先に固定の列を定義
    columns_to_extract = ['仕入先名', '発送場所名', '受入', '納入先', '納入LT(H)', '回数']

    # 1便から24便までの列名を生成して追加
    columns_to_extract += [f'{i}便' for i in range(1, 25)]

    # 指定した列のみを抽出
    extracted_df = df[columns_to_extract]

    # 各仕入先に対して最大便間時間とその詳細、時間単位も計算
    try:
        extracted_df['最長便ピッチ時間（分）'], extracted_df['最長便間詳細'], extracted_df['最長便時間（h）'] = zip(*extracted_df.apply(
            lambda row: calculate_time_differences_adjacent_efficient([row[f'{i}便'] for i in range(1, 25) if pd.notna(row[f'{i}便'])]), axis=1))
        extracted_df['等ピッチ時間（分）'] = 1150 / extracted_df['回数']
        extracted_df['不等ピッチ時間（分）'] = extracted_df['最長便ピッチ時間（分）'] - extracted_df['等ピッチ時間（分）']
    except Exception as e:
        extracted_df['最長便ピッチ時間（分）'] = 0
        extracted_df['等ピッチ時間（分）'] = 0
        extracted_df['不等ピッチ時間（分）'] = 0


    # '回数'が1の場合、'不等ピッチ時間（分）'を0に設定
    extracted_df.loc[extracted_df['回数'] == 1, '不等ピッチ時間（分）'] = 0

    extracted_df['不等ピッチ係数（日）'] = extracted_df['不等ピッチ時間（分）'] / 1150

    # 列名変更
    extracted_df = extracted_df.rename(columns={'受入': '整備室コード'})
    extracted_df = extracted_df.rename(columns={'発送場所名': '仕入先工場名'})

    extracted_df.to_csv('sample_shiiresaki_daiya.csv', index=False, encoding='shift_jis', errors='ignore')

    return extracted_df

# MARK: 異常お知らせ版（工場）から稼働時間を読み込みする
# todo DB化したい
# todo ライン識別コードの確認
# todo ディレクトリ構成めちゃくちゃ
# todo 【要確認】残業1時間単位で探しているけど、0.5とかない？
# todo ライン識別コードを第一工場と第二工場で変える必要あり
@st.cache_data
def get_kado_schedule_from_172_20_113_185(start_datetime, end_datetime, day_col, night_col, time_granularity):

    # 基本稼働テーブル作成（平日、残業無し）
    def create_hourly_kado_time_table(start_datetime, end_datetime, time_granularity):
        """
        指定された期間で1時間毎の稼働時間データフレームを作成する

        Parameters
        ----------
        start_datetime : str
            開始日時 (形式: 'YYYY-MM-DD HH:MM:SS')
        end_datetime : str
            終了日時 (形式: 'YYYY-MM-DD HH:MM:SS')

        Returns
        -------
        pandas.DataFrame
            日時列と稼働時間列を持つデータフレーム
            - 日時: 1時間間隔の日時
            - 稼働フラグ: 非稼働時間=0, 8時・12時=0.5, 13時=0.58, 17時=0.3, その他の稼働時間=1
        """

        # 日時範囲の作成
        if time_granularity == 'h':
            date_range = pd.date_range(start = start_datetime, end = end_datetime, freq = time_granularity)
        elif time_granularity == '15min':
            date_range = pd.date_range(start = start_datetime, end = end_datetime, freq = time_granularity)
        
        # データフレーム作成
        df = pd.DataFrame({
            '日時': date_range,
            '稼働フラグ': 1  # デフォルトで1を設定
        })
        
        # 非稼働フラグを0に設定
        non_working_hours = [2, 6, 7, 18, 19, 20, 21]
        df.loc[df['日時'].dt.hour.isin(non_working_hours), '稼働フラグ'] = 0

        # 特定の時間帯の稼働フラグを設定
        df.loc[df['日時'].dt.hour == 8, '稼働フラグ'] = 0.5
        df.loc[df['日時'].dt.hour == 12, '稼働フラグ'] = 0.5
        df.loc[df['日時'].dt.hour == 13, '稼働フラグ'] = 0.58
        df.loc[df['日時'].dt.hour == 17, '稼働フラグ'] = 0.3

        # 土曜の8時から月曜の7時までの稼働時間を0に設定
        df.loc[
            (
                # 土曜日で8時以降
                ((df['日時'].dt.dayofweek == 5) & (df['日時'].dt.hour >= 8)) |
                # 日曜日は終日
                (df['日時'].dt.dayofweek == 6) |
                # 月曜日で7時まで
                ((df['日時'].dt.dayofweek == 0) & (df['日時'].dt.hour <= 7))
            ),
            '稼働フラグ'
        ] = 0
        
        return df

    # 残業計画に基づいて稼働テーブルを変更
    def update_kado_flag_with_overtime(kado_df, overtime_df, day_col, night_col):

        """
        残業計画（昼勤・夜勤）に基づいて稼働フラグを更新する

        Parameters
        ----------
        kado_df : pandas.DataFrame
            稼働フラグを含むデータフレーム
            - 日時列: datetime
            - 稼働フラグ列: int (0 or 1)
        overtime_df : pandas.DataFrame
            残業計画を含むデータフレーム
            - 日時列
            - 計画（昼）列: int (1-4)
            - 計画（夜）列: int (1-2)
        day_col : str, optional
            昼勤計画の列名 (default: '計画（昼）')
        night_col : str, optional
            夜勤計画の列名 (default: '計画（夜）')

        Returns
        -------
        pandas.DataFrame
            更新された稼働フラグを持つデータフレーム
        """

        print(overtime_df)

        # 日付型に変換
        overtime_df['日付'] = pd.to_datetime(overtime_df['日付'], format='%Y%m%d', errors='coerce')

        # NaNを0に置換
        overtime_df[day_col] = overtime_df[day_col].fillna(0)
        overtime_df[night_col] = overtime_df[night_col].fillna(0)

        # 昼勤の処理
        for date, day_plan, day_pattern in zip(overtime_df['日付'], overtime_df[day_col], overtime_df['稼働ﾊﾟﾀｰﾝ']):

            date_only = date.date()

            print(date_only)

            # day_planがNULLまたはNoneの場合の処理（休日）
            if pd.isna(day_pattern):
                print("test",day_pattern)
                # 8時から22時までの稼働フラグを0に設定
                kado_df.loc[(kado_df['日時'].dt.date == date_only) & 
                            (kado_df['日時'].dt.hour.between(8, 23)), '稼働フラグ'] = 0
                continue  # 以降の処理をスキップ
            
            if int(float(day_plan))  == 1:
                # 18時のみ稼働
                kado_df.loc[(kado_df['日時'].dt.date == date_only) & 
                        (kado_df['日時'].dt.hour == 18), '稼働フラグ'] = 1
            elif int(float(day_plan))  == 2:
                # 18時、19時稼働
                kado_df.loc[(kado_df['日時'].dt.date == date_only) & 
                        (kado_df['日時'].dt.hour.isin([18, 19])), '稼働フラグ'] = 1
            elif int(float(day_plan))  == 3:
                # 18時、19時、20時稼働
                kado_df.loc[(kado_df['日時'].dt.date == date_only) & 
                        (kado_df['日時'].dt.hour.isin([18, 19, 20])), '稼働フラグ'] = 1
            elif int(float(day_plan))  == 4:
                # 18時、19時、20時、21時稼働
                kado_df.loc[(kado_df['日時'].dt.date == date_only) & 
                        (kado_df['日時'].dt.hour.isin([18, 19, 20, 21])), '稼働フラグ'] = 1

        # 夜勤の処理（翌日の早朝）
        for date, night_plan, night_pattern in zip(overtime_df['日付'], overtime_df[night_col], overtime_df['稼働ﾊﾟﾀｰﾝ']):

            next_date = date.date() + pd.Timedelta(days=1)  # 翌日の日付

            # day_planがNULLまたはNoneの場合の処理（休日）
            if pd.isna(night_pattern):
                # 8時から22時までの稼働フラグを0に設定
                kado_df.loc[(kado_df['日時'].dt.date == next_date) & 
                            (kado_df['日時'].dt.hour.between(0, 7)), '稼働フラグ'] = 0
                continue  # 以降の処理をスキップ
            
            if int(float(night_plan)) == 1:
                # 翌日6時のみ稼働
                kado_df.loc[(kado_df['日時'].dt.date == next_date) & 
                        (kado_df['日時'].dt.hour == 6), '稼働フラグ'] = 1
            elif int(float(night_plan)) == 2:
                # 翌日6時、7時稼働
                kado_df.loc[(kado_df['日時'].dt.date == next_date) & 
                        (kado_df['日時'].dt.hour.isin([6, 7])), '稼働フラグ'] = 1
        
        return kado_df

    # 基本稼働時間テーブル読み込み
    hourly_kado_df = create_hourly_kado_time_table(start_datetime, end_datetime, time_granularity)
    # 実行結果の確認
    #st.dataframe(hourly_kado_df)
    #print(hourly_kado_df)
    
    # 異常お知らせ版データ読み込み
    try:
        base_path = r'\\172.20.113.185\異常お知らせ板\data'

        #　ライン識別コード
        line = "J6"

        base_path_line = os.path.join(base_path, line)

        # YYYYMMの形式にする
        start_datetime_formatted = start_datetime[:4] + start_datetime[5:7]
        end_datetime_formatted = end_datetime[:4] + end_datetime[5:7]

        # datatime型にする
        start_datetime_obj = dt.datetime.strptime(start_datetime_formatted, '%Y%m') #todo ○○月1日になるはず
        start_datetime_obj = start_datetime_obj - dt.timedelta(days=1) #0時から7時までの夜勤分の稼働フラグを正確に計算するために1日前から処理を開始する
        end_datetime_obj = dt.datetime.strptime(end_datetime_formatted, '%Y%m')
        end_datetime_obj = end_datetime_obj + dt.timedelta(days=1) #夜勤の計算がされないため、1日＋で計算する

        #　残業時間など格納用
        all_kado = []
        # 指定期間を調べる
        current = start_datetime_obj
        while current <= end_datetime_obj:
            
            # パスを作成するために文字列にする
            current_str = current.strftime('%Y%m')

            # ある年月のパスを作成する
            base_path_line_YYYYMM = os.path.join(base_path_line, current_str)

            #print(base_path_line_YYYYMM)

            # 残業時間記載のフォルダーを定義
            file_name = f"{current_str}_{line}残業.csv"

            #　ある年月のCSVファイル名作成
            file_path = os.path.join(base_path_line_YYYYMM, file_name)

            # ファイルの読み込み
            df = pd.read_csv(file_path, encoding='shift-jis')

            # 転置してインデックスをリセット
            df_transposed = df.transpose().reset_index()

            # 最初に1行目を削除
            temp_kado_df = df_transposed.iloc[1:]

            # 1行目をコラムに設定
            new_columns = df_transposed.iloc[0]
            temp_kado_df.columns = new_columns

            # Dの形をYYYYMMDDの形に変換する
            temp_kado_df['日付'] = current_str + temp_kado_df['日付'].astype(str).str.zfill(2)

            # リストにデータフレームを追加
            all_kado.append(temp_kado_df)

            #　次の年月
            current += relativedelta(months=1)

        # 最後にすべてのデータフレームを一度に結合
        overtime_df = pd.concat(all_kado, ignore_index=True)
        
        # 実行結果を確認する
        #print(overtime_df)

        overtime_df.to_csv(f'{TEMP_OUTPUTS_PATH}/残業時間テーブル.csv', index=False,encoding='shift-jis')

        # 残業時間を考慮して稼働時間テーブルを更新する
        temp_kado_df = hourly_kado_df.copy()

        # 関数呼び出し
        updated_hourly_kado_df = update_kado_flag_with_overtime(temp_kado_df, overtime_df,day_col,night_col)
        #st.dataframe(hourly_kado_df)
        #print(updated_hourly_kado_df)

        updated_hourly_kado_df.to_csv(f'{TEMP_OUTPUTS_PATH}/稼働時間テーブル_残業時間考慮.csv', index=False, encoding='shift-jis')

        # 期間でフィルタリング（開始だけ-1にしていたので）
        start_datetime_time = dt.datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        end_datetime_time = dt.datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
        updated_hourly_kado_df = updated_hourly_kado_df[
            (updated_hourly_kado_df['日時'] >= start_datetime_time) & 
            (updated_hourly_kado_df['日時'] <= end_datetime_time)
        ]
        #st.write(start_datetime_obj,end_datetime_obj)
        #st.dataframe(updated_hourly_kado_df)

        return updated_hourly_kado_df
    
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        print("基本稼働時間テーブルを返します。")
        return hourly_kado_df

# MARK: Drsumの接続情報を読み込み
@st.cache_data
def get_connection_string_for_Drsum(kojo):

    """
    Dr.Sumデータベースへの接続文字列と工場固有の設定を取得する関数

    Args:
        kojo (str): 工場コード（例：'anjo1', 'anjo2'）

    Returns:
        tuple: (接続文字列, ラックテーブル名, かんばんテーブル名)
            - 接続文字列: データベース接続用の文字列
            - ラックテーブル名: 工場固有のラック情報テーブル名
            - かんばんテーブル名: 工場固有のかんばん情報テーブル名

    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        json.JSONDecodeError: JSON形式が不正な場合
        KeyError: 必要な設定キーが存在しない場合
    """

    # 日本語があるため、UTF-8でファイルを読み込む
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    #Drsumの接続情報を読み込み
    db = config['database_Drsum']

    # 接続文字列の生成と工場固有の設定を返す
    return (
        f"Driver={db['driver']};"
        f"Server={db['server']};"
        f"Port={db['port']};"
        f"Database={db['database']};"
        f"UID={db['uid']};"
        f"PWD={db['pwd']};"
    ), db['details'][kojo]['rack'], db['details'][kojo]['kanban']

# MARK: IBMDBの接続情報を読み込み
@st.cache_data
def get_connection_string_for_IBMDB(kojo):

    # 日本語があるため、UTF-8でファイルを読み込む
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    #Drsumの接続情報を読み込み
    db = config['database_IBMDB']

    # 接続文字列の生成と工場固有の設定を返す
    return (
        f"Driver={db['driver']};"
        f"Database={db['database']};"
        f"Hostname={db['hostname']};"
        f"Port={db['port']};"
        f"Protocol={db['protocol']};"
        f"UID={db['uid']};"
        f"PWD={db['pwd']};"
    ), db['uid'], db['pwd'], db['details'][kojo]['seibishitsu']

# MARK 品番情報を提示 from 手配系データ
@st.cache_data
def get_hinban_info_detail(hinban_info, selected_datetime, flag_display, flag_useDataBase, kojo):

    # 手配データの読み込み
    time_granularity = 'h'
    df = compute_hourly_tehai_data_by_hinban(hinban_info, selected_datetime, selected_datetime, time_granularity, flag_useDataBase, kojo)

    # 列名のリストを変数として使用
    columns_to_select = ['整備室コード', '仕入先名', '仕入先工場名',
     '品番', '品名', '収容数',
      'サイクル間隔', 'サイクル回数', 'サイクル情報', '登録箱種',
      '月末までの最大日量数（箱数）']
    df_selected = df[columns_to_select]

    if flag_display == 1:

        hinban_indo_detail_df = df_selected

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### 基本情報")
            col1.metric(
                label="整備室コード",
                value=hinban_indo_detail_df['整備室コード'].iloc[0]
            )
            col1.metric(
                label="仕入先名",
                value=hinban_indo_detail_df['仕入先名'].iloc[0]
            )
            col1.metric(
                label="仕入先工場名",
                value=hinban_indo_detail_df['仕入先工場名'].iloc[0]
            )

        with col2:
            st.write("### 品番情報")
            col2.metric(
                label="品番",
                value=hinban_indo_detail_df['品番'].iloc[0]
            )
            col2.metric(
                label="品名",
                value=hinban_indo_detail_df['品名'].iloc[0]
            )
            col2.metric(
                label="収容数",
                value=hinban_indo_detail_df['収容数'].iloc[0]
            )
            col2.metric(
                label="登録箱種",
                value=hinban_indo_detail_df['登録箱種'].iloc[0]
            )

        with col3:
            st.write("### かんばんサイクル情報")
            col3.metric(
                label="サイクル間隔",
                value=hinban_indo_detail_df['サイクル間隔'].iloc[0]
            )
            col3.metric(
                label="サイクル回数",
                value=hinban_indo_detail_df['サイクル回数'].iloc[0]
            )
            col3.metric(
                label="サイクル情報",
                value=hinban_indo_detail_df['サイクル情報'].iloc[0]
            )

    return df_selected

# MARK: マスター品番データの作成 from 手配データ:
#! データ元が過去手配なので1日前のデータを使用している
@st.cache_data
def get_hinban_master():

    #　過去の手配データを読み込む
    tehai_all_data = pd.read_csv(TEHAI_DATA_PATH, encoding='shift_jis')
    #st.datadrame(tehai_all_data)
    #st.write(TEHAI_DATA_PATH)

    # ユニークな品番_整備室コードの組み合わせを調べる
    unique_combinations = tehai_all_data[['品番', '整備室コード']].drop_duplicates()
    result = unique_combinations.apply(lambda x: f"{x['品番']}_{x['整備室コード']}", axis=1).tolist()

    #print(result)

    return result

# MARK: 品番＆受入整備室毎の部品在庫を抽出する from Dr.sum関数、在庫推移テーブル
@st.cache_data
def compute_hourly_buhin_zaiko_data_by_hinban(hinban_info, start_datetime, end_datetime, flag_useDataBase, kojo):

    
    """

    以下を実現する関数:
    1. 対象品番＆対象期間の自動ラックの在庫データの読み込み
    2. 拠点所番地と整備室を紐づけるマスター作成
    3. 在庫データに整備室情報の追加
    4. 必要な列情報を選択
    
    Parameters:
    ----------
    hinban_info（list）:ある品番
    start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
    end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）
    flag_useDataBase (bool): データベース使用フラグ
    kojo (str): 工場コード
    
    Returns:
    ----------
    buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

    """

    # T403物流情報_在庫推移テーブル（T403とT157）の在庫データを読み込み
    def load_zaiko_data_from_Drsum( hinban, start_datetime, end_datetime, kojo):

        """
        対象品番＆対象期間の自動ラックの在庫データの読み込む

        Parameters:
        hinban_info（str）：ある品番
        start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
        end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）

        Returns:
        buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

        """

        connection_string, rack_table , _ = get_connection_string_for_Drsum(kojo)
        print(connection_string)

        connection = pyodbc.connect(connection_string)
        cur = connection.cursor()

        # todo 品番列に"-"や" "が入っている
        # SQL文の作成
        sql = """
            SELECT *
            FROM {}
            WHERE REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') = '{}' AND 更新日時 >= '{}' AND 更新日時 <= '{}'
        """.format(rack_table, hinban, start_datetime, end_datetime)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            buhin_zaiko_data_by_hinban_df = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()

        return buhin_zaiko_data_by_hinban_df

    # 拠点諸番地を整備室名に置き換えるためのマスターを作成する関数
    def map_kyotenshobanchi_to_seibishitsu( hinban, kojo):

        """
        拠点所番地を整備室名に置き換えるためのマスターデータを作成する関数

        Parameters:
        ----------
        hinban : str
            対象となる品番

        Returns:
        -------
        pandas.DataFrame
            品番、整備室コード、拠点所番地を含むマスターデータ
            - 品番: ハイフンと空白を除去した品番
            - 整備室コード: 整備室の識別コード
            - 拠点所番地: 対応する拠点所番地

        Notes:
        -----
        - Dr.Sumデータベースに接続してデータを取得
        - 品番から特殊文字（ハイフン、半角空白、全角空白）を除去
        - 指定された期間内のユニークな組み合わせのみを抽出
        """

        #　接続情報など読み込み
        connection_string, _ , kanban_table = get_connection_string_for_Drsum(kojo)

        connection = pyodbc.connect(connection_string)
        cur = connection.cursor()

        # SQL文の作成（ユニークな品番と整備室を抽出）
        sql = """
            SELECT DISTINCT 
                REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') as 品番, 
                整備室コード, 
                拠点所番地
            FROM {}
            WHERE REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') = '{}'
        """.format(kanban_table, hinban)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            master_kyotenshobanchi_to_seibishitsu = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()
        connection.close()

        return master_kyotenshobanchi_to_seibishitsu

    # 品番情報設定
    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]
    # 将来）T403とT447を識別できるようにする

    print(hinban,seibishitsu)

    # 対象品番＆対象期間のデータを抽出する
    buhin_zaiko_data_by_hinban_df = load_zaiko_data_from_Drsum(hinban, start_datetime, end_datetime, kojo)
    # print("該当品番のデータ表示")
    #print(buhin_zaiko_data_by_hinban_df)

    # 対象整備室のデータを抽出する
    # 品番、整備室、拠点所番地のマスターを作成する
    master_kyotenshobanchi_to_seibishitsu = map_kyotenshobanchi_to_seibishitsu(hinban, kojo)
    # print("該当品番と拠点所番地のマスターデータ表示")
    # master_kyotenshobanchi_to_seibishitsu.to_csv('sample.csv', encoding='shift-jis', index=False)

    # 統合
    # 対象整備室情報を追加するため
    buhin_zaiko_data_by_hinban_df['品番'] = buhin_zaiko_data_by_hinban_df['品番'].astype(str).str.strip()
    buhin_zaiko_data_by_hinban_df['拠点所番地'] = buhin_zaiko_data_by_hinban_df['拠点所番地'].astype(str).str.strip()
    master_kyotenshobanchi_to_seibishitsu['品番'] = master_kyotenshobanchi_to_seibishitsu['品番'].astype(str).str.strip()
    master_kyotenshobanchi_to_seibishitsu['拠点所番地'] = master_kyotenshobanchi_to_seibishitsu['拠点所番地'].astype(str).str.strip()
    buhin_zaiko_data_by_hinban_df = pd.merge(buhin_zaiko_data_by_hinban_df, master_kyotenshobanchi_to_seibishitsu, on=['品番','拠点所番地'], how='left')
    #print("該当品番の結果（整備室情報追加バージョン）を表示")
    #print(buhin_zaiko_data_by_hinban_df)
    #buhin_zaiko_data_by_hinban_df.to_csv('sample2.csv', encoding='shift-jis', index=False)
    #st.dataframe(buhin_zaiko_data_by_hinban_df)

    # 対象整備室だけ抽出
    buhin_zaiko_data_by_hinban_df['整備室コード'] = buhin_zaiko_data_by_hinban_df['整備室コード'].astype(str).str.strip()
    buhin_zaiko_data_by_hinban_df = buhin_zaiko_data_by_hinban_df[buhin_zaiko_data_by_hinban_df['整備室コード'] == seibishitsu]

    # 列名変更
    buhin_zaiko_data_by_hinban_df = buhin_zaiko_data_by_hinban_df.rename(columns={'更新日時': '日時'})
    buhin_zaiko_data_by_hinban_df = buhin_zaiko_data_by_hinban_df.rename(columns={'現在在庫（箱）': '在庫数（箱）'})

    return buhin_zaiko_data_by_hinban_df

# MARL: コア計算（関所毎の計算の外だし）
@st.cache_data
def core_compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(specific_checkpoint_kanban_data_by_hinban_df, flag_useDataBase, kojo):

    # IN時間から非稼働時間をスキップしてOUT時間を計算する関数
    #! 30分ごとの判定でいいのか？
    def calculate_output_datetime(input_datetime, lt_minutes):
        """
        入庫日時（input_datetime）から出庫予定日時（OUT時間）を計算する関数。
        
        リードタイム（lt_minutes）は「稼働時間ベース」で加算される。
        非稼働時間帯はスキップされるため、実際のカレンダー時間とはずれる場合がある。

        Parameters:
        ----------
        input_datetime : datetime
            入庫予定日時（開始時刻）

        lt_minutes : int or float
            リードタイム（分単位）。稼働時間に基づきこの分数だけ加算して出庫予定日時を決定。

        Returns:
        -------
        datetime
            出庫予定日時（非稼働時間を除いた加算結果）
        """

        def is_operating_minute(dt):
            """
            指定時刻が稼働時間かどうかを判定する内部関数。

            稼働時間のルール：
            ------------------
            - ⛔ 土曜 9:00 以降 ～ 月曜 8:00 までは非稼働
            - ⛔ 日曜 終日 非稼働
            - ⛔ 月〜金の以下の時間帯は非稼働：
                - 6:00 ～ 6:59
                - 7:00 ～ 7:59
                - 12:30 ～ 13:29
                - 18:00 ～ 21:59
            - ⛔ 特定の日付・時間帯は非稼働
            """
            # 特定の日付と時間帯を非稼働として定義
            non_operating_schedules = [
                # (開始日時, 終了日時)の形式で指定
                (pd.Timestamp('2024-12-26 00:00'), pd.Timestamp('2025-01-06 07:00')),  # 元日は終日休業
                (pd.Timestamp('2025-04-26 13:00'), pd.Timestamp('2025-05-06 07:00')),  # 1/2の13時-17時は休業
                # 必要に応じて期間を追加
            ]

            # 特定の日付・時間帯チェック
            for start_dt, end_dt in non_operating_schedules:
                if start_dt <= dt <= end_dt:
                    return False

            weekday = dt.weekday()
            hour = dt.hour
            minute = dt.minute

            # 土曜9:00以降、日曜終日、月曜8:00未満は非稼働
            if (weekday == 5 and hour >= 9) or weekday == 6 or (weekday == 0 and hour < 8):
                return False

            # 月〜土曜の特定時間帯は非稼働
            if weekday in range(0, 6):  # 月〜土
                if hour in [6, 7]:  # 朝6〜7時台
                    return False
                if (hour == 12 and minute >= 30) or (hour == 13 and minute < 30):  # 昼休み
                    return False
                if 18 <= hour <= 21:  # 18:00〜21:59 は非稼働
                    return False

            # それ以外の時間は稼働とみなす
            return True

        # 入力が欠損 or リードタイムが0の場合、そのまま返す
        if pd.isna(input_datetime) or pd.isna(lt_minutes) or lt_minutes == 0:
            return input_datetime

        # カウントダウンしながら稼働分を加算
        current_time = input_datetime
        remaining_minutes = lt_minutes

        # 1分ずつ進めて、稼働時間だけ加算する
        while remaining_minutes > 0:
            current_time += pd.Timedelta(minutes=30)
            if is_operating_minute(current_time):
                remaining_minutes -= 30

        return current_time
        
    #　稼働時間分のリードタイムを計算する関数
    def calculate_true_operating_lt(row):
        """
        入庫日時から出庫日時までの間における「稼働時間（分単位）」をもとに
        正味のリードタイム（時間単位, float）を返す関数。
        """

        def is_operating_minute(dt):
            """
            指定された日時が「稼働している1分間」であるかを判定する内部関数。

            非稼働時間帯のルール：
            - 土曜8:00以降、日曜終日、月曜8:00未満は非稼働
            - 月〜土の以下時間帯も非稼働：
                ・6:00〜7:59
                ・2:00台
                ・18:00〜21:59
            """
            weekday = dt.weekday()
            hour = dt.hour
            minute = dt.minute

            if (weekday == 5 and hour >= 8) or weekday == 6 or (weekday == 0 and hour < 8):
                return False
            if weekday in range(0, 6):
                if hour in [6, 7] or hour == 2 or (18 <= hour <= 21):
                    return False
            return True

        # 入出庫時刻を取得
        start = row["順立装置入庫日時_FIFO補正済み"]
        end = row["順立装置出庫日時_FIFO補正済み"]

        # チェック（欠損または逆転）
        if pd.isna(start) or pd.isna(end) or end <= start:
            return np.nan

        # 1分刻みでタイムスタンプ生成（両端を含む）
        minute_range = pd.date_range(start=start.floor("min"), end=end.ceil("min"), freq="min")

        # 稼働時間（分）をカウント
        working_minutes = sum(is_operating_minute(dt) for dt in minute_range)

        # 分→時間へ変換
        working_hours = working_minutes / 60.0

        return working_hours

    # datetime型にする
    specific_checkpoint_kanban_data_by_hinban_df['検収日時'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['検収日時'], errors='coerce')
    specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時'], errors='coerce')
    specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時'], errors='coerce')
    #st.write("datetime型に変換完了")

    #「順立装置入庫日時」が 欠損していない行だけに絞る
    # その中で検収日時と順立装置入庫日時のペアを作成し、 
    # 最初の値を選ぶ
    # 検収日時 ➝ 入庫日時 のマッピングができる（Series 形式）
    kensyu_nyuko_map = (
        specific_checkpoint_kanban_data_by_hinban_df
        .dropna(subset=["順立装置入庫日時"])
        .groupby("検収日時")["順立装置入庫日時"]
        .first()  # 同じ検収日時が複数あれば最初の1つでよい
    )
    # 欠損している行（NaT）のマスクを作成
    mask = specific_checkpoint_kanban_data_by_hinban_df["順立装置入庫日時"].isna()
    # 欠損行だけに対して、検収日時 をキーに kensyu_nyuko_map を照会、順立装置入庫日時に補完値を代入
    specific_checkpoint_kanban_data_by_hinban_df.loc[mask, "順立装置入庫日時"] = specific_checkpoint_kanban_data_by_hinban_df.loc[mask, "検収日時"].map(kensyu_nyuko_map)
    # ここまでで、順立装置入庫日時に補完値を代入できる
    #st.write("順立装置入庫日時の置換完了")

    # 補完後も残っているNaNに対して現在時刻から1年後の日時を設定
    # 現在時刻から1年後の日時を設定
    one_year_from_now = pd.Timestamp.now() + pd.DateOffset(years=1)
    remaining_null_mask = specific_checkpoint_kanban_data_by_hinban_df["順立装置入庫日時"].isna()
    specific_checkpoint_kanban_data_by_hinban_df.loc[remaining_null_mask, "順立装置入庫日時"] = one_year_from_now
    remaining_null_mask = specific_checkpoint_kanban_data_by_hinban_df["順立装置出庫日時"].isna()
    specific_checkpoint_kanban_data_by_hinban_df.loc[remaining_null_mask, "順立装置出庫日時"] = one_year_from_now
    #st.write("順立装置入庫日時、出庫日時の穴埋め完了")

    # 仕入先ダイヤの読み込み（不等ピッチ等も計算する）
    shiiresaki_bin_data = get_shiiresaki_bin_data(kojo)
    #print(shiiresaki_bin_data)

    # 仕入先ダイヤを統合
    shiiresaki_bin_data = shiiresaki_bin_data.rename(columns={'発送場所名': '仕入先工場名'})
    shiiresaki_bin_data['仕入先工場名'] = shiiresaki_bin_data['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
    specific_checkpoint_kanban_data_by_hinban_df['仕入先工場名'] = specific_checkpoint_kanban_data_by_hinban_df['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
    specific_checkpoint_kanban_data_by_hinban_df = pd.merge(specific_checkpoint_kanban_data_by_hinban_df, shiiresaki_bin_data, on=['仕入先名','仕入先工場名','整備室コード'], how='left')
    #print(specific_checkpoint_kanban_data_by_hinban_df)

    # 抽出したデータに対して処理
    # "納入便"列から数値を取得
    specific_checkpoint_kanban_data_by_hinban_df['納入便'] = specific_checkpoint_kanban_data_by_hinban_df['納入便'].astype(int)
    # "○便"列の値を取得して新しい列"納入予定時間"に格納
    specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'] = specific_checkpoint_kanban_data_by_hinban_df.apply(
        lambda row: row[f"{row['納入便']}便"] if f"{row['納入便']}便" in specific_checkpoint_kanban_data_by_hinban_df.columns else None, axis=1)

    # "納入予定時間"列が0時～8時の場合に"納入日_補正"列を1日後に設定
    specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'], format='%H:%M:%S', errors='coerce').dt.time
    #todo 夜勤便は+1が必要！！今の計算でいいか不明！！
    specific_checkpoint_kanban_data_by_hinban_df['納入日_補正'] = specific_checkpoint_kanban_data_by_hinban_df.apply(lambda row: (pd.to_datetime(row['納入日']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                                                if row['納入予定時間'] and 0 <= row['納入予定時間'].hour < 6 else row['納入日'], axis=1)

    # "納入日_補正"列と"納入予定時間"列を統合し"納入予定日時"列を作成
    specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['納入日_補正']) + pd.to_timedelta(specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'].astype(str))
    #print(specific_checkpoint_kanban_data_by_hinban_df)
    #st.write("納入予定日時の計算完了")

    # 入庫予定日時の計算
    #todo 便毎に納入LTは違うよな
    # NAを0で埋めてから処理
    specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'] = specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'].fillna(0)
    #! Before
    # # 納入LTが0の場合と0以外の場合で分けて処理
    # specific_checkpoint_kanban_data_by_hinban_df['入庫予定日時'] = np.where(
    #     specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'] == 0,
    #     specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'],  # 納入LTが0の場合
    #     specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] + 
    #     pd.to_timedelta(specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'].astype(int), unit='hours')  # 納入LTが0以外の場合
    # )
    # #実行結果の確認
    # #st.dataframe(specific_checkpoint_kanban_data_by_hinban_df)
    #! after
    # 分単位に変換して新しい列に追加
    specific_checkpoint_kanban_data_by_hinban_df['納入LT(m)'] = specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'] * 60
    # 入庫予定日時（datetime型）と、LT（分） を使って出庫予定日時を作成
    specific_checkpoint_kanban_data_by_hinban_df["入庫予定日時"] = specific_checkpoint_kanban_data_by_hinban_df.apply(
        lambda row: calculate_output_datetime(row["納入予定日時"], row["納入LT(m)"]),
        axis=1
    )
    #st.write("入庫予定日時の計算完了")

    # 出庫予定日時の計算
    # 先入れ後出しを修正する
    # ! before
    # # 入庫日時でソート
    # specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.sort_values('順立装置入庫日時')
    # # 出庫日時を昇順ソートして再割り当て
    # specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時_FIFO補正済み'] = sorted(specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時'])
    # ! after
    # # 入庫予定日時でソート
    # specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.sort_values('入庫予定日時')
    # # 入庫日時を昇順ソートして再割り当て
    # specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時_FIFO補正済み'] = sorted(specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時'])
    # # 入庫日時でソート
    # specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.sort_values('順立装置入庫日時_FIFO補正済み')
    # # 出庫日時を昇順ソートして再割り当て
    # specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時_FIFO補正済み'] = sorted(specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時'])
    #! 高速化
    # 1. inplaceパラメータを使用
    specific_checkpoint_kanban_data_by_hinban_df.sort_values('入庫予定日時', inplace=True)
    # ソートと代入を一度の操作で行う
    specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時_FIFO補正済み'] = np.sort(specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時'].values)
    specific_checkpoint_kanban_data_by_hinban_df.sort_values('順立装置入庫日時_FIFO補正済み', inplace=True)
    specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時_FIFO補正済み'] = np.sort(specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時'].values)
    #st.write("順立装置入庫出庫日時_FIFO補正済みの計算完了")

    # float型に変換してから差分を計算
    # ! Before
    # 差分を計算し、時間単位を24で割って日数単位（24時間=1）に変換
    # specific_checkpoint_kanban_data_by_hinban_df['入庫出庫補正LT'] =(
    #     specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時_FIFO補正済み'] - specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時']).dt.total_seconds() / (60 * 60 * 24)
    # 全体（平日・休日区別なし）の場合
    # ! After
    specific_checkpoint_kanban_data_by_hinban_df["入庫出庫補正LT"] = specific_checkpoint_kanban_data_by_hinban_df.apply(calculate_true_operating_lt, axis=1)
    # # 中央値を計算
    # median_lt_all = specific_checkpoint_kanban_data_by_hinban_df['入庫出庫補正LT'].median()
    # # 中央値を設定
    # specific_checkpoint_kanban_data_by_hinban_df['入庫出庫補正LT（中央値）'] = median_lt_all
    # # 平日/休日の判定カラムを作成（例：土日を休日とする場合）
    # specific_checkpoint_kanban_data_by_hinban_df['is_holiday'] = specific_checkpoint_kanban_data_by_hinban_df['入庫予定日時'].dt.dayofweek.isin([5, 6])
    # # 平日と休日それぞれの中央値を計算
    # median_lt_weekday = specific_checkpoint_kanban_data_by_hinban_df[~specific_checkpoint_kanban_data_by_hinban_df['is_holiday']]['入庫出庫補正LT'].median()
    # median_lt_holiday = specific_checkpoint_kanban_data_by_hinban_df[specific_checkpoint_kanban_data_by_hinban_df['is_holiday']]['入庫出庫補正LT'].median()
    # # 条件に応じて中央値を設定
    # specific_checkpoint_kanban_data_by_hinban_df['入庫出庫補正LT（中央値）'] = np.where(
    #     specific_checkpoint_kanban_data_by_hinban_df['is_holiday'],
    #     median_lt_holiday,
    #     median_lt_weekday
    # )

    # # 非稼働時間（土日）を考慮して出庫予定時間を計算
    # def calculate_output_datetime(input_datetime, lt_minutes):
    #     """
    #     入庫日時とLTから出庫日時を計算する関数
    #     土曜9:00から月曜8:00までの時間をスキップ
        
    #     Parameters:
    #     input_datetime: 入庫日時
    #     lt_minutes: リードタイム（分）
    #     """
    #     if lt_minutes == 0:
    #         return input_datetime
        
    #     current_time = input_datetime
    #     remaining_minutes = lt_minutes
        
    #     while remaining_minutes > 0:
    #         # 1分進める
    #         current_time += pd.Timedelta(minutes=1)
            
    #         # 土曜9:00から月曜8:00の場合はスキップ
    #         if (current_time.dayofweek == 5 and current_time.hour >= 8) or \
    #         (current_time.dayofweek == 6) or \
    #         (current_time.dayofweek == 0 and current_time.hour < 7):
    #             continue
                
    #         remaining_minutes -= 1
        
    #     return current_time
        
    # # ! Before
    # # データフレームに適用
    # specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時'] = specific_checkpoint_kanban_data_by_hinban_df.apply(
    #     lambda row: calculate_output_datetime(
    #         row['入庫予定日時'],
    #         row['入庫出庫補正LT（中央値）'] * 24 * 60
    #     ),
    #     axis=1
    # )
    #! After
    # 1. KDEで分布学習（日数ベース）
    lt_data = specific_checkpoint_kanban_data_by_hinban_df["入庫出庫補正LT"].dropna()

    try:
        # KDEの計算を試みる
        kde = gaussian_kde(lt_data)
        kde = gaussian_kde(lt_data)
        #st.write("分布から入庫出庫補正LT計算完了")

        # 2. サンプリング：DataFrameの行数と同じ数だけ生成
        generated_lts = kde.resample(size=len(specific_checkpoint_kanban_data_by_hinban_df)).flatten()

        # 3. 分単位に変換して新しい列に追加
        specific_checkpoint_kanban_data_by_hinban_df["分布由来LT（分）"] = generated_lts * 60

        # 出庫予定日時を計算して新しい列 "出庫予定日時_temp" に格納
        # 条件：
        # - 順立装置入庫日時_FIFO補正済み があればそれを使用
        # - なければ 入庫予定日時 を代わりに使う
        # - リードタイム（分）には 分布由来LT（分）を使用
        specific_checkpoint_kanban_data_by_hinban_df["出庫予定日時_temp"] = specific_checkpoint_kanban_data_by_hinban_df.apply(
            lambda row: calculate_output_datetime(
                # 入庫日時の決定：補正済みがあれば優先、それがなければ入庫予定日時を使う
                row["順立装置入庫日時_FIFO補正済み"]
                if pd.notna(row["順立装置入庫日時_FIFO補正済み"])
                else row["入庫予定日時"],

                # 使用するリードタイム（分）は事前にKDEからサンプリングした「分布由来LT」
                row["分布由来LT（分）"]
            ),
            axis=1  # 各行ごとに処理
        )

        # # 入庫予定日時でソート
        # specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.sort_values('入庫予定日時')
        # # 出庫日時を昇順ソートして再割り当て
        # specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時'] = sorted(specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時_temp'])
        #! 高速化
        # 入庫予定日時でソート（inplace=Trueで代入を省略）
        specific_checkpoint_kanban_data_by_hinban_df.sort_values('入庫予定日時', inplace=True)
        # 出庫予定日時の再割り当てをnp.sortで高速化
        specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時'] = np.sort(specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時_temp'].values)
        #st.write("出庫予定日時計算完了")
    except ValueError:
        #! リミット計算するときは要素１で分布を計算できないため
        return specific_checkpoint_kanban_data_by_hinban_df

    #実行結果の確認
    #specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時'] = specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時']

    return specific_checkpoint_kanban_data_by_hinban_df

# MARK: 品番＆受入整備室毎のかんばんデータを関所毎に抽出して、合計枚数を計算する関数 from Dr.sum関数、所在管理テーブル
# todo 入出庫タイムスタンプのみ存在する品番データあり。現状はNone行は削除。納入予定日の計算でエラーが出るため
#! 仕入先ダイヤエクセル読み込みあり
@st.cache_data
def compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

    """
    品番と受入整備室毎のかんばんデータを1時間単位で集計する関数

    Args:
        hinban_info (tuple): (品番, 整備室コード)
        target_column (str): 集計対象の列名
        start_datetime (datetime): 集計開始日時
        end_datetime (datetime): 集計終了日時
        flag_useDataBase (bool): データベース使用フラグ
        kojo (str): 工場コード

    Returns:
        DataFrame: 1時間毎のかんばん数集計結果
    """

    # 品番情報設定
    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]

    #　接続情報など読み込み
    connection_string, _ , kanban_table = get_connection_string_for_Drsum(kojo)

    # データベースへの接続を確立
    connection = pyodbc.connect(connection_string)
    cur = connection.cursor()

    # SQL文の作成（ユニークな品番と整備室を抽出）
    #! 発注取り消しもあるので、発注取り消しされていないデータを吸い出す
    # Other DBAPI2
    sql = """
        SELECT *
        FROM {}
        WHERE REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') = '{}' AND 整備室コード = '{}' AND 更新日時 >= '{}' AND 更新日時 <= '{}' AND 発注取消日時 IS NULL
    """.format(kanban_table, hinban, seibishitsu, start_datetime, end_datetime)
    # SQL文の実行
    cur.execute(sql)
    # 結果をデータフレームに読み込み
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        specific_checkpoint_kanban_data_by_hinban_df = pd.read_sql(sql, con=connection)
    #print(specific_checkpoint_kanban_data_by_hinban_df)
    # 接続を閉じる
    cur.close()

    #! Dr.Sum ODBCドライバーがSQLAlchemyの標準的なSQL構文と互換性がない
    # # SQLAchemy版
    # # エンジンの作成
    # engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}")
    # # SQLクエリの作成
    # sql = text("""
    #     SELECT *
    #     FROM :table
    #     WHERE REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') = :hinban 
    #     AND 整備室コード = :seibishitsu 
    #     AND 更新日時 >= :start_datetime 
    #     AND 更新日時 <= :end_datetime 
    #     AND 発注取消日時 IS NULL
    # """)
    # # パラメータの設定
    # params = {
    #     "table": kanban_table,
    #     "hinban": hinban,
    #     "seibishitsu": seibishitsu,
    #     "start_datetime": start_datetime,
    #     "end_datetime": end_datetime
    # }
    # # クエリの実行とデータフレームへの読み込み
    # with engine.connect() as connection:
    #     specific_checkpoint_kanban_data_by_hinban_df = pd.read_sql(sql, connection, params=params)
    
    # 特定の列('納入便')が None の行を削除
    #!　入庫出庫はあるが納入便などがないデータ、品番がある。Noneの行は削除
    specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.dropna(subset=['納入便'])
    #st.dataframe(specific_checkpoint_kanban_data_by_hinban_df)

    #　かんばん1枚以上の場合
    if len(specific_checkpoint_kanban_data_by_hinban_df) != 0:

        specific_checkpoint_kanban_data_by_hinban_df = core_compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(specific_checkpoint_kanban_data_by_hinban_df, flag_useDataBase, kojo)
        
    #　かんばん0枚の場合
    else:
        #todo エラー出ないようにするための一時的な対応
        specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] = specific_checkpoint_kanban_data_by_hinban_df['検収日時']
        specific_checkpoint_kanban_data_by_hinban_df['入庫予定日時'] = specific_checkpoint_kanban_data_by_hinban_df['順立装置入庫日時']
        specific_checkpoint_kanban_data_by_hinban_df['出庫予定日時'] = specific_checkpoint_kanban_data_by_hinban_df['順立装置出庫日時']

    # memo
    # ここまでに完全なかんばんデータが作成される

    #実行結果の保存
    specific_checkpoint_kanban_data_by_hinban_df.to_csv('sample_kanban_data.csv', encoding='utf-8-sig', index=False)

    # target_column毎の処理を開始

    if target_column != '西尾東~部品置き場の間の滞留' and target_column != '期待かんばん在庫' and target_column != '順立装置内の滞留と前倒し出庫の差分':

        # 1時間毎のかんばん数を計算する
        # 納入予定日時を datetime 型に変換
        specific_checkpoint_kanban_data_by_hinban_df[target_column] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df[target_column])
        # 時間単位で丸める（1時間単位にグループ化）
        if time_granularity == 'h':
            specific_checkpoint_kanban_data_by_hinban_df['日時'] = specific_checkpoint_kanban_data_by_hinban_df[target_column].dt.floor(time_granularity)
        elif time_granularity == '15min':
            specific_checkpoint_kanban_data_by_hinban_df['日時'] = specific_checkpoint_kanban_data_by_hinban_df[target_column].dt.floor(time_granularity)
        # グループ化して集計
        column_name = target_column + "のかんばん数"
        hourly_specific_checkpoint_kanbansu_data_by_hinban = specific_checkpoint_kanban_data_by_hinban_df.groupby('日時').size().reset_index(name=column_name)

        #　対象時間を計算する
        if time_granularity == 'h':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)
        elif time_granularity == '15min':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)

        # 対象時間でデータフレームを作成し、欠損値は0で埋める
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full = pd.DataFrame(full_time_range, columns=['日時']).merge(hourly_specific_checkpoint_kanbansu_data_by_hinban, on='日時', how='left').fillna(0)

        # かんばん数を整数に戻す
        hourly_specific_checkpoint_kanbansu_data_by_hinban[column_name] = hourly_specific_checkpoint_kanbansu_data_by_hinban[column_name].astype(int)
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full[column_name] = hourly_specific_checkpoint_kanbansu_data_by_hinban_full[column_name].astype(int)

    elif target_column == '西尾東~部品置き場の間の滞留':

        #　対象時間を計算する
        if time_granularity == 'h':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)
        elif time_granularity == '15min':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)

        # 対象かんばんデータ
        df = specific_checkpoint_kanban_data_by_hinban_df
        
        # 各時点での未入庫数を計算
        results = []
        for time_point in full_time_range:
            unstocked_parts = df[
                (df['入庫予定日時'] <= time_point) & 
                ((df['順立装置入庫日時_FIFO補正済み'] > time_point) | (df['順立装置入庫日時_FIFO補正済み'].isna())) &
                (df['順立装置出庫日時_FIFO補正済み'] > time_point)
            ].copy()

            unstocked_parts["滞留時間_時間単位"] = (time_point - unstocked_parts["入庫予定日時"]).dt.total_seconds() / 3600

            total_stay_time = unstocked_parts["滞留時間_時間単位"].sum()

            results.append({
                "日時": time_point,
                "西尾東~部品置き場の間の滞留かんばん数_枚数単位": len(unstocked_parts),
                "西尾東~部品置き場の間の滞留かんばん数_時間単位": total_stay_time
            })

        # データフレームとして返す
        hourly_specific_checkpoint_kanbansu_data_by_hinban = pd.DataFrame(results)
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full = pd.DataFrame(results)

    elif target_column == '期待かんばん在庫':

        #　対象時間を計算する
        if time_granularity == 'h':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)
        elif time_granularity == '15min':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)

        # 対象かんばんデータ
        df = specific_checkpoint_kanban_data_by_hinban_df
        
        # 各時点でのかんばんベースを計算
        results = []
        for time_point in full_time_range:
            # その時点までに入庫予定があり、かつまだ出庫予定がないかんばん数
            unstocked_parts = df[
                (df['入庫予定日時'] <= time_point) & 
                ((df['出庫予定日時'] > time_point) | (df['出庫予定日時'].isna()))
            ]
            
            results.append({
                "日時": time_point,
                "期待かんばん在庫数": len(unstocked_parts)
            })

        # データフレームとして返す
        hourly_specific_checkpoint_kanbansu_data_by_hinban = pd.DataFrame(results)
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full = pd.DataFrame(results)

    elif target_column == '順立装置内の滞留と前倒し出庫の差分':

        #　対象時間を計算する
        if time_granularity == 'h':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)
        elif time_granularity == '15min':
            full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq=time_granularity)

        # 対象かんばんデータ
        df = specific_checkpoint_kanban_data_by_hinban_df
        
        results = []
        for time_point in full_time_range:
            unstocked_parts_plus = df[
                (df['出庫予定日時'] <= time_point) & 
                ((df['順立装置出庫日時'] > time_point) | (df['順立装置出庫日時'].isna()))
            ].copy()

            unstocked_parts_minus = df[
                (df['順立装置出庫日時'] <= time_point) & 
                (df['出庫予定日時'] > time_point)
            ].copy()

            unstocked_parts_plus["滞留時間_時間単位"] = (time_point - unstocked_parts_plus["出庫予定日時"]).dt.total_seconds() / 3600
            unstocked_parts_minus["前倒し時間_時間単位"] = (unstocked_parts_minus["出庫予定日時"] - time_point).dt.total_seconds() / 3600

            sum_plus = unstocked_parts_plus["滞留時間_時間単位"].sum()
            sum_minus = unstocked_parts_minus["前倒し時間_時間単位"].sum()

            results.append({
                "日時": time_point,
                "順立装置内の滞留と前倒し出庫の差分_枚数単位": len(unstocked_parts_plus) - len(unstocked_parts_minus),
                "順立装置内の滞留と前倒し出庫の差分_時間単位": sum_plus - sum_minus
            })

        # データフレームとして返す
        hourly_specific_checkpoint_kanbansu_data_by_hinban = pd.DataFrame(results)
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full = pd.DataFrame(results)

    #実行結果の保存
    #specific_checkpoint_kanban_data_by_hinban_df.to_csv('sample_syozai_kanban.csv', encoding='cp932', index=False)
    #hourly_specific_checkpoint_kanbansu_data_by_hinban.to_csv('sample_syozai.csv', encoding='shift-jis', index=False)
    
    # 0でない枚数の日時のときのデータフレームのみ、対象全時間のデータフレーム
    return hourly_specific_checkpoint_kanbansu_data_by_hinban, hourly_specific_checkpoint_kanbansu_data_by_hinban_full

# MARK: 品番＆受入整備室毎の部品在庫を抽出する（★全品番） from Dr.sum関数、在庫推移テーブル
# ある時点で使用すること推奨
@st.cache_data
def compute_hourly_buhin_zaiko_data_by_all_hinban(start_datetime, end_datetime, flag_useDataBase, kojo):

    """

    以下を実現する関数:
    1. 対象品番＆対象期間の自動ラックの在庫データの読み込み
    2. 拠点所番地と整備室を紐づけるマスター作成
    3. 在庫データに整備室情報の追加
    4. 必要な列情報を選択
    
    Parameters:
    ----------
    hinban_info（list）:ある品番
    start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
    end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）
    flag_useDataBase (bool): データベース使用フラグ
    kojo (str): 工場コード
    
    Returns:
    ----------
    buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

    """

    # T403物流情報_在庫推移テーブル（T403とT157）の在庫データを読み込み
    def load_zaiko_data_from_Drsum(start_datetime, end_datetime, kojo):

        """
        全品番＆対象期間の自動ラックの在庫データの読み込む

        Parameters:
        hinban_info（str）：ある品番
        start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
        end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）

        Returns:
        buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

        """

        connection_string, rack_table , _ = get_connection_string_for_Drsum(kojo)
        print(connection_string)

        connection = pyodbc.connect(connection_string)
        cur = connection.cursor()

        # todo 品番列に"-"や" "が入っている
        # SQL文の作成
        sql = """
            SELECT *
            FROM {}
            WHERE  更新日時 >= '{}' AND 更新日時 <= '{}'
        """.format(rack_table, start_datetime, end_datetime)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        buhin_zaiko_data_by_hinban_df = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()

        return buhin_zaiko_data_by_hinban_df

    # 拠点諸番地を整備室名に置き換えるためのマスターを作成する関数
    def map_kyotenshobanchi_to_seibishitsu(kojo):

        """
        拠点所番地を整備室名に置き換えるためのマスターデータを作成する関数

        Parameters:
        ----------
        hinban : str
            対象となる品番

        Returns:
        -------
        pandas.DataFrame
            品番、整備室コード、拠点所番地を含むマスターデータ
            - 品番: ハイフンと空白を除去した品番
            - 整備室コード: 整備室の識別コード
            - 拠点所番地: 対応する拠点所番地

        Notes:
        -----
        - Dr.Sumデータベースに接続してデータを取得
        - 品番から特殊文字（ハイフン、半角空白、全角空白）を除去
        - 指定された期間内のユニークな組み合わせのみを抽出
        """

        #　接続情報など読み込み
        connection_string, _ , kanban_table = get_connection_string_for_Drsum(kojo)

        connection = pyodbc.connect(connection_string)
        cur = connection.cursor()

        # SQL文の作成（ユニークな品番と整備室を抽出）
        sql = """
            SELECT DISTINCT 
                REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') as 品番, 
                整備室コード, 
                拠点所番地
            FROM {}
        """.format(kanban_table)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        master_kyotenshobanchi_to_seibishitsu = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()
        connection.close()

        return master_kyotenshobanchi_to_seibishitsu

    
    #print(hinban,seibishitsu)

    # 全品番＆対象期間のデータを抽出する
    buhin_zaiko_data_by_hinban_df = load_zaiko_data_from_Drsum(start_datetime, end_datetime, kojo)
    # print("該当品番のデータ表示")
    #print(buhin_zaiko_data_by_hinban_df)

    # 対象整備室のデータを抽出する
    # 品番、整備室、拠点所番地のマスターを作成する
    master_kyotenshobanchi_to_seibishitsu = map_kyotenshobanchi_to_seibishitsu(kojo)
    # print("該当品番と拠点所番地のマスターデータ表示")
    # master_kyotenshobanchi_to_seibishitsu.to_csv('sample.csv', encoding='shift-jis', index=False)

    # 統合
    # 対象整備室情報を追加するため
    buhin_zaiko_data_by_hinban_df['品番'] = buhin_zaiko_data_by_hinban_df['品番'].astype(str).str.strip()
    buhin_zaiko_data_by_hinban_df['拠点所番地'] = buhin_zaiko_data_by_hinban_df['拠点所番地'].astype(str).str.strip()
    master_kyotenshobanchi_to_seibishitsu['品番'] = master_kyotenshobanchi_to_seibishitsu['品番'].astype(str).str.strip()
    master_kyotenshobanchi_to_seibishitsu['拠点所番地'] = master_kyotenshobanchi_to_seibishitsu['拠点所番地'].astype(str).str.strip()
    buhin_zaiko_data_by_hinban_df = pd.merge(buhin_zaiko_data_by_hinban_df, master_kyotenshobanchi_to_seibishitsu, on=['品番','拠点所番地'], how='left')
    #print("該当品番の結果（整備室情報追加バージョン）を表示")
    #print(buhin_zaiko_data_by_hinban_df)
    #buhin_zaiko_data_by_hinban_df.to_csv('sample2.csv', encoding='shift-jis', index=False)

    # 列名変更
    buhin_zaiko_data_by_hinban_df = buhin_zaiko_data_by_hinban_df.rename(columns={'更新日時': '日時'})
    buhin_zaiko_data_by_hinban_df = buhin_zaiko_data_by_hinban_df.rename(columns={'現在在庫（箱）': '在庫数（箱）'})

    return buhin_zaiko_data_by_hinban_df

# MARK: 品番＆受入整備室毎のかんばんデータを関所毎に抽出して、合計枚数を計算する関数（★全品番） from Dr.sum関数、所在管理テーブル
# todo 入出庫タイムスタンプのみ存在する品番データあり。現状はNone行は削除。納入予定日の計算でエラーが出るため
#! 仕入先ダイヤエクセル読み込みあり
@st.cache_data
def compute_hourly_specific_checkpoint_kanbansu_data_by_all_hinban(target_column, start_datetime, end_datetime, flag_useDataBase, kojo):

    """
    品番と受入整備室毎のかんばんデータを1時間単位で集計する関数

    Args:
        hinban_info (tuple): (品番, 整備室コード)
        target_column (str): 集計対象の列名
        start_datetime (datetime): 集計開始日時
        end_datetime (datetime): 集計終了日時
        flag_useDataBase (bool): データベース使用フラグ
        kojo (str): 工場コード

    Returns:
        DataFrame: 1時間毎のかんばん数集計結果
    """

    #　接続情報など読み込み
    connection_string, _ , kanban_table = get_connection_string_for_Drsum(kojo)

    # データベースへの接続を確立
    connection = pyodbc.connect(connection_string)
    cur = connection.cursor()

    # SQL文の作成（ユニークな品番と整備室を抽出）
    #! 発注取り消しもあるので、発注取り消しされていないデータを吸い出す
    sql = """
        SELECT *
        FROM {}
        WHERE  更新日時 >= '{}' AND 更新日時 <= '{}' AND 発注取消日時 IS NULL
    """.format(kanban_table, start_datetime, end_datetime)

    # SQL文の実行
    cur.execute(sql)

    # 結果をデータフレームに読み込み
    specific_checkpoint_kanban_data_by_hinban_df = pd.read_sql(sql, con=connection)
    print(specific_checkpoint_kanban_data_by_hinban_df)

    # 接続を閉じる
    cur.close()

    # 特定の列('納入便')が None の行を削除
    #!　入庫出庫はあるが納入便などがないデータ、品番がある。Noneの行は削除
    specific_checkpoint_kanban_data_by_hinban_df = specific_checkpoint_kanban_data_by_hinban_df.dropna(subset=['納入便'])
    #st.dataframe(specific_checkpoint_kanban_data_by_hinban_df)

    #　かんばん1枚以上の場合
    if len(specific_checkpoint_kanban_data_by_hinban_df) != 0:

        # 仕入先ダイヤの読み込み（不等ピッチ等も計算する）
        shiiresaki_bin_data = get_shiiresaki_bin_data(kojo)
        #print(shiiresaki_bin_data)

        # 仕入先ダイヤを統合
        shiiresaki_bin_data = shiiresaki_bin_data.rename(columns={'発送場所名': '仕入先工場名'})
        shiiresaki_bin_data['仕入先工場名'] = shiiresaki_bin_data['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
        specific_checkpoint_kanban_data_by_hinban_df['仕入先工場名'] = specific_checkpoint_kanban_data_by_hinban_df['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
        specific_checkpoint_kanban_data_by_hinban_df = pd.merge(specific_checkpoint_kanban_data_by_hinban_df, shiiresaki_bin_data, on=['仕入先名','仕入先工場名','整備室コード'], how='left')
        print(specific_checkpoint_kanban_data_by_hinban_df)

        # 抽出したデータに対して処理
        # "納入便"列から数値を取得
        specific_checkpoint_kanban_data_by_hinban_df['納入便'] = specific_checkpoint_kanban_data_by_hinban_df['納入便'].astype(int)
        # "○便"列の値を取得して新しい列"納入予定時間"に格納
        specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'] = specific_checkpoint_kanban_data_by_hinban_df.apply(
            lambda row: row[f"{row['納入便']}便"] if f"{row['納入便']}便" in specific_checkpoint_kanban_data_by_hinban_df.columns else None, axis=1)

        # "納入予定時間"列が0時～8時の場合に"納入日_補正"列を1日後に設定
        specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'], format='%H:%M:%S', errors='coerce').dt.time
        #todo 夜勤便は+1が必要！！今の計算でいいか不明！！
        specific_checkpoint_kanban_data_by_hinban_df['納入日_補正'] = specific_checkpoint_kanban_data_by_hinban_df.apply(lambda row: (pd.to_datetime(row['納入日']) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                                                    if row['納入予定時間'] and 0 <= row['納入予定時間'].hour < 6 else row['納入日'], axis=1)

        # "納入日_補正"列と"納入予定時間"列を統合し"納入予定日時"列を作成
        specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] = pd.to_datetime(specific_checkpoint_kanban_data_by_hinban_df['納入日_補正']) + pd.to_timedelta(specific_checkpoint_kanban_data_by_hinban_df['納入予定時間'].astype(str))
        print(specific_checkpoint_kanban_data_by_hinban_df)
        specific_checkpoint_kanban_data_by_hinban_df.to_csv('sample.csv', encoding='utf-8-sig', index=False)

        # NAを0で埋めてから処理
        specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'] = specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'].fillna(0)
        # 納入LTが0の場合と0以外の場合で分けて処理
        specific_checkpoint_kanban_data_by_hinban_df['入庫予定日時'] = np.where(
            specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'] == 0,
            specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'],  # 納入LTが0の場合
            specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] + 
            pd.to_timedelta(specific_checkpoint_kanban_data_by_hinban_df['納入LT(H)'].astype(int), unit='hours')  # 納入LTが0以外の場合
        )

    #　かんばん0枚の場合
    else:
        specific_checkpoint_kanban_data_by_hinban_df['納入予定日時'] = specific_checkpoint_kanban_data_by_hinban_df['検収日時']

    # memo
    # ここまでに完全なかんばんデータが作成される

    #st.dataframe(specific_checkpoint_kanban_data_by_hinban_df)

    # ユニークな組み合わせを取得
    unique_hinban_seibishitsu = specific_checkpoint_kanban_data_by_hinban_df[['品番', '整備室コード']].drop_duplicates()

    dfs_list = []
    dfs_list_full = []

    # target_column毎の処理を開始
    for _, row in unique_hinban_seibishitsu.iterrows():

        hinban = row['品番']
        seibishitsu = row['整備室コード']

        # 特定の品番と整備室コードの組み合わせに該当する行を抽出する
        filtered_df = specific_checkpoint_kanban_data_by_hinban_df[(specific_checkpoint_kanban_data_by_hinban_df['品番'] == hinban) & (specific_checkpoint_kanban_data_by_hinban_df['整備室コード'] == seibishitsu)]

        # 1時間毎のかんばん数を計算する
        # 納入予定日時を datetime 型に変換
        filtered_df[target_column] = pd.to_datetime(filtered_df[target_column])
        # 時間単位で丸める（1時間単位にグループ化）
        filtered_df['日時'] = filtered_df[target_column].dt.floor('h')
        # グループ化して集計
        column_name = target_column + "のかんばん数"
        hourly_specific_checkpoint_kanbansu_data_by_hinban = filtered_df.groupby('日時').size().reset_index(name=column_name)

        #　対象時間を計算する
        full_time_range = pd.date_range( start=start_datetime, end=end_datetime,freq='h')

        # 対象時間でデータフレームを作成し、欠損値は0で埋める
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full = pd.DataFrame(full_time_range, columns=['日時']).merge(hourly_specific_checkpoint_kanbansu_data_by_hinban, on='日時', how='left').fillna(0)

        # かんばん数を整数に戻す
        hourly_specific_checkpoint_kanbansu_data_by_hinban[column_name] = hourly_specific_checkpoint_kanbansu_data_by_hinban[column_name].astype(int)
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full[column_name] = hourly_specific_checkpoint_kanbansu_data_by_hinban_full[column_name].astype(int)

        hourly_specific_checkpoint_kanbansu_data_by_hinban["品番"] = hinban
        hourly_specific_checkpoint_kanbansu_data_by_hinban["整備室コード"] = seibishitsu
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full["品番"] = hinban
        hourly_specific_checkpoint_kanbansu_data_by_hinban_full["整備室コード"] = seibishitsu

        # リストに追加
        dfs_list.append(hourly_specific_checkpoint_kanbansu_data_by_hinban)
        dfs_list_full.append(hourly_specific_checkpoint_kanbansu_data_by_hinban_full)

    # リストからデータフレームを結合
    dfs_list_df = pd.concat(dfs_list, ignore_index=True)
    dfs_list_full_df = pd.concat(dfs_list_full, ignore_index=True)

    dfs_list_df['品番'] = dfs_list_df['品番'].astype(str).str.strip()
    dfs_list_full_df['品番'] = dfs_list_full_df['品番'].astype(str).str.strip()

    # 0でない枚数の日時のときのデータフレームのみ、対象全時間のデータフレーム
    return dfs_list_df, dfs_list_full_df

# MARK: 手配必要数データを読み込む
# todo 何度も読み込まないように全品番にしている
#! 実行日から半年前を保存
#! キャッシュ無しか
@st.cache_data
def compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

    #todo　手配区分4だと収容数が空白になる

    # 品番＆受入整備室＆仕入先工場＆仕入先工場名のマスターを作成
    # Activeにデータがないため
    def compute_master_of_shiiresaki(start_datetime, end_datetime, kojo):

        # 利用機会
        # 手配情報データを作成するとき

        #　接続情報など読み込み
        connection_string, _ , kanban_table = get_connection_string_for_Drsum(kojo)

        connection = pyodbc.connect(connection_string)
        cur = connection.cursor()

        # SQL文の作成（ユニークな品番と整備室を抽出）
        sql = """
            SELECT DISTINCT 
                REPLACE(REPLACE(REPLACE(品番, '-', ''), ' ', ''), '　', '') as 品番, 
                整備室コード, 
                拠点所番地,
                仕入先名,
                仕入先工場名
            FROM {}
            WHERE 更新日時 >= '{}' AND 更新日時 <= '{}'
        """.format(kanban_table, start_datetime, end_datetime)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            master_of_shiiresaki = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()
        connection.close()

        #todo 拠点所番地が空白の行がありダブる
        #! 以下列の値が空白でない行のみ抽出
        master_of_shiiresaki = master_of_shiiresaki[
            (master_of_shiiresaki['品番'].notna()) &
            (master_of_shiiresaki['品番'].str.strip() != '') &
            (master_of_shiiresaki['整備室コード'].notna()) &
            (master_of_shiiresaki['整備室コード'].str.strip() != '') &
            (master_of_shiiresaki['拠点所番地'].notna()) &
            (master_of_shiiresaki['拠点所番地'].str.strip() != '')&
            (master_of_shiiresaki['仕入先名'].notna()) &
            (master_of_shiiresaki['仕入先名'].str.strip() != '')
        ]

        # 品番情報統一
        master_of_shiiresaki['品番'] = master_of_shiiresaki['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')

        #　仕入先ダイヤの読み込み
        shiiresaki_df = get_shiiresaki_bin_data(kojo)

        # 結果をCSVに保存（必要に応じて）
        master_of_shiiresaki.to_csv(f'temp_品番整備室仕入先マスター.csv', index=False, encoding='shift_jis')

        return master_of_shiiresaki

    # 最新の手配必要数テーブル（TDTHK_手配必要数管理）の読み込み
    # todo 対象工場の全品番を対象にしている（1日1回程度で読み込むため）
    def get_active_data_tehaisu(start_date, end_date, kojo):

        """
        手配必要数管理テーブル(TDTHK)から最新のデータを読み込む関数

        Parameters
        ----------
        start_date : str
            読み込み開始日(YYYYMM形式)
        end_date : str 
            読み込み終了日(YYYYMM形式)
        kojo : str
            対象工場コード

        Returns
        -------
        df : pandas.DataFrame
            手配必要数管理テーブルのデータ
            - 対象年月: 対象となる年月
            - 工場区分: 工場を識別するコード  
            - 計算区分: データのバージョン
            - 品番: 部品の品番
            - 受入場所: 部品の受入場所
            - 手配担当整備室: 担当する整備室
            - 手配区分: 手配の区分
            - 1~31: 1日目から31日目の手配数量
        """

        #　接続情報の読み込み
        dsn, dsn_uid, dsn_pwd, seibishitsu = get_connection_string_for_IBMDB(kojo)
        #print(seibishitsu)

        # データベースに接続
        conn = ibm_db.connect(dsn, dsn_uid, dsn_pwd)

        #　各年月と各整備室のユニークなバージョンを調べる
        sql = f"""
            SELECT DISTINCT 
                FDTHK01,
                FDTHK13,
                FDTHK03
            FROM awj.TDTHK
            WHERE FDTHK01 >= '{start_date}' 
                AND FDTHK01 <= '{end_date}' 
            ORDER BY FDTHK01, FDTHK13, FDTHK03
            """

        # クエリの実行
        stmt = ibm_db.exec_immediate(conn, sql)

        # 結果を取得してデータフレームに変換
        data = []
        row = ibm_db.fetch_assoc(stmt)
        while row:
            # 日本語のエンコーディングを考慮してデコード
            decoded_row = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in row.items()}
            data.append(decoded_row)
            row = ibm_db.fetch_assoc(stmt)

        # YYYYMM毎にユニークなverの列が計算された結果をデータフレームに保存する
        version_df = pd.DataFrame(data)
        # 先頭の0を削除した後に、空文字列になった場合（元々が'00'だった場合）は'0'に置き換える
        version_df['FDTHK03'] = version_df['FDTHK03'].str.lstrip('0').replace('', '0')

        # ユニークなバージョンを表示する
        #print(version_df)

        # 対象整備室の情報だけ抽出する
        version_df = version_df[version_df['FDTHK13'].isin(seibishitsu)]
        #print(version_df)

        # グループ化して最大値を取得
        # アルファベットを削除
        version_df['FDTHK03'] = version_df['FDTHK03'].str.replace('[a-zA-Z]', '', regex=True)
        # アルファベット以外の最大数値を取得（最新バージョンを取得）
        latest_version_df = version_df.groupby(['FDTHK01', 'FDTHK13'])['FDTHK03'].max().reset_index()
        # 再度0をつける（検索用)
        latest_version_df['FDTHK03'] = latest_version_df['FDTHK03'].astype(str)#文字に直す
        latest_version_df['FDTHK03'] = latest_version_df['FDTHK03'].where(latest_version_df['FDTHK03'].str.len() > 1, '0' + latest_version_df['FDTHK03'])

        # 最新バージョンを表示する
        print(latest_version_df)

        # ここからデータ抽出

        # 動的にSQLクエリを構築
        # '対象年月', '工場区分', '計算区分', '品番', '受入場所' ,'手配担当整備室','手配区分','計算開始日','計算終了日'
        columns = ["FDTHK01","FDTHK02","FDTHK03","FDTHK04","FDTHK11","FDTHK13","FDTHK16","FDTHK121","FDTHK122"]

        # 必要数は手メンテによる修正を反映した数値、計算数はACTIVEで純粋に計算した結果の数値
        # 計算数列の計算
        for i in range(20, 111, 3):
            columns.append(f"FDTHK{i}")

        ## リストとして定義
        fdthk13_values = seibishitsu
        # リストを文字列に変換（'1Y', '1Z' の形式に）
        fdthk13_str = "', '".join(fdthk13_values)

        # 組み合わせ条件を作成
        # 対象年月＆対象整備室を満たす
        conditions = " OR ".join([
            f"(FDTHK01 = '{row['FDTHK01']}' AND FDTHK03 = '{row['FDTHK03']}' AND FDTHK13 IN ('{fdthk13_str}'))"
            for _, row in latest_version_df.iterrows()
            ])

        # SQL
        sql = f"""SELECT {', '.join(columns)} FROM awj.TDTHK
                    WHERE {conditions}
                    """

        # クエリの実行
        stmt = ibm_db.exec_immediate(conn, sql)

        # 結果を取得してデータフレームに変換
        data = []
        row = ibm_db.fetch_assoc(stmt)
        while row:
            # 日本語のエンコーディングを考慮してデコード
            decoded_row = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in row.items()}
            data.append(decoded_row)
            row = ibm_db.fetch_assoc(stmt)

        df = pd.DataFrame(data)
        #print(df)

        # # 例：FDTHK13列のユニークな値を取得
        # unique_dates = df['FDTHK13'].unique()
        # # 値を確認
        # print(unique_dates)

        # カラム名を変更
        df.rename(columns={'FDTHK01': '対象年月'}, inplace=True)
        df.rename(columns={'FDTHK02': '工場区分'}, inplace=True)
        df.rename(columns={'FDTHK03': '計算区分'}, inplace=True)
        df.rename(columns={'FDTHK04': '品番'}, inplace=True)
        df.rename(columns={'FDTHK11': '受入場所'}, inplace=True)
        df.rename(columns={'FDTHK13': '手配担当整備室'}, inplace=True)
        df.rename(columns={'FDTHK16': '手配区分'}, inplace=True)
        df.rename(columns={'FDTHK121': '計算開始日'}, inplace=True)
        df.rename(columns={'FDTHK122': '計算終了日'}, inplace=True)

        # 10列目以降のカラム名を日付に置換
        start_index = 10  # 10列目から
        new_columns = list(df.columns[:start_index - 1])  # 8列目より前のカラム名は保持（データフレームは0スタートなので-1）

        # 8列目以降のカラム名を1から始まる整数に置換
        new_columns += [str(i - (start_index - 1) + 1) for i in range(start_index - 1, len(df.columns))]

        # 新しいカラム名を設定
        df.columns = new_columns

        # 結果をCSVに保存
        df.to_csv(f'{TEMP_OUTPUTS_PATH}/手配必要数.csv', index=False, encoding='shift-jis')
        print(df)
    
        return df

    # 手配必要数テーブルの縦展開
    def melt_active_data_tehaisu(df):

        """
        手配必要数データを日付ごとに展開して整形する関数

        Parameters
        ----------
        df : pandas.DataFrame
            手配必要数のデータフレーム
            - 対象年月: YYYYMM形式の年月
            - その他カラム: 工場区分、計算区分、品番など
            - 1~31: 各日の手配数量 

        Returns
        -------
        df_melted : pandas.DataFrame
            日付ごとに展開された手配必要数データ
            - 対象年月: 元の年月
            - 年: YYYY形式の年
            - 月: MM形式の月  
            - 日付: YYYY-MM-DD形式の日付
            - 日量数: その日の手配必要数量
            - その他カラム: 工場区分、計算区分、品番などの属性情報
        """

        df['年'] = df['対象年月'].astype(str).str[:4]  # 最初の4文字を年として抽出
        df['月'] = df['対象年月'].astype(str).str[4:6]  # 次の2文字を月として抽出

        # データフレームを縦に展開
        # id_varsで指定していない列を縦に展開
        df_melted = df.melt(id_vars=['対象年月','工場区分','計算区分','品番', '受入場所', '手配担当整備室', '手配区分', '年', '月'], 
                                    var_name='日付', value_name='日量数')

        # 日付列を整数型に変換（欠損値がある場合はそれを除外）
        df_melted['日付'] = df_melted['日付'].str.extract(r'(\d+)')

        # 欠損値を除外してから整数型に変換
        df_melted = df_melted.dropna(subset=['日付'])  # NaN行を削除
        df_melted['日付'] = df_melted['日付'].astype(int)

        # 年・月・日を結合し、不正な日付はNaTに変換
        df_melted['日付'] = pd.to_datetime(df_melted.apply(lambda row: f"{row['年']}-{row['月']}-{row['日付']}", axis=1), 
                                        errors='coerce')

        # 不正な日付（NaT）を除外
        # 不正な日付（例えば、2月30日など）を含む行を除外
        df_melted = df_melted.dropna(subset=['日付'])

        df_melted['品番'] = df_melted['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')

        # 結果を保存
        #df_melted.to_csv('sample_tehai2.csv', index=False, encoding='shift_jis')

        return df_melted

    # 手配運用情報（TDTUK_手配運用管理）テーブルの読み込み
    def get_active_data_tehaiunyo(kojo):

        """
        手配運用管理テーブル(TDTUK)から設定情報を読み込む関数

        Parameters
        ----------
        kojo : str
            対象工場コード

        Returns
        -------
        df : pandas.DataFrame
            手配運用管理テーブルのデータ
            - 品番: 部品の品番
            - 受入場所: 部品の受入場所(工場、整備室)
            - 設定日: レコード設定日付
            - 削除日: レコード削除日付  
            - 発注区分: 発注種類(0:別途発注、1:かんばん等)
            - 手配担当整備室: 担当整備室
            - 品名: 部品の名称
            - 通箱コード: 使用する通箱のコード
            - 通箱: 通箱の名称
            - 収容数: 通箱当たりの収容数
            - 基準在庫日数: 基準となる在庫日数
            - 基準在庫枚数: 基準となる在庫枚数
            - サイクル間隔: 納入サイクルの間隔
            - サイクル回数: 1日当たりの納入回数
            - サイクル情報: サイクル関連情報
            - 不良率: 不良品の発生率
            - 便平均上限振れ率: かんばん枚数調整用上限
            - 便平均下限振れ率: かんばん枚数調整用下限
            - MIN在庫日数: 最小在庫日数
            - MAX在庫日数: 最大在庫日数
            - 登録箱種: 登録されている箱の種類
        """

        # 列の値を変換する関数
        def convert_value(value):
            if value.startswith('0'):
                return int(value[1])
            else:
                return int(value) 

         #　接続情報の読み込み
        dsn, dsn_uid, dsn_pwd, seibishitsu = get_connection_string_for_IBMDB(kojo)

        # データベースに接続
        conn = ibm_db.connect(dsn, dsn_uid, dsn_pwd)

        ## リストとして定義
        fdtuk08_values = seibishitsu
        # リストを文字列に変換（'1Y', '1Z' の形式に）
        fdtuk08_str = "', '".join(fdtuk08_values)
        print(fdtuk08_str)

        # 動的にSQLクエリを構築
        # 品番、受入場所、サイクル間隔、サイクル回数、サイクル情報、不良率など
        columns = ["FDTUK01","FDTUK08","FDTUK16","FDTUK18","FDTUK19","FDTUK21","FDTUK28","FDTUK31","FDTUK37","FDTUK29","FDTUK30","FDTUK35","FDTUK36","FDTUK38","FDTUK39","FDTUK47","FDTUK61","FDTUK62","FDTUK64","FDTUK65","FDTUK80"]

        sql = f"""SELECT {', '.join(columns)} FROM awj.TDTUK
            WHERE  FDTUK08 IN  ('{fdtuk08_str}')
            """

        # クエリの実行
        stmt = ibm_db.exec_immediate(conn, sql)

        # 結果を取得してデータフレームに変換
        data = []
        row = ibm_db.fetch_assoc(stmt)
        while row:
            # 日本語のエンコーディングを考慮してデコード
            decoded_row = {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in row.items()}
            data.append(decoded_row)
            row = ibm_db.fetch_assoc(stmt)

        df = pd.DataFrame(data)

        # カラム名を変更
        df.rename(columns={'FDTUK01': '品番'}, inplace=True)
        df.rename(columns={'FDTUK08': '受入場所'}, inplace=True)#受入する工場、整備室
        df.rename(columns={'FDTUK16': '設定日'}, inplace=True)#当該ﾚｺｰﾄﾞを設定させた日付
        df.rename(columns={'FDTUK18': '削除日'}, inplace=True)#当該ﾚｺｰﾄﾞを削除させた日付
        df.rename(columns={'FDTUK19': '発注区分'}, inplace=True)#発注種類を表す区分（0:別途発注、1:かんばん、2:数量指示、3:資材量指示、6:調達指示輸入品等）
        df.rename(columns={'FDTUK21': '手配担当整備室'}, inplace=True)
        df.rename(columns={'FDTUK28': '品名'}, inplace=True)
        df.rename(columns={'FDTUK29': '通箱コード'}, inplace=True)
        df.rename(columns={'FDTUK30': '通箱'}, inplace=True)
        df.rename(columns={'FDTUK31': '収容数'}, inplace=True)
        df.rename(columns={'FDTUK35': '基準在庫日数'}, inplace=True)
        df.rename(columns={'FDTUK36': '基準在庫枚数'}, inplace=True)
        df.rename(columns={'FDTUK37': 'サイクル間隔'}, inplace=True)
        df.rename(columns={'FDTUK38': 'サイクル回数'}, inplace=True)
        df.rename(columns={'FDTUK39': 'サイクル情報'}, inplace=True)
        df.rename(columns={'FDTUK47': '不良率'}, inplace=True)
        df.rename(columns={'FDTUK61': '便平均上限振れ率'}, inplace=True)#かんばん平均枚数を調節するためのパラメータ
        df.rename(columns={'FDTUK62': '便平均下限振れ率'}, inplace=True)#かんばん平均枚数を調節するためのパラメータ
        df.rename(columns={'FDTUK64': 'MIN在庫日数'}, inplace=True)
        df.rename(columns={'FDTUK65': 'MAX在庫日数'}, inplace=True)
        df.rename(columns={'FDTUK80': '登録箱種'}, inplace=True)

        # 列の値を変換
        # 納入便の値を「0X」から「X」にする
        df['サイクル間隔'] = df['サイクル間隔'].apply(convert_value)
        df['サイクル回数'] = df['サイクル回数'].apply(convert_value)
        df['サイクル情報'] = df['サイクル情報'].apply(convert_value)

        df['品番'] = df['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')

        # 結果をCSVに保存（必要に応じて）
        #df.to_csv('sample_unyo.csv', index=False, encoding='shift_jis')
    
        return df

    # 手配必要数テーブル（縦展開）と手配運用情報テーブルを統合
    def merge_tehaisu_melt_df_and_tehaiunyo_df(tehaisu_melt_df, tehaiunyo_df):

        """
        手配必要数データと手配運用データを日付条件で結合する関数

        Parameters
        ----------
        tehaisu_melt_df : pandas.DataFrame
            日付ごとに展開された手配必要数データ
            - 日付: YYYY-MM-DD形式の日付
            - 品番: 部品の品番
            - 手配担当整備室: 手配担当の整備室
            
        tehaiunyo_df : pandas.DataFrame  
            手配運用管理データ
            - 品番: 部品の品番
            - 手配担当整備室: 手配担当の整備室
            - 設定日: レコード設定日付
            - 削除日: レコード削除日付

        Returns
        -------
        merged_df : pandas.DataFrame
            日付条件で結合された手配データ
            各日付に対して、その時点で有効な運用情報が紐づく
        """

        # 日付型の統一（datetime64[ns]型に変換）
        tehaisu_melt_df = tehaisu_melt_df.copy()  # 警告を避けるためにコピー
        tehaiunyo_df = tehaiunyo_df.copy()
        
        tehaisu_melt_df['日付'] = pd.to_datetime(tehaisu_melt_df['日付'])
        tehaiunyo_df['設定日'] = pd.to_datetime(tehaiunyo_df['設定日'], errors='coerce')
        
        # 削除日の処理（2099-12-31をデフォルトに）
        max_valid_date = pd.Timestamp('2099-12-31')
        tehaiunyo_df['削除日'] = pd.to_datetime(tehaiunyo_df['削除日'], errors='coerce').fillna(max_valid_date)

        # 結合のキーとなるカラムでソート
        tehaisu_melt_df = tehaisu_melt_df.sort_values(['品番', '手配担当整備室', '日付'])
        tehaiunyo_df = tehaiunyo_df.sort_values(['品番', '手配担当整備室', '設定日'])

        # NULL値や無効な日付を除外
        tehaiunyo_df = tehaiunyo_df.dropna(subset=['設定日'])
        tehaisu_melt_df = tehaisu_melt_df.dropna(subset=['日付'])

        # 結合処理
        merged_df = pd.merge(
            tehaisu_melt_df,
            tehaiunyo_df,
            on=['品番', '手配担当整備室'],
            how='left'
        )

        # 日付の条件でフィルタリング
        merged_df = merged_df[
            (merged_df['日付'] >= merged_df['設定日']) & 
            (merged_df['日付'] <= merged_df['削除日'])
        ]

        #todo まだダブりある（削除されていないのに新しい設定がある）
        # 各グループ内で最新の設定日を持つレコードを抽出
        idx = merged_df.groupby(['品番', '手配担当整備室', '日付'])['設定日'].transform('max') == merged_df['設定日']
        merged_df = merged_df[idx]

        # 品番情報統一
        merged_df['品番'] = merged_df['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')

        # コラム名変更
        merged_df = merged_df.rename(columns={'手配担当整備室':'整備室コード'})

        # 結果をCSVに保存（必要に応じて）
        #merged_df.to_csv('sample_merged.csv', index=False, encoding='shift_jis')

        return merged_df

    #　設計値MINとMAXを計算する
    def calclulate_min_max(hinban_info, start_datetime, end_datetime, kojo):

        # 品番情報設定
        hinban = hinban_info[0]
        seibishitsu = hinban_info[1]
        
        # 期間修正
        start_date = start_datetime[:7].replace('-', '')
        end_date = end_datetime[:7].replace('-', '')
        #print(start_date,end_date)
        
        # 手配必要数データ抽出
        tehaisu_df = get_active_data_tehaisu(start_date, end_date, kojo)
        #print(tehaisu_df)

        # 手配必要数（縦展開）
        tehaisu_melt_df = melt_active_data_tehaisu(tehaisu_df)
        #print(tehaisu_melt_df)

        # 手配運用情報データ抽出
        tehaiunyo_df = get_active_data_tehaiunyo(kojo)
        print(tehaiunyo_df)

        # 手配必要数と手配運用情報テーブルの統合
        # todo 手配必要数テーブル（TDTHK_手配必要数管理）の受入場所が手配担当整備室と一致しないため、手配担当整備室の方で統合
        tehai_merged_df = merge_tehaisu_melt_df_and_tehaiunyo_df(tehaisu_melt_df, tehaiunyo_df)
        print(tehai_merged_df)

        df_final = tehai_merged_df
        df_final['週番号'] = df_final['日付'].dt.isocalendar().week

        # # '手配区分'が4でないものを抽出（4は収容数が0のため）
        # df_final['手配区分'] = df_final['手配区分'].astype(int)
        # df_final = df_final[df_final['手配区分'] != 4]

        # 値クリーニング
        df_final['品番'] = df_final['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')
        df_final['整備室コード'] = df_final['整備室コード'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)

        # 週最大日量数の計算
        df_final['週最大日量数'] = df_final.groupby(['品番', '週番号'])['日量数'].transform('max')

        # # 週最大日量数（箱数）の計算
        df_final['週最大日量数'] = pd.to_numeric(df_final['週最大日量数'], errors='coerce')
        df_final['収容数'] = pd.to_numeric(df_final['収容数'], errors='coerce')
        df_final['週最大日量数（箱数）'] = df_final['週最大日量数']/df_final['収容数']

        # 設計値MIN（小数点以下を切り上げ）
        df_final['設計値MIN'] = np.ceil(0.1*(df_final['週最大日量数（箱数）']*df_final['サイクル間隔']*(1+df_final['サイクル情報'])/df_final['サイクル回数'])).astype(int)
        # 設計値MAX（小数点以下を切り上げ）
        df_final['便Ave'] = np.ceil(df_final['週最大日量数（箱数）']/df_final['サイクル回数']).astype(int)
        df_final['設計値MAX'] = df_final['設計値MIN'] + df_final['便Ave']

        # 日量箱数の計算
        df_final['日量数'] = pd.to_numeric(df_final['日量数'], errors='coerce')
        df_final['収容数'] = pd.to_numeric(df_final['収容数'], errors='coerce')
        df_final['日量数（箱数）'] = df_final['日量数']/df_final['収容数']

        # 保存
        df_final.to_csv(f'{TEHAI_DATA_PATH}', index=False, encoding='shift_jis')

        return df_final

    # 昼勤夜勤考慮
    def adjust_datetime_for_shift(x):
        if 0 <= x.hour < 8:
            # 日付をプラス１日して時間はそのまま
            return x + pd.Timedelta(days=1)
        else:
            # そのままの日付を返す
            return x
    
    # 品番情報設定
    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]

    # todo 半年分のデータを指定
    # 月単位で読み込む仕様になっている
    # データを読み込むようの期間決定
    end_datetime_temp = dt.datetime.now()#dt.datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
    start_datetime_temp = end_datetime_temp - dt.timedelta(days=180)
    # 開始と終了を決定
    start_datetime_for_get = start_datetime_temp.strftime('%Y-%m-%d %H:%M:%S')
    end_datetime_for_get = end_datetime_temp.strftime('%Y-%m-%d %H:%M:%S')#end_datetime
    #st.write(start_datetime_for_get,end_datetime_for_get)
    
    #　ファイルがない場合、新規作成する
    if (not Path(TEHAI_DATA_PATH).exists()):

        print("手配系のデータファイルがありません。新規で作成します")

        tehai_all_data = calclulate_min_max(hinban_info, start_datetime_for_get, end_datetime_for_get, kojo)

    # ファイルがある場合
    else:

        modification_time = os.path.getmtime(TEHAI_DATA_PATH)
        # UNIX時間からdatetimeに変換
        modification_date = dt.datetime.fromtimestamp(modification_time).date()
        print(f"最終更新日: {modification_date}")
        today = dt.date.today()
        print(f"本日: {today}")

        #　今日以外でデータ作成したなら、新規作成する
        # 1日1回だけ読み込むため
        if modification_date != today:
            
            print("手配データを新規作成します")
            tehai_all_data = calclulate_min_max(hinban_info, start_datetime_for_get, end_datetime_for_get, kojo)

        #　今日既に作成した場合は、既存ファイルを読み込む
        else:

            print("既存手配データを読み込みます")
            tehai_all_data = pd.read_csv(TEHAI_DATA_PATH, encoding='shift_jis')

    
    # 全品番のデータ確認
    # print(tehai_all_data)

    #　仕入先工場マスター取得
    master_of_shiiresaki = compute_master_of_shiiresaki(start_datetime_temp, end_datetime_temp, kojo)
    print(master_of_shiiresaki)
    #st.dataframe(master_of_shiiresaki)

    # 仕入先工場マスターの統合
    # 不等ピッチ情報を紐づけるために、仕入先情報が必要
    tehai_all_data = pd.merge(tehai_all_data, master_of_shiiresaki, on=['品番','整備室コード'], how='left')
    #tehai_all_data.to_csv('sample_merge2.csv', index=False, encoding='shift_jis')
    #print(tehai_all_data)

    # 不等ピッチデータの読み込み
    shiiresaki_bin_data = get_shiiresaki_bin_data(kojo)
    #st.write("不等ピッチの確認")
    #st.dataframe(shiiresaki_bin_data)

    # 空白やNaNを <NULL> に置き換える
    tehai_all_data['仕入先工場名'] = tehai_all_data['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
    shiiresaki_bin_data['仕入先工場名'] = shiiresaki_bin_data['仕入先工場名'].replace('', '<NULL>').fillna('<NULL>')
    
    # 不等ピッチデータの統合
    tehai_all_data = pd.merge(tehai_all_data, shiiresaki_bin_data[['仕入先名', '仕入先工場名','不等ピッチ係数（日）','不等ピッチ時間（分）', '納入先', '整備室コード', '納入LT(H)']], on=['仕入先名', '仕入先工場名','整備室コード'], how='left')
    #st.dataframe(tehai_all_data)

    # 設計値MAX更新
    tehai_all_data['設計値MAX'] = tehai_all_data['設計値MAX'] + tehai_all_data['週最大日量数（箱数）']*tehai_all_data['不等ピッチ係数（日）']

    # 対象品番抽出
    tehai_data_hinban = tehai_all_data[(tehai_all_data['品番'] == hinban)&(tehai_all_data['整備室コード'] == seibishitsu)]

    # '日付' をdatetime型に変換
    #tehai_data_hinban['日付'] = pd.to_datetime(tehai_data_hinban['日付'])
    tehai_data_hinban.loc[:, '日付'] = pd.to_datetime(tehai_data_hinban['日付'])
    #st.dataframe(tehai_data_hinban)

    # 日付を基準に〇時間ごとのデータに変換
    # 処理を軽くするため対象年月だけ抽出
    start_date_temp = start_datetime_for_get[:7].replace('-', '')
    end_date_temp = end_datetime_for_get[:7].replace('-', '')
    tehai_data_hinban[(tehai_data_hinban['対象年月'] == start_date_temp) | (tehai_data_hinban['対象年月'] == end_date_temp)]
    tehai_data_hinban.to_csv('sample_error_tehai.csv', index=False, encoding='shift_jis')
    # 期間修正
    #! 一致する品番ないと⇓でエラーでる
    tehai_data_hinban = tehai_data_hinban.set_index('日付').resample(time_granularity).ffill().reset_index()
    # if time_granularity == '15min':
    #     tehai_data_hinban = tehai_data_hinban.set_index('日付').resample(time_granularity).ffill().reset_index()
   
    # コラム名変更
    tehai_data_hinban = tehai_data_hinban.rename(columns={'日付': '日時'})

    # 昼勤夜勤の考慮
    # todo 昼勤夜勤の考慮で最初の日の0時から7時は無し
    tehai_data_hinban['日時'] = tehai_data_hinban['日時'].apply(adjust_datetime_for_shift)
    # 昼夜入れ替えをしたため、日時列を昇順にする
    tehai_data_hinban = tehai_data_hinban.sort_values('日時', ascending=True)

    # 月末までに最大日量MAX
    # 新しい列「月末までの最大日量数（箱数）」を初期化
    tehai_data_hinban['月末までの最大日量数'] = None
    # 処理対象列を決定
    target_column = '日量数'
    # 年・月ごとにグループ化して処理
    for (year, month), group in tehai_data_hinban.groupby(['年', '月']):
        for idx, row in group.iterrows():
            current_time = row['日時']
            # 現在時刻以降、同じ月内のデータをフィルタ
            subset = group[group['日時'] >= current_time]
            # 対象の「日量数（箱数）」の最大値を取得
            max_value = subset[target_column].max()
            # 値をセット
            tehai_data_hinban.loc[idx, '月末までの最大日量数'] = max_value
    tehai_data_hinban['月末までの最大日量数（箱数）'] = tehai_data_hinban['月末までの最大日量数']/tehai_data_hinban['収容数']

    # 実行結果保存
    #tehai_data_hinban.to_csv('sample_merge_hinban_hiru_yakin.csv', index=False, encoding='shift_jis')

    # 指定期間のデータ抽出
    tehai_data_hinban = tehai_data_hinban[(tehai_data_hinban['日時'] >= start_datetime) & (tehai_data_hinban['日時'] <= end_datetime)]

    print(tehai_data_hinban)

    # 返すのは、特定品番、特定期間のデータ
    return tehai_data_hinban

# MARK: 生産物流システムの着工データを読み込む
# todo データは手動で準備すること
# todo 構成品番とAT品番（完成品）を紐づけるマスター作成（今はCSVファイルの読み込み、しかも1Yのみ）
# todo テーブルデータ化されていない
#! 2分ごとに更新があるのでキャッシュを使わない
@st.cache_data
def compute_hourly_chakou_data_by_hinban(hinban_info, start_datetime, end_datetime):

    # todo HULFTDATAのBACKUPフォルダーに合わせて作成
    def process_multiple_months(base_path, year_months):

        """
        指定された年月のデータを処理する
        
        Parameters:
        base_path (str): データが格納されているベースディレクトリのパス
        year_months (list): 処理したい年月のリスト (例: ['202501', '202502'])
        
        Returns:
        pandas.DataFrame: 処理結果
        """

        def create_dataframe_from_directory(base_dir):

            """
            Returns:
            pandas.DataFrame: 以下のような形式の処理結果

            datetime	data
            0	2025-01-05 16:27:00	P8 202501067517623061046021 020250105163107
            1	2025-01-05 16:28:00	P8 202501067517633061046021 020250105163206
            2	2025-01-05 16:29:00	P8 202501067517643061046021 020250105163305
            3	2025-01-05 16:31:00	P8 202501067517653061046021 020250105163455
            4	2025-01-05 16:32:00	P8 202501067517663061046021 020250105163606
            """

            # 結果を格納するリスト
            all_data = []
            
            # YYYYMMの下の日付フォルダーをすべて取得
            day_folders = glob.glob(os.path.join(base_dir, "*"))
            
            for day_folder in day_folders:
                print(day_folder)
                
                # 日付を取得 (フォルダ名から)
                day = os.path.basename(day_folder)
                
                # 各日付フォルダー内の全ファイルを取得
                files = glob.glob(os.path.join(day_folder, "*"))
                
                for file in files:
                    filename = os.path.basename(file)
                    if '_' in filename:  # ファイル名に_が含まれているか確認
                        # 時刻を取得 (_の後ろ4文字)
                        time_str = filename.split('_')[1][:4]
                        
                        # YYYYMMDDを作成
                        year_month = os.path.basename(base_dir)
                        date_str = f"{year_month}{day}"
                        
                        # datetime作成
                        datetime_val = dt.datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
                        
                        # ファイルを読み込む
                        try:
                            with open(file, 'r') as f:
                                data = f.read().strip()
                            
                            # データを追加
                            all_data.append({
                                'datetime': datetime_val,
                                'data': data
                            })
                        except Exception as e:
                            print(f"Error reading file {file}: {e}")
            
            # リストをデータフレームに変換
            df = pd.DataFrame(all_data)
            
            # datetime列でソート
            df = df.sort_values('datetime')
            
            return df

        def split_data_columns(df):

            """
            Returns:
            pandas.DataFrame: 以下のような形式の処理結果

            datetime	data1	data2	data3	data4	data5	data6	data7	data8	data9	...	data15	data16	data17	data18	data19	data20	data21	data22	data23	data24
            0	2025-01-05 16:27:00	P8 202501067517623061046021 020250105163107	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            1	2025-01-05 16:28:00	P8 202501067517633061046021 020250105163206	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            2	2025-01-05 16:29:00	P8 202501067517643061046021 020250105163305	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            3	2025-01-05 16:31:00	P8 202501067517653061046021 020250105163455	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            4	2025-01-05 16:32:00	P8 202501067517663061046021 020250105163606	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
            995	2025-01-08 06:15:00	P8 202501087732653061058010 020250108061833	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            996	2025-01-08 06:16:00	P8 202501087732663061058010 020250108061858	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            997	2025-01-08 06:17:00	P8 202501087733613061042021 020250108061949	P8 202501087733623061042021 020250108062040	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            998	2025-01-08 06:19:00	P8 202501087733633061042021 020250108062239	None	None	None	None	None	None	None	None	...	None	None	None	None	None	None	None	None	None	None
            999	2025-01-08 06:20:00	P8 202501087733643061042021 020250108062304	None	
            """

            # データを改行で分割して複数列に展開
            split_data = df['data'].str.split('\n', expand=True)
            
            # 新しい列名を作成（data1, data2, ...）
            new_columns = [f'data{i+1}' for i in range(len(split_data.columns))]
            split_data.columns = new_columns
            
            # 元のデータフレームからdata列を削除し、分割したデータを結合
            df = df.drop('data', axis=1)
            df = pd.concat([df, split_data], axis=1)
            
            # 空の文字列をNaNに変換（オプション）
            df = df.replace(r'^\s*$', pd.NA, regex=True)
            
            return df

        def process_dataframe(df):

            """
            datetime_H	306100R032	3061040012	3061042011	3061042021	3061042031	3061042032	3061046011	3061046021	3061046022	3061046030	3061048011	3061048012	3061058010	3061058011	7552029242	7552029252
            0	2025-01-05 16:00:00	0	0	0	0	0	0	0	5	0	0	0	0	0	0	0	0
            1	2025-01-06 07:00:00	0	0	6	0	0	0	0	0	0	0	0	0	0	0	0	0
            2	2025-01-06 08:00:00	0	0	0	1	6	0	0	0	0	6	0	0	0	0	0	0
            3	2025-01-06 09:00:00	0	0	6	5	6	0	6	0	0	0	0	0	6	0	0	0
            4	2025-01-06 10:00:00	0	0	6	12	6	0	0	0	0	0	0	0	6	0	0	0
            ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
            424	2025-01-31 18:00:00	0	0	6	8	0	12	0	0	0	6	0	18	0	3	0	0
            425	2025-01-31 19:00:00	0	0	0	6	0	6	6	0	0	0	0	5	0	3	0	0
            426	2025-01-31 22:00:00	0	6	6	12	0	6	0	0	0	0	0	13	0	6	0	0
            427	2025-01-31 23:00:00	0	0	6	12	0	12	0	0	6	0	0	12	0	0	0	0
            428	2025-02-01 00:00:00	0	0	0	0	0	0	0	
            """

            # 全てのdata列に対して処理を行う
            data_columns = [col for col in df.columns if col.startswith('data')]
            
            # 結果を格納するリスト
            results = []
            
            for col in data_columns:
                # 空白を削除
                df[col] = df[col].str.strip()
                
                # 空でないデータのみを処理
                mask = df[col].notna()
                
                if mask.any():
                    # 日時文字列を抽出（後ろ14文字）
                    datetime_str = df.loc[mask, col].str[-14:]
                    
                    # 品番を抽出（17文字目から26文字目）
                    product_code = df.loc[mask, col].str[18:29]
                    
                    # 日時文字列をdatetime型に変換
                    datetime_data = pd.to_datetime(datetime_str, format='%Y%m%d%H%M%S')
                    
                    # データを一時的なデータフレームに格納
                    temp_df = pd.DataFrame({
                        'datetime': datetime_data,
                        'product_code': product_code
                    })
                    
                    results.append(temp_df)
            
            # 全てのデータを結合
            result_df = pd.concat(results, ignore_index=True)
            
            # 1時間単位で集計
            # datetime列を時間単位に切り捨て
            result_df['datetime_H'] = result_df['datetime'].dt.floor('h')
            
            # 品番ごとの出現回数を集計
            pivot_df = pd.pivot_table(
                result_df,
                index='datetime_H',
                columns='product_code',
                aggfunc='size',
                fill_value=0
            ).reset_index()
            
            return pivot_df, result_df

        all_results = []
        
        for year_month in year_months:
            # 年月のディレクトリパスを作成
            month_dir = os.path.join(base_path, year_month)
            
            if os.path.exists(month_dir):
                # 既存の処理を実行
                df = create_dataframe_from_directory(month_dir)
                df_split = split_data_columns(df)
                result, _ = process_dataframe(df_split)
                
                all_results.append(result)
            else:
                print(f"Directory not found: {month_dir}")
        
        # 全ての結果を結合
        if all_results:
            final_result = pd.concat(all_results, ignore_index=True)
            # datetime_Hでソート
            final_result = final_result.sort_values('datetime_H')
            return final_result
        # データがない場合
        else:
            return pd.DataFrame()

    # 対象AT品番抽出
    def extract_target_columns(df, unique_list):

        """
        指定されたリストの値に一致する列を抽出する
        
        Parameters:
        df: データフレーム
        unique_list: 抽出したい列名のリスト
        
        Returns:
        選択された列のみを含むデータフレーム
        """

        # リストの値に一致する列を選択
        selected_columns = ['日時'] + [col for col in df.columns if col in unique_list]
        
        # 選択された列のデータフレームを返す
        return df[selected_columns]

    # 対象合計品番
    def add_total_production(df):

        """
        datetime_H列を除く全ての列の合計を計算し、'生産台数'列として追加
        """

        # datetime_H列以外の列を選択
        numeric_columns = [col for col in df.columns if col != '日時']
        
        # 合計を計算して新しい列として追加
        df['生産台数'] = df[numeric_columns].sum(axis=1)
        
        return df


    # 品番情報設定
    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]

    folder_path_for_reader = r'\\172.16.136.121\d\HULFTDATA\S8F010\BACKUP'
    folder_path_for_writer = SEIBUTSU_BASE_PATH
    with open(SEIBUTSU_ZIKKOU_TIME, 'r') as f:
        date_str = f.read().strip()
        # 文字列をdatetime型に変換
        last_date = dt.datetime.strptime(date_str, "%Y%m%d")
    copy_differences(folder_path_for_reader, folder_path_for_writer, last_date)#!消すと早くなる

    # マスター品番を読み込む
    #file_path = "C:\\Users\\1082794-Z100\\Documents\\Model\\zaiko\\生データ\\T403部品品番-AT品番\\マスター品番.csv"
    file_path = "生産物流システム_流動機種_マスター品番.csv"
    master_df = pd.read_csv(file_path, encoding='shift_jis')

    # 対応するAT品番抽出
    #todo 1Yのみ
    master_df = master_df[master_df['品番'] == hinban]
    unique_AT_hinban = master_df['AT品番'].unique()

    # データ計算
    #year_months = ['202410','202411','202412','202501', '202502','202503','202504']  # 処理したい年月のリスト

    def generate_past_six_months_list(end_date=None):
        # 終了日を指定しない場合は現在の日付を使用
        if end_date is None:
            end_date = dt.datetime.now()
        elif isinstance(end_date, str):
            # YYYYMM形式の文字列が渡された場合
            end_date = dt.datetime.strptime(end_date, "%Y%m")
        
        year_months = []
        
        # 指定された月から5ヶ月前までの6ヶ月分を生成
        for i in range(5, -1, -1):  # 5から0まで逆順
            # 月を計算
            year = end_date.year + ((end_date.month - i - 1) // 12)
            month = ((end_date.month - i - 1) % 12) + 1
            year_months.append(f"{year}{month:02d}")
        
        print(f"Generated months from {year_months[0]} to {year_months[-1]}")
        return year_months

    # 現在の日付から過去6ヶ月を生成
    year_months = generate_past_six_months_list()
    print("生物データ",year_months)

    result = process_multiple_months(SEIBUTSU_BASE_PATH, year_months)

    # 列名修正、すべての空白を削除（列名の中間の空白も含む）
    result.columns = result.columns.str.replace(' ', '')

    result = result.rename(columns={
        'datetime_H':'日時',
        '3061042011' : '30610ECB014',
        '3061042021' : '30610ECC013',
        '3061042031' : '30610ECD018',
        '3061042032' : '30610ECD027',
        '3061048011' : '30610ECD022',
        '3061048012' : '30610ECD030',
        '3061046011' : '30610ECB016',
        '3061046021' : '30610ECD020',
        '3061046022' : '30610ECD032',
        '3061040011' : '30610ECD026',
        '3061040012' : '30610ECD029',
        '3061058010' : '30610ECD025',
        '3061058011' : '30610ECD031',
        '3061046030' : '30610ECB017',
        '30610B5010' : '30610ECE001',
        '7552029252' : '30610ECB018',
        '7552029242' : '30610ECE009',
        '306100R011' : '30610ECB015',
        '306100R021' : '30610ECC014',
        '306100R031' : '30610ECD019',
        '306100R032' : '30610ECD028',
        '3061042040' : '30610ECF001',
        '3061042050' : '30610ECF004'
    })

    #print(unique_AT_hinban)

    # 使用
    seisan_buturyu_data = extract_target_columns(result, unique_AT_hinban)

    seisan_buturyu_data = add_total_production(seisan_buturyu_data)

    return seisan_buturyu_data

#MARK: あるフォルダーの内容を別フォルダーに差分コピー
#! 生物データをコピーする用
@st.cache_data
def copy_differences(source_path, target_path, last_date):

    def is_valid_folder(folder_path):
        try:
            # フォルダー名がYYYYMM形式かチェック
            folder_name = folder_path.name
            year_month = dt.datetime.strptime(folder_name, "%Y%m")
            
            # サブフォルダー（日付）を取得
            for day_folder in folder_path.iterdir():
                if day_folder.is_dir():
                    try:
                        # 日付フォルダー名（DD）を解析
                        day = int(day_folder.name)
                        if 1 <= day <= 31:
                            # 完全な日付を作成
                            folder_date = dt.datetime(year_month.year, year_month.month, day)
                            # 指定した日付以降であればTrueを返す
                            if folder_date >= BASE_DATE:
                                return True
                    except ValueError:
                        continue
            return False
        except ValueError:
            return False

    def get_latest_month_folder(path):
        latest_month = None
        latest_folder = None
        
        for item in path.iterdir():
            if item.is_dir() and is_valid_folder(item):
                try:
                    folder_date = dt.datetime.strptime(item.name, "%Y%m")
                    if latest_month is None or folder_date > latest_month:
                        latest_month = folder_date
                        latest_folder = item
                except ValueError:
                    continue
        
        return latest_folder

    
    # 基準日時を設定
    BASE_DATE = last_date  # 年月日を指定
    
    # 読み込み用フォルダー
    source = Path(source_path)

    # 書き込み用フォルダー
    target = Path(target_path)
    
    # ターゲットディレクトリが存在しない場合は作成
    if not target.exists():
        target.mkdir(parents=True)

    # 最新の月のフォルダーを取得
    latest_month_folder = get_latest_month_folder(source)
    if not latest_month_folder:
        print("No valid month folders found")
        return

    print(f"Processing files from month: {latest_month_folder.name}")

    # 日付の比較関数
    def is_after_base_date(file_path):
        try:
            # パスから年月日を抽出
            year_month = latest_month_folder.name  # YYYYMM
            # 親フォルダーの名前（DD）を取得
            day = file_path.parent.name
            # 完全な日付文字列を作成
            date_str = f"{year_month}{day}"
            file_date = dt.datetime.strptime(date_str, "%Y%m%d")
            return file_date >= BASE_DATE
        except ValueError:
            return False

    # 最新の月フォルダー配下の全ファイルとフォルダーを処理
    for sub_item in latest_month_folder.rglob('*'):

        # 基準日以降のファイルのみ処理
        if sub_item.is_file() and not is_after_base_date(sub_item):
            continue

        # ソースパスからの相対パス
        relative_path = sub_item.relative_to(source)
        target_item = target / relative_path

        if sub_item.is_dir():

            # フォルダーの場合
            if not target_item.exists():
                # 差分フォルダーを作成
                shutil.copytree(str(sub_item), str(target_item))
                print(f"Copied folder: {relative_path}")
        
        elif sub_item.is_file():
            if not target_item.exists():
                # 差分ファイルをコピー
                target_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(sub_item), str(target_item))
                print(f"Copied file: {relative_path}")
            else:
                # ファイルが存在する場合、内容を比較
                if not filecmp.cmp(str(sub_item), str(target_item), shallow=False):
                    shutil.copy2(str(sub_item), str(target_item))
                    print(f"Updated file: {relative_path}")

    # 現在の日付を取得してYYYYMMDD形式に変換、最終実行日を記録
    current_date = dt.datetime.now().strftime("%Y%m%d")
    with open(SEIBUTSU_ZIKKOU_TIME, 'w') as f:
        f.write(current_date)

# MARK: 単独テスト用
if __name__ == "__main__":

    kojo = 'anjo1'
    hinban_info = ['351060LC040', '1Z']#第一工場
    #hinban_info = ['019128GA010', '2S']#第二工場
    start_datetime = '2024-07-01 01:00:00'
    end_datetime = '2025-04-10 09:00:00'
    selected_datetime = '2025-03-12 09:00:00'
    flag_useDataBase = 1
    target_column = '順立装置出庫日時'
    #target_column = '検収日時'
    time_granularity = '1h'

    #! 仕入先ダイヤ情報抽出
    # df = get_shiiresaki_bin_data(kojo)
    # print(df)

    #! 品番、整備室、仕入先名、仕入先工場名のマスター作成
    # df = compute_master_of_shiiresaki(start_datetime, end_datetime,kojo)
    # print(df)

    # #! 自動ラックの在庫データの読み込みテスト
    # df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info, start_datetime, end_datetime, flag_useDataBase, kojo)
    # print(df)

    # #! 関所毎のかんばんデータの読み込みテスト
    target_column = "順立装置出庫日時"
    time_granularity = 'h'
    df, df2 = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
    print(df)
    print(df2)

    #! 手配データの読み込みテスト
    # df = compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
    # print(df)

    #! 着工データの読み込み
    # seisan_buturyu_data = compute_hourly_chakou_data_by_hinban(hinban_info, start_datetime, end_datetime)
    # print(seisan_buturyu_data)

    #! 異常お知らせ版の残業時間データを読み込む
    # day_col='計画(昼)'
    # night_col='計画(夜)'
    # df = get_kado_schedule_from_172_20_113_185(start_datetime, end_datetime, day_col, night_col)
    # print(df)

    #! 品番情報提示
    # df = get_hinban_info_detail(hinban_info, selected_datetime, flag_useDataBase, kojo)
    # print(df)

    # #! あるフォルダーの内容をコピー
    # folder_path_for_reader = r'\\172.16.136.121\d\HULFTDATA\S8F010\BACKUP'
    # #folder_path_for_reader = "C:\\Users\\1082794-Z100\\Documents\\Model\\zaiko\\生データ\\202501-02_P8 1"
    # #folder_path_for_writer = "C:\\Users\\1082794-Z100\\Documents\\Model\\zaiko\\生データ\\202501-02_P8 1"
    # folder_path_for_writer = "../../data/生産物流システム"
    # file_path = "../../data/生産物流システム/最終実行日.txt"
    # with open(file_path, 'r') as f:
    #     date_str = f.read().strip()
    #     # 文字列をdatetime型に変換
    #     last_date = dt.datetime.strptime(date_str, "%Y%m%d")
    # copy_differences(folder_path_for_reader, folder_path_for_writer, last_date)