import os
import pandas as pd
import warnings
import numpy as np
import pandas as pd
#%matplotlib inline#Jupyter Notebook 専用のマジックコマンド
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


def calculate_hourly_counts(df, part_number, time_col, start_date, end_date):
    
    """
    指定された品番のイベント時間（1時間）ごとのかんばん数、納入便、平均検収時間を計算する関数
    
    Parameters:
    df (pd.DataFrame): データフレーム
    part_number (str): 対象品番
    time_col (str): イベント時間を表す列名
    start_date (str): 開始日付（YYYY-MM-DD形式）
    end_date (str): 終了日付（YYYY-MM-DD形式）
    
    Returns:
    pd.DataFrame: 1時間ごとのかんばん数と納入便および平均検収時間
    """
    
    # 品番でフィルタリング
    filtered_df = df[df['品番'] == part_number].copy()
    
    # イベント時間を1時間単位に丸める
    filtered_df['イベント時間'] = pd.to_datetime(filtered_df[time_col], errors='coerce').dt.floor('H')
    
    # 指定された期間内のイベント時間ごとのかんばん数を計算
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # 非稼動日の時間を削除する関数
    # 土曜日9時から月曜7時までの時間を除外
    def is_excluded_time(timestamp):
        #土曜日（weekday() == 5）の9時以降であれば、True を返す
        if timestamp.weekday() == 5 and timestamp.hour >= 9:
            return True
        #日曜日（weekday() == 6）であれば、True を返す。
        if timestamp.weekday() == 6:
            return True
        #月曜日（weekday() == 0）の8時未満であれば、True を返す
        if timestamp.weekday() == 0 and timestamp.hour < 8:
            return True
        return False
    
    # 除外された時間をフィルタリング
    date_range = [dt for dt in date_range if not is_excluded_time(dt)]
    filtered_df = filtered_df[filtered_df['イベント時間'].isin(date_range)]
    
    # 非稼動日時間を削除済み
    hourly_counts = filtered_df.groupby('イベント時間').size().reindex(date_range, fill_value=0)
    
    delivery_totals = []
    reception_times = []
    if time_col == '検収日時':
        delivery_col = '納入便'
        # 納入便をint型に変換
        filtered_df[delivery_col] = filtered_df[delivery_col].astype(int)
        # イベント時間ごとの納入便の合計を計算
        delivery_totals = filtered_df.groupby('イベント時間')[delivery_col].sum().reindex(date_range, fill_value=0)
        # 結果データフレームを作成して納入便の合計を追加
        delivery_totals = delivery_totals / hourly_counts # 納入便数の計算
        delivery_totals = delivery_totals.fillna(0)  # NaNを0に置き換え
        delivery_totals = delivery_totals.astype(int) # int型に変換
        # イベント時間ごとの平均検収時間を計算
        reception_times = filtered_df.groupby('イベント時間')[time_col].apply(lambda x: pd.to_datetime(x.dt.time.astype(str)).mean().time()).reindex(date_range)
        reception_times = reception_times.fillna('00:00:00')  # NaNを00:00:00に置き換え
    
    return hourly_counts, delivery_totals, reception_times

def calculate_business_time_base(row, order_datetime, warehouse_datetime):
    
    # 発注日時または順立装置入庫日時がNaTの場合、NaNを返す
    if pd.isna(order_datetime) or pd.isna(warehouse_datetime):
        return np.nan

    # 全体の時間差を計算
    total_time = warehouse_datetime - order_datetime
    current_datetime = order_datetime

    # 発注日時から順立装置入庫日時までの期間を1時間ごとに進める
    while current_datetime < warehouse_datetime:
        # 土曜の9時から月曜の7時までの間の時間を除去
        if current_datetime.weekday() == 5 and current_datetime.hour >= 9:  # 土曜日の9時以降
            next_monday = current_datetime + pd.Timedelta(days=2)
            next_monday = next_monday.replace(hour=7, minute=0, second=0)
            if next_monday < warehouse_datetime:
                total_time -= next_monday - current_datetime
                current_datetime = next_monday
            else:
                total_time -= warehouse_datetime - current_datetime
                break
        else:
            # 1時間進める
            current_datetime += pd.Timedelta(hours=1)
            if current_datetime.weekday() == 5 and current_datetime.hour == 9:
                # 進めた結果が土曜日の9時に到達した場合
                next_monday = current_datetime + pd.Timedelta(days=2)
                next_monday = next_monday.replace(hour=7, minute=0, second=0)
                if next_monday < warehouse_datetime:
                    total_time -= next_monday - current_datetime
                    current_datetime = next_monday
                else:
                    total_time -= warehouse_datetime - current_datetime
                    break

    # 日数として計算し、小数点形式に変換
    return total_time.total_seconds() / (24 * 3600)

# ビジネスタイムの差分を計算する関数
def calculate_business_time_order(row):
    
    order_datetime = row['発注日時']
    warehouse_datetime = row['順立装置入庫日時']

    # 日数として計算し、小数点形式に変換
    return calculate_business_time_base(row, order_datetime, warehouse_datetime)

def calculate_business_time_reception(row):
    
    order_datetime = row['検収日時']
    warehouse_datetime = row['順立装置入庫日時']

    # 日数として計算し、小数点形式に変換
    return calculate_business_time_base(row, order_datetime, warehouse_datetime)

def calculate_median_lt(product_code,df):
    
    """
    指定された品番の発注〜順立装置入庫LTの中央値を計算する関数。

    Parameters:
    product_code (str): 品番

    Returns:
    float: 発注〜順立装置入庫LTの中央値
    """
    
    # 指定された品番のデータをフィルタリング
    filtered_df = df[df['品番'] == product_code]

    # フィルタリングされたデータが空かどうかをチェック
    if filtered_df.empty:
        return None

    # '発注〜順立装置入庫LT'
    filtered_df['発注〜順立装置入庫LT（非稼動日削除）'] = filtered_df.apply(calculate_business_time_order, axis=1)
    filtered_df['検収〜順立装置入庫LT（非稼動日削除）'] = filtered_df.apply(calculate_business_time_reception, axis=1)

    # フィルタリングされたデータが空かどうかを再チェック
    if filtered_df.empty:
        return None

    # '発注〜順立装置入庫LT'の中央値を計算
    median_value_order = filtered_df['発注〜順立装置入庫LT（非稼動日削除）'].median()
    median_value_reception = filtered_df['検収〜順立装置入庫LT（非稼動日削除）'].median()
    
    #確認用
    #print(filtered_df[['発注日時','順立装置入庫日時','発注〜順立装置入庫LT','発注〜順立装置入庫LT（非稼動日削除）']].head(50))

    return median_value_order, median_value_reception

def find_best_lag_range(hourly_data, hourly_target, min_lag, max_lag, label_name):
    
    """
    指定された範囲内でイベントの最適な遅れ範囲を見つける関数。
    
    Parameters:
    hourly_data (pd.Series): 1時間ごとのイベント数
    hourly_target (pd.Series): 1時間ごとのターゲットイベント数
    min_lag (int): 最小遅れ時間
    max_lag (int): 最大遅れ時間
    label_name (str): ローリング平均のラベル名
    
    Returns:
    tuple: (最適な相関係数, 最適な遅れ範囲の開始, 最適な遅れ範囲の終了)
    """
    
    best_corr = -1
    best_range_start = None
    best_range_end = None
    
    # 最適な遅れ範囲を探索
    for range_start in range(min_lag, max_lag - 1):
        for range_end in range(range_start + 2, max_lag + 1):
            lag_features = pd.DataFrame(index=hourly_data.index)
            
            # 遅れ範囲に基づいてイベント数のローリング平均を計算
            lag_features[f'{label_name}_lag_{range_start}-{range_end}'] = hourly_data.shift(range_start).rolling(window=range_end - range_start + 1).mean()
            lag_features['入庫かんばん数（t）'] = hourly_target
            
            # 欠損値を除去
            lag_features = lag_features.dropna()
            
            # 相関を計算し、最適な範囲を見つける
            corr = lag_features[f'{label_name}_lag_{range_start}-{range_end}'].corr(lag_features['入庫かんばん数（t）'])
            if corr > best_corr:
                if (range_end-range_start)<5:#時間幅を設定
                    best_corr = corr
                    best_range_start = range_start
                    best_range_end = range_end
    
    return best_corr, best_range_start, best_range_end

def create_lagged_features(hourly_data, hourly_target, hourly_leave, best_range_start, best_range_end, label_name, delivery_info, reception_times):
    
    """
    最適な遅れ範囲に基づいてイベントのローリング合計を計算し、説明変数として追加する関数。
    
    Parameters:
    hourly_data (pd.Series): 1時間ごとのイベント数
    hourly_target (pd.Series): 1時間ごとのターゲットイベント数
    best_range_start (int): 最適な遅れ範囲の開始（時間単位）
    best_range_end (int): 最適な遅れ範囲の終了（時間単位）
    label_name (str): ローリング平均のラベル名
    delivery_info (pd.Series): 1時間ごとの納入便数
    reception_times (pd.Series): 1時間ごとの平均検収時間（HH:MM:SS形式）
    
    Returns:
    pd.DataFrame: ローリング合計を含むデータフレーム
    """
    
    lag_features = pd.DataFrame(index=hourly_data.index)
    
    # 最適な遅れ範囲に基づいてローリング平均を計算し、説明変数として追加
    lag_features[f'{label_name}（t-{best_range_start}~t-{best_range_end}）'] = hourly_data.shift(best_range_start).rolling(window=best_range_end - best_range_start + 1).sum()
    lag_features['入庫かんばん数（t）'] = hourly_target
    lag_features['出庫かんばん数（t）'] = hourly_leave
    
    # 納入かんばん数を与えた場合は、平均納入時間（XX:XX）を計算
    if label_name == '納入かんばん数':
        lag_features[f'納入便（t-{best_range_start}~t-{best_range_end}）'] = delivery_info.shift(best_range_start).rolling(window=best_range_end - best_range_start + 1).max()
        
        # 時刻データを秒に変換
        reception_times = pd.to_datetime(reception_times, format='%H:%M:%S', errors='coerce')
        seconds = reception_times.dt.hour * 3600 + reception_times.dt.minute * 60 + reception_times.dt.second

        # シフトとローリング合計を計算
        shifted_seconds = seconds.shift(best_range_start).rolling(window=best_range_end - best_range_start + 1).sum()

        # 秒を時間に戻す関数
        def seconds_to_time(seconds):
            if pd.isna(seconds):
                return pd.NaT
            total_seconds = int(seconds)
            hours = total_seconds // 3600
            if hours >= 24:
                hours = hours -24
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f'{hours:02}:{minutes:02}:{seconds:02}'

        # シフトされた秒を時間に戻してデータフレームに追加
        shifted_times = shifted_seconds.apply(seconds_to_time)
        lag_features[f'平均納入時間（t-{best_range_start}~t-{best_range_end}）'] = shifted_times

        # 欠損値を除去
        lag_features = lag_features.dropna()

    return lag_features

def add_part_supplier_info(df, lagged_features, part_number):
    
    """
    元のデータフレームから品番と仕入先名を抽出し、
    lagged_featuresに品番と仕入先名を結合する関数。

    Parameters:
    df (DataFrame): 元のデータフレーム
    lagged_features (DataFrame): ラグ特徴量のデータフレーム
    part_number (str): 品番

    Returns:
    DataFrame: 品番と仕入先名が追加されたlagged_features
    """
    
    # 元のデータフレームから該当品番、仕入先、仕入先工場列を抽出
    part_supplier_info = df[['品番', '仕入先名', '仕入先工場名']].drop_duplicates()
    
    #-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★
    # 特定の文字列を含む行を削除
    # 仕入先工場名は何もないものと<NULL>が混在しているものがある
    part_supplier_info = part_supplier_info[~part_supplier_info['仕入先名'].str.contains('< NULL >', na=False)].dropna(subset=['仕入先工場名'])
    #filtered_df = df[df['品番'] == part_number]
    #print(filtered_df)#仕入先名に<NULL>がある
    #-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★-★

    # indexを日付に設定
    lagged_features = lagged_features.reset_index()
    lagged_features = lagged_features.rename(columns={'イベント時間': '日時'})

    # lagged_features に品番と仕入先名を結合
    lagged_features['品番'] = part_number  # 品番を追加
    lagged_features = lagged_features.merge(part_supplier_info, on='品番', how='left')

    return lagged_features

# 特定の言葉を含む列名を見つける関数
def find_columns_with_word_in_name(df, word):
    columns_with_word = [column for column in df.columns if word in column]
    return ', '.join(columns_with_word)

def calculate_elapsed_time_since_last_dispatch(lagged_features):
    
    """
    出庫からの経過行数および出庫間隔の中央値を計算する関数。
    
    Parameters:
    lagged_features (pd.DataFrame): 出庫データを含むデータフレーム
    
    Returns:
    pd.DataFrame: 出庫からの経過行数を含むデータフレーム
    int: 出庫間隔の中央値（行数）
    """
    
    lagged_features['過去の出庫からの経過行数'] = None
    last_dispatch_index = None

    for index, row in lagged_features.iterrows():
        if last_dispatch_index is not None:
            elapsed_rows = index - last_dispatch_index
            lagged_features.at[index, '過去の出庫からの経過行数'] = elapsed_rows
        if row['出庫かんばん数（t）'] > 0:
            last_dispatch_index = index
    
    # 表示のための調整（最初の行の '過去の出庫からの経過行数' 列は None にする）
    lagged_features.at[0, '過去の出庫からの経過行数'] = None
    
    # 出庫かんばん数が1以上の行インデックスを特定
    dispatch_indices = lagged_features[lagged_features['出庫かんばん数（t）'] >= 1].index

    # 出庫間隔を計算
    dispatch_intervals = dispatch_indices.to_series().diff().dropna()

    # 出庫間隔の中央値を計算
    median_interval = dispatch_intervals.median()
    
    #print("出庫間隔の中央値（行数）:", median_interval)
    
    return lagged_features, int(median_interval)

# timedeltaをHH:MM:SS形式に変換する関数
def timedelta_to_hhmmss(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# 仕入先便到着フラグを設定する関数
def set_arrival_flag(row, columns_delibery_num,columns_delibery_times):
    #print(row['早着'],row[columns_delibery_times],columns_delibery_num,columns_delibery_times)
    early = pd.to_datetime(row['早着'], format='%H:%M:%S').time()
    on_time = pd.to_datetime(row['定刻'], format='%H:%M:%S').time()
    late = pd.to_datetime(row['遅着'], format='%H:%M:%S')
    #print(row[columns_delibery_times])
    delivery_time = pd.to_datetime(row[columns_delibery_times], format='%H:%M:%S').time()
    
    late_plus_2hrs = (late + timedelta(hours=2)).time()
     
    if row[columns_delibery_num] != 0:
        if early < delivery_time < on_time:
            return 0 #'早着'
        elif on_time < delivery_time < late.time():
            return 1 #'定刻'
        elif late.time() < delivery_time < late_plus_2hrs:
            return 2 #'遅着'
        else:
            return 3 #'ダイヤ変更'
    else:
        return 4#'便無し'
    
def drop_columns_with_word(df, word):
    
    """
    特定の文字列を含む列を削除する関数

    Parameters:
    df (pd.DataFrame): 対象のデータフレーム
    word (str): 削除したい列名に含まれる文字列

    Returns:
    pd.DataFrame: 指定された文字列を含む列が削除されたデータフレーム
    """
    
    columns_to_drop = [column for column in df.columns if word in column]
    return df.drop(columns=columns_to_drop)

# 過去X時間前からY時間前前までの平均生産台数_加重平均済を計算する関数
def calculate_window_width(data, start_hours_ago, end_hours_ago, timelag, reception_timelag):
    data = data.sort_values(by='日時')
    # 生産台数の加重平均を計算
    data[f'生産台数_加重平均（t-{end_hours_ago}~t-{timelag}）'] = data['生産台数_加重平均済'].rolling(window=start_hours_ago+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 計画生産台数の加重平均を計算
    data[f'計画生産台数_加重平均（t-{end_hours_ago}~t-{timelag}）'] = data['生産台数_加重平均済'].rolling(window=timelag+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 計画生産台数の加重平均を計算
    data[f'計画達成率_加重平均（t-{end_hours_ago}~t-{timelag}）'] = data['計画達成率_加重平均済'].rolling(window=timelag+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 入庫かんばん数の合計を計算
    data[f'入庫かんばん数（t-{end_hours_ago}~t-{timelag}）'] = data['入庫かんばん数（t）'].rolling(window=timelag+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 出庫かんばん数の合計を計算
    data[f'出庫かんばん数（t-{end_hours_ago}~t-{timelag}）'] = data['出庫かんばん数（t）'].rolling(window=timelag+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 在庫増減数の合計を計算
    data[f'在庫増減数（t-{end_hours_ago}~t-{timelag}）'] = data['在庫増減数(t)'].rolling(window=timelag+1-end_hours_ago, min_periods=1).sum().shift(end_hours_ago)
    # 発注かんばん数の合計を計算
    data[f'発注かんばん数（t-{timelag}~t-{timelag*2}）'] = data['発注かんばん数(t)'].rolling(window=timelag+1, min_periods=1).sum().shift(timelag)
    # 納入かんばん数の合計を計算
    data[f'納入かんばん数（t-{reception_timelag}~t-{timelag+reception_timelag}）'] = data['納入かんばん数(t)'].rolling(window=timelag+1, min_periods=1).sum().shift(timelag)
    # 在庫数（箱）のシフト
    data[f'在庫数（箱）（t-{timelag}）'] = data['在庫数（箱）'].shift(timelag)
    
    return data
