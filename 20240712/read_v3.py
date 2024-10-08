#ライブラリのimport
import os
#os.add_dll_directory('C:/Program Files/IBM/IBM DATA SERVER DRIVER/bin')
#import ibm_db
import pandas as pd
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, time
import glob
import re
#import pyodbc
import streamlit as st
import json
import datetime as dt

#! 必要データ読み込み
def read_data(start_date, end_date):

    #DBから取得するかのフラグ
    #DBに接続できない環境なら利用
    #対応日付は'2024-05-01-00'から'2024-08-31-00'
    flag_DB = 0

    #!-----------------------------------------------------------------------
    #! ろじれこのデータの読み込み
    #! Args：flag_DB、start_date, end_date
    #! Return：teikibin_df（前処理済みのろじれこデータ）
    #!-----------------------------------------------------------------------
    #* ＜DBからデータ取得する場合＞
    if flag_DB == 1:
        teikibin_df = process_teikibin(start_date, end_date)
        teikibin_df['日時'] = pd.to_datetime(teikibin_df['日時'])
    #* ＜ローカルデータ利用する場合＞
    elif flag_DB == 0:
        file_path = '中間成果物/定期便前処理.csv'
        teikibin_df = pd.read_csv(file_path, encoding='shift_jis')
        teikibin_df['日時'] = pd.to_datetime(teikibin_df['日時'])
        teikibin_df= teikibin_df.loc[(teikibin_df['日時'] >= start_date) & (teikibin_df['日時'] <= end_date)]
    #　重複無しなので重複削除処理無し
    #! 実行結果の確認
    st.header("✅ろじれこデータの読み込み完了しました")
    st.dataframe(teikibin_df)
    
    #!-----------------------------------------------------------------------
    #! 所在管理リードタイムのデータ
    #! Args：start_date, end_date
    #! Return：Timestamp_df（所在管理MBのテーブルデータ）
    #!-----------------------------------------------------------------------
    # ↓アーカイブデータと所在管理DBの読み込み
    #! 所在管理MBのテーブルデータ
    Timestamp_df = read_syozailt_by_using_archive_data(start_date, end_date)
    #! 実行結果の確認
    st.header("✅所在管理リードタイムデータの読み込み完了しました")
    st.dataframe(Timestamp_df)

    #! 品番＆仕入先＆仕入先工場のマスターテーブルを作成する
    # ユニークな組み合わせを抽出
    # todo MBテーブルなので１Yと１Zに絞られている。将来的には抽出方法の変更が必要
    unique_hinban_plant = Timestamp_df[['品番', '仕入先名', '仕入先工場名']].drop_duplicates()
    #! 結果をCSVに保存（必須。次のActivedataの読込で使用する）
    unique_hinban_plant.to_csv('temp/マスター_品番&仕入先名&仕入先工場名.csv', index=False, encoding='shift_jis')

    #!-----------------------------------------------------------------------
    #! Activeのデータの読み込み。関数内で仕入先ダイヤを参照統合
    #! Args：flag_DB、start_date, end_date
    #! Return：active_df（Active情報＋仕入先ダイヤをもとに設計値MIN、設計値MAXを計算）
    #!-----------------------------------------------------------------------
    #todo 負荷軽減を考えて、差分期間抽出がいいかも
    #todo 最終次変を採用する方がいいかも（今は正式板）
    active_df = read_activedata_by_using_archive_data(start_date, end_date, flag_DB)
    #!実行結果の確認
    st.header("✅Activeのデータの読み込み完了しました")
    st.dataframe(active_df)

    #!-----------------------------------------------------------------------
    #! 自動ラックの在庫推移のデータ
    #! Args：start_date, end_date
    #! Return：zaiko_df（在庫推移MBのテーブルデータ）
    #!-----------------------------------------------------------------------
    # ↓アーカイブデータと所在管理DBの読み込み
    zaiko_df = read_zaiko__by_using_archive_data(start_date, end_date)
    #!実行結果の確認
    st.header("✅在庫データの読み込み完了しました")
    st.dataframe(zaiko_df.head(50000))#表示できる限界の200MBを超えるため部分的に表示
    #st.dataframe(zaiko_df[zaiko_df['品番'] == "35300ECB010"])

    #!------------------------------------------------------------------------
    #! 自動ラックの間口別在庫数や全入庫数のデータ計算
    #! Args：zaiko_df
    #! Return：AutomatedRack_Details_df（ラック別の在庫推移のデータ）
    #!------------------------------------------------------------------------
    AutomatedRack_Details_df = calculate_AutomatedRack_Details(zaiko_df)
    #!実行結果の確認
    st.header("✅ラック別の在庫数の計算しました")
    st.dataframe(AutomatedRack_Details_df)

    #!------------------------------------------------------------------------
    #! 仕入先ダイヤ別の早着や遅れ時間を計算
    #! Args：無し（関数内で仕入先ダイヤパスを参照する）
    #! Return：arrival_times_df（仕入先別の早着や遅れ時間を計算）
    #!------------------------------------------------------------------------
    #arrival_times_df = calculate_supplier_truck_arrival_types()
    arrival_times_df = calculate_supplier_truck_arrival_types2()#藤井さんに頂いた新しい便ダイヤ
    #!実行結果の確認
    st.header("✅仕入先ダイヤの読み込み完了しました")
    st.dataframe(arrival_times_df)
    
    #!------------------------------------------------------------------------
    #! 組立実績データの加重平均を計算
    #! Args：start_date, end_date
    #! Return：kumitate_df
    #!------------------------------------------------------------------------
    kumitate_df = calculate_weighted_average_of_kumitate(start_date, end_date)

    #! 重複を削除 ＜重要＞
    #! 'KUMI_CD', 'LINE_CD', 'LINE_DATE', 'PLAN_PRODUCT_CNT' ,'PRODUCT_CNT','TYOKU_KBN','JIKANWARI_KBN'列に基づいて重複行を削除
    kumitate_df = kumitate_df.drop_duplicates(subset=['KUMI_CD', 'LINE_CD', 'LINE_DATE', 'PLAN_PRODUCT_CNT','PRODUCT_CNT','TYOKU_KBN','JIKANWARI_KBN'])

    #実行結果の確認
    st.header("✅IT生産管理版データの読み込み完了しました")
    st.dataframe(kumitate_df)

    #データを返す
    return AutomatedRack_Details_df, arrival_times_df, kumitate_df, teikibin_df, Timestamp_df, zaiko_df

def get_date_differences(file_path, start_date, end_date):#! 差分期間計算

    #! JSONファイルが存在するかチェック。無ければ作る。あれば読み込む
    # もし無ければ今回の日付をJSONファイルに書き込む
    if not os.path.exists(file_path):
        # 変数としてまとめる
        data = {
            "start_date": start_date,
            "end_date": end_date
        }
        # JSONファイルを書き込む
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        # JSONファイルを読み込む
        with open(file_path, 'r') as json_file:
            loaded_data = json.load(json_file)
        flag_first = 1 
    # もしあれば読み込む
    else:
        # JSONファイルを読み込む
        with open(file_path, 'r') as json_file:
            loaded_data = json.load(json_file)
        flag_first = 0
    
    #! 差分期間の計算
    #　読み込んだデータを日付を文字列型からdatetimeオブジェクトに変換
    start_date = datetime.strptime(start_date, '%Y-%m-%d-%H')
    end_date = datetime.strptime(end_date, '%Y-%m-%d-%H')
    start_date_archive = datetime.strptime(loaded_data["start_date"], '%Y-%m-%d-%H')
    end_date_archive = datetime.strptime(loaded_data["end_date"], '%Y-%m-%d-%H')
    flag = 0
    #! アーカイブ期間と指定期間がまったく同じ場合（初回でjsonファイルを作った場合など）
    if flag_first == 1: 
        start_date_all = start_date
        start_date_dif = start_date
        end_date_all = end_date
        end_date_dif = end_date
        #初回で作った場合はデータないので、flag=1にして読み込むようにする
        flag = 1
    elif (start_date == start_date_archive) & (end_date == end_date_archive):
        start_date_all = start_date
        start_date_dif = start_date
        end_date_all = end_date
        end_date_dif = end_date
        #まったく同じなら読み込む必要ないのでflag=0にして読み込みを禁止する
        flag = 0
    #! 指定期間がまったく同じでない場合
    else:
        #! アーカイブに存在する期間を計算
        start_date_all = min(start_date, start_date_archive)
        end_date_all = max(end_date, end_date_archive)
        start_date_dif = None
        end_date_dif = None
        flag = 0
        #! 存在する期間の修正
        if end_date < start_date_archive:
            start_date_dif = start_date
            end_date_dif = start_date_archive
            flag = 1
        elif (start_date < start_date_archive) & (start_date_archive < end_date):
            start_date_dif = start_date
            end_date_dif = start_date_archive
            flag = 1
        elif (start_date_archive < start_date) & (end_date < end_date_archive):
            flag = 0
        elif (start_date < end_date_archive) & (end_date_archive < end_date):
            start_date_dif = end_date_archive
            end_date_dif = end_date
            flag = 1
        elif end_date_archive < start_date:
            start_date_dif = end_date_archive
            end_date_dif = end_date
            flag = 1

    #! アーカイブ期間の更新
    # JSONファイルに書き込む
    data = {
        "start_date": start_date_all.strftime('%Y-%m-%d-%H'),
        "end_date": end_date_all.strftime('%Y-%m-%d-%H')
    }
    # ファイルパスを作成
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)

    #実行結果の確認
    #st.header(flag)
    #st.header(start_date_dif)

    #DBからデータ吸い出す場合はこの処理不要
    #if flag == 1:
        # 再度文字列形式に戻す
        #start_date_dif = start_date_dif.strftime('%Y-%m-%d-%H')
        #end_date_dif = end_date_dif.strftime('%Y-%m-%d-%H')

    return start_date_dif, end_date_dif, flag

def process_teikibin(start_date, end_date):#! ろじれこデータ取得

    # SQLデータベース接続関数
    def sql_connection():
        connection_string = (
            "Driver={SQL Server};"
            "Server=MMD-LOGISTICS\SQLEXPRESS;"
            "Database=LogiRecoDB;"
            "UID=dev;"
            "PWD=awadmin;"
        )
        connection = pyodbc.connect(connection_string)
        connection.cursor().execute("SET NOCOUNT ON")
        return connection

    # 実績読み込み関数
    def read_performance():

        base_cd = "IN03"
        #KOTEI_IDは02を使用

        # SQL実行
        conn = sql_connection()
        sql = f"""SELECT KOTEI_ID, WORK_ID, JISEKI_DT, JISEKI_DT2, TYOKU 
                FROM TBD_JISEKI 
                WHERE BASE_CD='{base_cd}' AND LINE_DATE >= '{start_date}' AND LINE_DATE <= '{end_date}'
                ORDER BY SEQ"""
        df = pd.read_sql(sql, conn)
        print(df.head(100))
        print(len(df))
        
        filtered_df = df[df['KOTEI_ID'] == '02']
        
        filtered_df.loc[:, 'JISEKI_DT'] = pd.to_datetime(filtered_df['JISEKI_DT'], format='%Y-%m-%d %H:%M:%S.%f')
        filtered_df.loc[:, 'JISEKI_DT2'] = pd.to_datetime(filtered_df['JISEKI_DT2'], format='%Y-%m-%d %H:%M:%S.%f')

        # 統合したデータを新しいCSVファイルに保存
        with open("生データ/ろじれこ/定期便.csv", mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
            filtered_df.to_csv(f)
            
        filtered_df.head(10)
        
    def add_previous_hours_data(df, X):
        """
        データフレームに1時間前からX時間前までの「便合計」のデータ列を追加する関数。

        Args:
        df (DataFrame): 入力データフレーム。
        X (int): 追加する時間の範囲（1時間前からX時間前まで）。

        Returns:
        DataFrame: 更新されたデータフレーム。
        """
        for i in range(1, X + 1):
            df[f'荷役時間(t-{i})'] = df['荷役時間'].shift(i)
        return df
        
    # 評価実行関数
    def evaluate():
        read_performance()

    #! 実行
    #抽出したデータを定期便.csvに入れる
    evaluate()

    #! ファイル読み込み
    file_path_teikibin = '生データ/ろじれこ/定期便.csv'
    teikibin_data = pd.read_csv(file_path_teikibin, encoding='shift_jis')

    #! 日時の列を datetime 型に変換
    teikibin_data['JISEKI_DT'] = pd.to_datetime(teikibin_data['JISEKI_DT'])
    teikibin_data['JISEKI_DT2'] = pd.to_datetime(teikibin_data['JISEKI_DT2'])
    #1時間単位に変換
    teikibin_data['定期便到着時刻（1H）'] = pd.to_datetime(teikibin_data['JISEKI_DT']).dt.floor('H')
    teikibin_data['定期便出発時刻（1H）'] = pd.to_datetime(teikibin_data['JISEKI_DT2']).dt.floor('H')
    # 日時の差を計算
    teikibin_data["荷役時間"] = teikibin_data['JISEKI_DT2'] - teikibin_data['JISEKI_DT']

    #! 各WORK_IDと定期便到着時刻（1H）の組み合わせに対して荷役時間の合計を計算
    grouped = teikibin_data.groupby(['WORK_ID', '定期便到着時刻（1H）'])['荷役時間'].sum().reset_index()

    #! 1時間毎のデータフレームに各WORK_IDごとの「定期便到着時刻（1H）」列を追加する
    date_range = pd.date_range(start = start_date, end = end_date, freq='H')

    #! YYYYMMDDHに全ての時間帯をマッピング
    all_hours_df = pd.DataFrame(date_range, columns=['YYYYMMDDH']).set_index('YYYYMMDDH')

    #! 結果を保存するための空のDataFrameを準備
    result_df = all_hours_df.copy()

    #! 元のデータセットからユニークなWORK_IDを抽出する
    unique_work_ids = teikibin_data['WORK_ID'].unique()

    for work_id in unique_work_ids:
        # 特定のWORK_IDに対する荷役時間を含む時間帯のデータフレームを抽出
        work_times = grouped[grouped['WORK_ID'] == work_id]
        work_times = work_times.set_index('定期便到着時刻（1H）')
        # 荷役時間を1時間ごとのデータフレームにマージ
        result_df[f'荷役時間_便_{work_id}'] = work_times['荷役時間']

    result_df_reset = result_df.reset_index()

    # 荷役時間を分単位に変換し、float型で保存するために、Timedeltaを分に変換する処理を行います。
    for col in result_df_reset.columns:
        if "便" in col:
            # Timedeltaを分に変換
            result_df_reset[col] = result_df_reset[col].dt.total_seconds() / 60

    pattern_columns = result_df_reset.filter(regex='荷役時間_便_[0\d\W]+').columns
    print(pattern_columns)

    result_df_reset['荷役時間']=result_df_reset[pattern_columns].sum(axis=1)
    result_df_reset.fillna(0, inplace=True)  # 一括でNaNを0に変換

    # 関数を使用してデータフレームを更新
    X = 8  # 1時間前から8時間前までのデータ列を追加
    updated_df = add_previous_hours_data(result_df_reset, X)

    # 更新されたデータフレームの最初の数行を表示して内容を確認
    updated_df.head()

    # 結果の一部を表示して確認
    result_df_reset.head(100)
    # 日付範囲に基づいてデータをフィルタリング
    start_date_temp = pd.Timestamp(start_date)
    end_date_temp = pd.Timestamp(end_date)
    f = result_df_reset[(result_df_reset['YYYYMMDDH'] >= start_date_temp) & (result_df_reset['YYYYMMDDH'] <= end_date_temp)]
    f.head(20)

    updated_df = updated_df.rename(columns={'YYYYMMDDH': '日時'})

    return updated_df

def read_syozailt(start_date, end_date):#! 所在管理MBのデータ取得

    connection_string = (
    "Driver={Dr.Sum 5.5 ODBC Driver};"
    "Server=10.88.11.114;"
    "Port=6001;"
    "Database=本番;"
    "UID=1082794-Z100;"
    "PWD=11Sasa0302;"
    )

    connection = pyodbc.connect(connection_string)
    cur = connection.cursor()

    # SQL文の作成
    sql = """
        SELECT *
        FROM T403物流情報_所在管理_リードタイム
        WHERE 更新日時 >= '{}' AND 更新日時 <= '{}'
    """.format(start_date, end_date)

    # SQL文の実行
    cur.execute(sql)

    # 結果をデータフレームに読み込み
    df = pd.read_sql(sql, con=connection)

    # 接続を閉じる
    cur.close()

    return df

def read_syozailt_by_using_archive_data(start_date, end_date):#! 所在管理MBのデータ取得（アーカイブデータを使用して）
        #* アーカイブ期間、差分期間の計算 -----------------------------------------
        # 保存先のフォルダー
        folder_path = 'archive_data/leadtime'
        # ファイル名を指定
        file_name = "archive_period.json"
        # ファイルパスを作成
        file_path = os.path.join(folder_path, file_name)
        # 差分期間の計算
        start_date_dif, end_date_dif, flag = get_date_differences(file_path, start_date, end_date)
        #*-----------------------------------------------------------------------
        #* 差分期間があれば読み込む
        if flag == 1:
            Timestamp_df = read_syozailt(start_date_dif, end_date_dif)
            # 品番列の空白を削除
            Timestamp_df['品番'] = Timestamp_df['品番'].str.strip()
            # 印刷日時、入庫日時、出庫日時、検収日時をdatetime型に変換
            Timestamp_df['発注日時'] = pd.to_datetime(Timestamp_df['発注日時'], errors='coerce')
            Timestamp_df['印刷日時'] = pd.to_datetime(Timestamp_df['印刷日時'], errors='coerce')
            Timestamp_df['順立装置入庫日時'] = pd.to_datetime(Timestamp_df['順立装置入庫日時'], errors='coerce')
            Timestamp_df['順立装置出庫日時'] = pd.to_datetime(Timestamp_df['順立装置出庫日時'], errors='coerce')
            Timestamp_df['検収日時'] = pd.to_datetime(Timestamp_df['検収日時'], errors='coerce')
        
            # 列の値を変換する関数
            def convert_value(value):
                if value.startswith('0'):
                    return int(value[1])
                else:
                    return int(value) 

            #! 列の値を変換
            #! What：納入便の値を「0X」から「X」にする
            Timestamp_df['納入便'] = Timestamp_df['納入便'].apply(convert_value)

            # 差分データを新アーカイブデータとして保存
            # ファイル名を指定
            file_name = f"{start_date_dif.strftime('%Y-%m-%d-%H')}~{end_date_dif.strftime('%Y-%m-%d-%H')}.csv"
            # ファイルパスを作成
            file_path = os.path.join(folder_path, file_name)

            # 差分データを新アーカイブデータとして保存
            with open(file_path, mode='w', newline='', encoding='shift_jis',errors='ignore') as f:
                Timestamp_df.to_csv(f)

        #! フォルダー内のすべてのCSVファイルを読み込む
        all_dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, encoding='shift_jis')
                all_dataframes.append(df)

        #! データフレームを統合
        Timestamp_df = pd.concat(all_dataframes, ignore_index=True)
        Timestamp_df['発注日時'] = pd.to_datetime(Timestamp_df['発注日時'], errors='coerce')
        Timestamp_df['印刷日時'] = pd.to_datetime(Timestamp_df['印刷日時'], errors='coerce')
        Timestamp_df['順立装置入庫日時'] = pd.to_datetime(Timestamp_df['順立装置入庫日時'], errors='coerce')
        Timestamp_df['順立装置出庫日時'] = pd.to_datetime(Timestamp_df['順立装置出庫日時'], errors='coerce')
        Timestamp_df['検収日時'] = pd.to_datetime(Timestamp_df['検収日時'], errors='coerce')
        Timestamp_df['更新日時'] = pd.to_datetime(Timestamp_df['更新日時'], errors='coerce')

        #! 重複を削除 ＜重要＞
        # 'かんばんシリアル' が重複する場合、最新の '更新日時' を持つ行を残す
        Timestamp_df = Timestamp_df.sort_values('更新日時').drop_duplicates(subset=['かんばんシリアル'], keep='last')

        return Timestamp_df

def read_zaiko(start_date, end_date):#! 在庫推移MBのデータ取得

    connection_string = (
    "Driver={Dr.Sum 5.5 ODBC Driver};"
    "Server=10.88.11.114;"
    "Port=6001;"
    "Database=本番;"
    "UID=1082794-Z100;"
    "PWD=11Sasa0302;"
    )

    connection = pyodbc.connect(connection_string)
    cur = connection.cursor()

    # 抽出する期間と品番の指定
    #start_date = '2024-01-09'
    #end_date = '2024-09-09'
    #product_code = '9056451A089'

    # SQL文の作成
    sql = """
        SELECT 品番, 品名, 前工程コード, 前工程工場コード, 仕入先名, 現在在庫（箱）, 現在在庫（台）, 更新日時, 入庫（箱）, 出庫（箱）, 入庫（台）, 出庫（台）, 拠点所番地
        FROM T403物流情報_在庫推移
        WHERE 更新日時 >= '{}' AND 更新日時 <= '{}'
    """.format(start_date, end_date)

    # SQL文の実行
    cur.execute(sql)

    # 結果をデータフレームに読み込み
    df = pd.read_sql(sql, con=connection)

    # 接続を閉じる
    cur.close()

    return df

def read_zaiko__by_using_archive_data(start_date, end_date):#! 在庫推移MBのデータ取得（アーカイブデータを使用して）
        #* アーカイブ期間、差分期間の計算 -----------------------------------------
        # 保存先のフォルダー
        folder_path = 'archive_data/rack'
        # ファイル名を指定
        file_name = "archive_period.json"
        # ファイルパスを作成
        file_path = os.path.join(folder_path, file_name)
        # 差分期間の計算
        start_date_dif, end_date_dif, flag = get_date_differences(file_path, start_date, end_date)
        #*-----------------------------------------------------------------------
        #* 差分期間があれば読み込む
        if flag == 1:
            zaiko_df =  read_zaiko(start_date_dif, end_date_dif)
            # 品番列の空白を削除
            zaiko_df['品番'] = zaiko_df['品番'].str.strip()
            zaiko_df['更新日時'] = pd.to_datetime(zaiko_df['更新日時'], errors='coerce')
            zaiko_df = zaiko_df.rename(columns={'更新日時': '日時'})
            zaiko_df = zaiko_df.rename(columns={'入庫（箱）': '入庫数（箱）', '出庫（箱）': '出庫数（箱）', '現在在庫（箱）': '在庫数（箱）'})

            #! 差分データを新アーカイブデータとして保存
            # ファイル名を指定
            file_name = f"{start_date_dif.strftime('%Y-%m-%d-%H')}~{end_date_dif.strftime('%Y-%m-%d-%H')}.csv"
            # ファイルパスを作成
            file_path = os.path.join(folder_path, file_name)
            # 差分データを新アーカイブデータとして保存
            with open(file_path, mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
                zaiko_df.to_csv(f)

        #! フォルダー内のすべてのCSVファイルを読み込む
        all_dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, encoding='shift_jis')
                all_dataframes.append(df)
        # データフレームを統合
        zaiko_df = pd.concat(all_dataframes, ignore_index=True)
        zaiko_df['日時'] = pd.to_datetime(zaiko_df['日時'], errors='coerce')

        #! 重複を削除 ＜重要＞
        # '品番', '日時', '入庫数（箱）', '出庫数（箱）' ,'拠点所番地'列に基づいて重複行を削除
        zaiko_df = zaiko_df.drop_duplicates(subset=['品番', '日時', '入庫数（箱）', '出庫数（箱）','拠点所番地'])

        return zaiko_df

def process_Activedata():#! Active(旧)

    # ディレクトリ内のすべてのCSVファイルを取得
    file_paths = glob.glob('生データ/手配必要数/*.csv')

    # 統合結果を保存するリスト
    all_data = []

    # 各CSVファイルに対して処理を実行
    for file_name in file_paths:
        # ファイル名から年と月を抽出
        year_month = re.findall(r'\d{6}', os.path.basename(file_name))[0]  # ファイル名の頭6文字から年と月を抽出
        year = int(year_month[:4])#YYYY
        month = int(year_month[4:6])#MM

        # CSVファイルを読み込む
        # todo 必要数のみ読み取り
        df_raw = pd.read_csv(file_name, encoding='shift_jis', skiprows=9, usecols=range(70))  # 10行目から読み込むために9行スキップ

        # 列名のクリーニング
        df_raw.columns = df_raw.columns.str.replace('="', '').str.replace('"', '')

        # 日量数列の選択
        daily_columns = df_raw.columns[df_raw.columns.str.contains(r'\d+\(.*\)')].tolist()
        print(daily_columns)
        df_relevant = df_raw[['品番','品名', '仕入先名/工場名','発送場所名','収容数', '整備室','サイクル間隔', 'サイクル回数', 'サイクル情報'] + daily_columns]

        # データフレームを縦に展開
        df_melted = df_relevant.melt(id_vars=['品番','品名','仕入先名/工場名','発送場所名', '収容数', '整備室','サイクル間隔', 'サイクル回数', 'サイクル情報'], var_name='日付', value_name='日量数')

        # 日付の列を整数型に変換
        df_melted['日付'] = df_melted['日付'].str.extract(r'(\d+)').astype(int)

        # 日付列に年と月を統合
        df_melted['日付'] = pd.to_datetime(df_melted.apply(lambda row: f"{year}-{month}-{row['日付']}", axis=1))

        # 値クリーニング
        df_melted['日量数'] = df_melted['日量数'].str.replace(',', '').fillna(0).astype(int)
        df_melted['品番'] = df_melted['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')
        df_melted['品名'] = df_melted['品名'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)
        df_melted['仕入先名/工場名'] = df_melted['仕入先名/工場名'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)
        df_melted['発送場所名'] = df_melted['発送場所名'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)
        df_melted['収容数'] = df_melted['収容数'].str.replace(',', '').fillna(0).astype(int)
        df_melted['整備室'] = df_melted['整備室'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)
        df_melted['サイクル間隔'] = df_melted['サイクル間隔'].astype(str).str.replace('="', '').str.replace('"', '').astype(int)
        df_melted['サイクル回数'] = df_melted['サイクル回数'].astype(str).str.replace('="', '').str.replace('"', '').astype(int)
        df_melted['サイクル情報'] = df_melted['サイクル情報'].astype(str).str.replace('="', '').str.replace('"', '').astype(float)

        #
        df_melted['日量数（箱数）'] = df_melted['日量数']/df_melted['収容数']
        # 年と週番号を追加
        df_melted['年'] = df_melted['日付'].dt.year
        df_melted['週番号'] = df_melted['日付'].dt.isocalendar().week

        # 結果をリストに追加
        all_data.append(df_melted)

    #! すべてのデータフレームを統合
    df_final = pd.concat(all_data, ignore_index=True)

    #! 週最大日量数の計算
    df_final['週最大日量数'] = df_final.groupby(['品番', '週番号'])['日量数'].transform('max')
    df_final['週最大日量数（箱数）'] = df_final['週最大日量数']//df_final['収容数']

    #! 設計値MIN（小数点以下を切り上げ）
    df_final['設計値MIN'] = np.ceil(0.1*(df_final['週最大日量数（箱数）']*df_final['サイクル間隔']*(1+df_final['サイクル情報'])/df_final['サイクル回数'])).astype(int)
    #! 設計値MAX（小数点以下を切り上げ）
    df_final['便Ave'] = np.ceil(df_final['週最大日量数（箱数）']/df_final['サイクル回数']).astype(int)
    df_final['設計値MAX'] = df_final['設計値MIN'] + df_final['便Ave']

    pitch = calculate_pitch()
    pitch = pitch.rename(columns={'仕入先名':'仕入先名/工場名'})# コラム名変更
    pitch = pitch.rename(columns={'受入':'整備室'})# コラム名変更
    # 空白やNaNを <NULL> に置き換える
    df_final['発送場所名'] = df_final['発送場所名'].replace('', '<NULL>').fillna('<NULL>')
    pitch['発送場所名'] = pitch['発送場所名'].replace('', '<NULL>').fillna('<NULL>')
    df_final = pd.merge(df_final, pitch[['仕入先名/工場名', '発送場所名','不等ピッチ係数（日）','不等ピッチ時間（分）','整備室']], on=['仕入先名/工場名', '発送場所名','整備室'], how='left')

    df_final['設計値MAX'] = df_final['設計値MAX'] + df_final['週最大日量数（箱数）']*df_final['不等ピッチ係数（日）']
    
    #実行結果の確認
    #st.header("Activeの計算結果")
    #st.dataframe(df_final[(df_final['品番'] == "01912ECB060")])
    #st.dataframe(pitch)

    return df_final

def read_activedata_from_IBMDB2(start_date, end_date, ver):#! Active、資材参照サーバーからデータ取得

    #todo
    #手配区分4だと収容数が空白になる

    #! 手配必要数テーブルの読み込み
    def read_active_data_tehaisu(start_date, end_date, ver):

        # データベース接続情報
        dsn_hostname = "mng-rht-rr01"
        dsn_port = 50000
        dsn_database = "HDUP"
        dsn_uid = "xais025"
        dsn_pwd = "xais025"

        # 接続文字列の作成
        dsn = (
            "DRIVER={{IBM DB2 ODBC DRIVER}};"
            "DATABASE={0};"
            "HOSTNAME={1};"
            "PORT={2};"
            "PROTOCOL=TCPIP;"
            "UID={3};"
            "PWD={4};"
        ).format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)

        # データベースに接続
        conn = ibm_db.connect(dsn, dsn_uid, dsn_pwd)

        # 動的にSQLクエリを構築
        #対象年月、工場区分、手配担当整備室
        columns = ["FDTHK01","FDTHK02","FDTHK03","FDTHK04","FDTHK11","FDTHK13","FDTHK16"]
        #計算数
        for i in range(20, 111, 3):
            columns.append(f"FDTHK{i}")

        #SQL
        sql = f"""SELECT {', '.join(columns)} FROM awj.TDTHK
            WHERE FDTHK01 >= '{start_date}' AND FDTHK01 <= '{end_date}' AND FDTHK03 = '{ver}' AND FDTHK13 IN ('1Y', '1Z') 
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
        df.rename(columns={'FDTHK01': '対象年月'}, inplace=True)
        df.rename(columns={'FDTHK02': '工場区分'}, inplace=True)
        df.rename(columns={'FDTHK03': '計算区分'}, inplace=True)
        df.rename(columns={'FDTHK04': '品番'}, inplace=True)
        df.rename(columns={'FDTHK11': '受入場所'}, inplace=True)
        df.rename(columns={'FDTHK13': '手配担当整備室'}, inplace=True)
        df.rename(columns={'FDTHK16': '手配区分'}, inplace=True)

        # 4列目以降のカラム名を置換
        start_index = 8  # 8列目から
        new_columns = list(df.columns[:start_index - 1])  # 4列目より前のカラム名を保持

        # 4列目以降のカラム名を1から始まる整数に置換
        new_columns += [str(i - (start_index - 1) + 1) for i in range(start_index - 1, len(df.columns))]

        # 新しいカラム名を設定
        df.columns = new_columns

        # 結果をCSVに保存（必要に応じて）
        #df.to_csv('temo/tehai.csv', index=False, encoding='shift_jis')
    
        return df

    #! 手配必要数テーブルの縦展開
    def process_active_data_tehaisu(df):

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

        return df_melted

    #! 手配運用情報テーブルの読み込み
    def read_active_data_tehaiunyo():

        # データベース接続情報
        dsn_hostname = "mng-rht-rr01"
        dsn_port = 50000
        dsn_database = "HDUP"
        dsn_uid = "xais025"
        dsn_pwd = "xais025"

        # 接続文字列の作成
        dsn = (
            "DRIVER={{IBM DB2 ODBC DRIVER}};"
            "DATABASE={0};"
            "HOSTNAME={1};"
            "PORT={2};"
            "PROTOCOL=TCPIP;"
            "UID={3};"
            "PWD={4};"
        ).format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)

        # データベースに接続
        conn = ibm_db.connect(dsn, dsn_uid, dsn_pwd)

        # 動的にSQLクエリを構築
        #品番、受入場所、サイクル間隔、サイクル回数、サイクル情報、不良率
        columns = ["FDTUK01","FDTUK08","FDTUK28","FDTUK31","FDTUK37","FDTUK38","FDTUK39","FDTUK47"]

        #本番
        #sql = f"SELECT {', '.join(columns)} FROM awj.TDTHK "

        sql = f"""SELECT {', '.join(columns)} FROM awj.TDTUK
            WHERE  FDTUK08 IN ('1Y', '1Z') 
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
        df.rename(columns={'FDTUK08': '受入場所'}, inplace=True)
        df.rename(columns={'FDTUK28': '品名'}, inplace=True)
        df.rename(columns={'FDTUK31': '収容数'}, inplace=True)
        df.rename(columns={'FDTUK37': 'サイクル間隔'}, inplace=True)
        df.rename(columns={'FDTUK38': 'サイクル回数'}, inplace=True)
        df.rename(columns={'FDTUK39': 'サイクル情報'}, inplace=True)
        df.rename(columns={'FDTUK47': '不良率'}, inplace=True)

        # 列の値を変換する関数
        def convert_value(value):
            if value.startswith('0'):
                return int(value[1])
            else:
                return int(value) 

        #! 列の値を変換
        #! What：納入便の値を「0X」から「X」にする
        df['サイクル間隔'] = df['サイクル間隔'].apply(convert_value)
        df['サイクル回数'] = df['サイクル回数'].apply(convert_value)
        df['サイクル情報'] = df['サイクル情報'].apply(convert_value)

        # 結果をCSVに保存（必要に応じて）
        df.to_csv('test_unyo.csv', index=False, encoding='shift_jis')
    
        return df

    #! 手配必要数データ抽出
    tehaisu_df = read_active_data_tehaisu(start_date, end_date, ver)

    #! 手配必要数（縦展開）
    tehaisu_melt_df = process_active_data_tehaisu(tehaisu_df)

    #! 手配運用情報データ抽出
    tehaiunyo_df = read_active_data_tehaiunyo()

    #! 品番＆仕入先＆仕入先工場のマスターテーブルを読み込む
    # メモ：MBテーブルを使用しているので １Yと１Zに絞られている
    unique_hinban_and_plant = pd.read_csv('temp/マスター_品番&仕入先名&仕入先工場名.csv', encoding='shift_jis')

    #! 手配必要数と手配運用情報テーブルの統合
    merged_df = pd.merge(tehaisu_melt_df, tehaiunyo_df, on=['品番','受入場所'], how='left')

    #! 仕入先と仕入先工場名を統合
    merged_df = pd.merge(merged_df, unique_hinban_and_plant, on=['品番'], how='left')
    merged_df = merged_df.rename(columns={'仕入先名':'仕入先名/工場名'})# コラム名変更
    merged_df = merged_df.rename(columns={'仕入先工場名': '発送場所名'})# コラム名変更

    # 結果をCSVに保存（必要に応じて）
    merged_df.to_csv('temp/active_merge.csv', index=False, encoding='shift_jis')

    df_final = merged_df
    df_final['週番号'] = df_final['日付'].dt.isocalendar().week
    # 特定の列名を変更
    df_final.rename(columns={'手配担当整備室': '整備室'}, inplace=True)
    # '手配区分'が4でないものを抽出（4は収容数が0のため）
    df_final['手配区分'] = df_final['手配区分'].astype(int)
    df_final = df_final[df_final['手配区分'] != 4]

    #st.dataframe(df_final)

    # 値クリーニング
    df_final['品番'] = df_final['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '').str.replace(' ', '')
    df_final['整備室'] = df_final['整備室'].astype(str).str.replace('="', '').str.replace('"', '').astype(str)

    #! 週最大日量数の計算
    df_final['週最大日量数'] = df_final.groupby(['品番', '週番号'])['日量数'].transform('max')

    df_final['週最大日量数'] = pd.to_numeric(df_final['週最大日量数'], errors='coerce')
    df_final['収容数'] = pd.to_numeric(df_final['収容数'], errors='coerce')

    df_final['週最大日量数（箱数）'] = df_final['週最大日量数']//df_final['収容数']

    #! 設計値MIN（小数点以下を切り上げ）
    df_final['設計値MIN'] = np.ceil(0.1*(df_final['週最大日量数（箱数）']*df_final['サイクル間隔']*(1+df_final['サイクル情報'])/df_final['サイクル回数'])).astype(int)
    #! 設計値MAX（小数点以下を切り上げ）
    df_final['便Ave'] = np.ceil(df_final['週最大日量数（箱数）']/df_final['サイクル回数']).astype(int)
    df_final['設計値MAX'] = df_final['設計値MIN'] + df_final['便Ave']

    #! 不等ピッチ読み込み
    pitch = calculate_pitch()
    pitch = pitch.rename(columns={'仕入先名':'仕入先名/工場名'})# コラム名変更
    pitch = pitch.rename(columns={'受入':'整備室'})# コラム名変更
    # 空白やNaNを <NULL> に置き換える
    df_final['発送場所名'] = df_final['発送場所名'].replace('', '<NULL>').fillna('<NULL>')
    pitch['発送場所名'] = pitch['発送場所名'].replace('', '<NULL>').fillna('<NULL>')
    #! 統合
    df_final = pd.merge(df_final, pitch[['仕入先名/工場名', '発送場所名','不等ピッチ係数（日）','不等ピッチ時間（分）','整備室']], on=['仕入先名/工場名', '発送場所名','整備室'], how='left')

    df_final['設計値MAX'] = df_final['設計値MAX'] + df_final['週最大日量数（箱数）']*df_final['不等ピッチ係数（日）']
    
    #実行結果の確認
    #st.header("Activeの計算結果")
    #st.dataframe(df_final[(df_final['品番'] == "01912ECB060")])
    #st.dataframe(pitch)

    return df_final

def read_activedata_by_using_archive_data(start_date, end_date,flag_DB):#! Activeのデータ取得（アーカイブデータを使用して）
    #* ＜DBからデータ取得する場合＞
    if flag_DB == 1:
        #! ハイフンを削除し前6桁を抽出
        start_date_active = start_date[:7].replace('-', '')# 年YYYYを抽出
        end_date_active = end_date[:7].replace('-', '')# 月MMを抽出
        #! バージョン
        ver = '00' #正式
        active_df = read_activedata_from_IBMDB2(start_date_active, end_date_active, ver)
        #! 結果をCSVに保存（★保存必須。STEP2の基準線描画時に読み込むため）
        active_df.to_csv('temp/activedata.csv', index=False, encoding='shift_jis')
    #* ＜ローカルデータ利用する場合＞
    elif flag_DB == 0:
        file_path = 'temp/activedata.csv'
        active_df = pd.read_csv(file_path, encoding='shift_jis')

    return active_df

def calculate_pitch():#! 不等ピッチ係数計算

    file_path = '生データ/便ダイヤ/仕入先便ダイヤ20240922.xlsx'#こっちは無理
    #file_path = '生データ/便ダイヤ/仕入先便ダイヤ20240922.xlsx'

    # openpyxlエンジンを使用してExcelファイルを読み込む
    df = pd.read_excel(file_path, engine='openpyxl', skiprows=4)

    # 列名をリセット
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

    # 抽出したい列名を指定
    # 先に固定の列を定義
    columns_to_extract = ['仕入先名', '発送場所名', '受入', '納入先', '回数']

    # 1便から24便までの列名を生成して追加
    columns_to_extract += [f'{i}便' for i in range(1, 25)]

    # 指定した列のみを抽出
    extracted_df = df[columns_to_extract]

    # 効率化された関数：隣合う便と最後の便から1便の経過時間を計算
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

    # 各仕入先に対して最大便間時間とその詳細、時間単位も計算
    extracted_df['最長便ピッチ時間（分）'], extracted_df['最長便間詳細'], extracted_df['最長便時間（h）'] = zip(*extracted_df.apply(
        lambda row: calculate_time_differences_adjacent_efficient([row[f'{i}便'] for i in range(1, 25) if pd.notna(row[f'{i}便'])]), axis=1))
    
    extracted_df['等ピッチ時間（分）']=1150/extracted_df['回数']

    extracted_df['不等ピッチ時間（分）']=extracted_df['最長便ピッチ時間（分）']-extracted_df['等ピッチ時間（分）']

    # '回数'が1の場合、'不等ピッチ時間（分）'を0に設定
    extracted_df.loc[extracted_df['回数'] == 1, '不等ピッチ時間（分）'] = 0

    extracted_df['不等ピッチ係数（日）'] = extracted_df['不等ピッチ時間（分）']/1150

    return extracted_df
    
def calculate_AutomatedRack_Details(zaiko_df):#! 間口

    #モーションボードの列名を修正する必要あり
    #ラック間口.csvどうする？
    
    #列名の変更
    zaiko_df = zaiko_df.rename(columns={'計測日時': '日時'})
    #品番の修正
    zaiko_df['品番'] = zaiko_df['品番'].astype(str).str.replace("-", "").str.replace(" ", "").str.rstrip()
    # NaN を 0 で置き換え
    zaiko_df.fillna(0, inplace=True)

    # 計測日時ごとに入庫数と出庫数の合計を計算
    summary_zaiko_df = zaiko_df.groupby('日時').agg({
        '入庫数（箱）': 'sum',
        '出庫数（箱）': 'sum'
    }).reset_index()

    # 品番グループデータの読み込み（仮のパスを使用）
    group_data = pd.read_csv("ラック間口.csv", encoding='shift_jis')
    group_data['品番'] = group_data['品番'].astype(str).str.replace("-", "").str.replace(" ", "").str.rstrip()

    # 在庫データの読み込み（すでに読み込んでいるデータを使用）
    stock_data = zaiko_df

    # 品番グループデータと在庫データの結合
    merged_data = pd.merge(stock_data, group_data, on='品番', how='left')

    # 計測日時と間口ごとに在庫数を集計
    pivot_table = merged_data.pivot_table(
        values='在庫数（箱）',
        index='日時',
        columns='間口',
        aggfunc='sum'
    )

    # 特定の品番グループの列だけを選択
    result = pivot_table[['A1', 'A2', 'B1','B2', 'B3','B4']].copy()

    # 列名をリネームして明確化
    result.rename(columns={
        'A1': '在庫数（箱）合計_A1',
        'A2': '在庫数（箱）合計_A2',
        'B1': '在庫数（箱）合計_B1',
        'B2': '在庫数（箱）合計_B2',
        'B3': '在庫数（箱）合計_B3',
        'B4': '在庫数（箱）合計_B4'
    }, inplace=True)

    # インデックスをリセットして、元のインデックスを列に含める
    result_reset = result.reset_index()

    #ロボット間口の統合
    AutomatedRack_Details_df = pd.merge(
        summary_zaiko_df[['日時', '入庫数（箱）']], result_reset[['日時', '在庫数（箱）合計_A1','在庫数（箱）合計_A2', 
                                                                    '在庫数（箱）合計_B1', '在庫数（箱）合計_B2','在庫数（箱）合計_B3', '在庫数（箱）合計_B4']], on=['日時'], how='left')

    AutomatedRack_Details_df = AutomatedRack_Details_df.rename(columns={'入庫数（箱）': '全品番の合計入庫かんばん数'})#コラム名変更

    return AutomatedRack_Details_df

def calculate_supplier_truck_arrival_types():#! 仕入先ダイヤ（旧）

    def calculate_arrival_times(df, time_columns):
        # タイプが着時刻の行のみを抽出
        arrival_df = df[df['タイプ'] == '着時刻']

        # 1便以降の日付データを時間だけに変換
        for col in time_columns:
            arrival_df[col] = pd.to_datetime(arrival_df[col].astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0], format='%H:%M:%S', errors='coerce').dt.time

        # 早着、定刻、遅着の時間帯を計算
        arrival_df_with_times = arrival_df.copy()
        for col in time_columns:
            arrival_time = pd.to_datetime(arrival_df[col].astype(str), format='%H:%M:%S', errors='coerce')
            arrival_df_with_times[col + '_早着'] = (arrival_time - pd.Timedelta(hours=2)).dt.time
            #arrival_df_with_times[col + '_早着_終了'] = arrival_time.dt.time
            arrival_df_with_times[col + '_定刻'] = arrival_time.dt.time
            #arrival_df_with_times[col + '_定刻_終了'] = (arrival_time + pd.Timedelta(hours=1)).dt.time
            arrival_df_with_times[col + '_遅着'] = (arrival_time + pd.Timedelta(hours=2)).dt.time
            #arrival_df_with_times[col + '_遅着_終了'] = (arrival_time + pd.Timedelta(hours=2)).dt.time

        # 必要な列のみを抽出して返す
        result_columns = ['仕入先名', '発送場所名', '受入'] + [col for col in arrival_df_with_times.columns if '早着' in col or '定刻' in col or '遅着' in col]
        return arrival_df_with_times[result_columns]

    # ファイルパス
    #file_path = '生データ/便ダイヤ/仕入先便ダイヤ20240922.xlsx'#こっちは無理
    file_path = "生データ\便ダイヤ\仕入先便ダイヤ20240922.xlsx"

    # openpyxlエンジンを使用してExcelファイルを読み込む
    df = pd.read_excel(file_path, engine='openpyxl')

    # 5行目を列名として設定し、6行目以降のデータを抽出
    df.columns = df.iloc[5]
    df = df[6:]

    # 列名をリセット
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

    # 抽出したい列名を指定
    columns_to_extract = ['仕入先名', '発送場所名', '受入', '納入先', '1便', '2便', '3便', '4便', '5便', '6便', '7便', '8便', '9便', '10便', '11便', '12便']
    #columns_to_extract = ['仕入先名', '発送場所名', '受入', 'タイプ','4便', '5便', '6便']

    # 指定した列のみを抽出
    extracted_df = df[columns_to_extract]

    # 関数を使用して早着、定刻、遅着の時間帯を計算
    arrival_times_df = calculate_arrival_times(extracted_df, columns_to_extract[4:])

    # NaNの値を'< NULL >'に置換
    # 所在管理と結合するため
    arrival_times_df = arrival_times_df.fillna('< NULL >')

    return arrival_times_df

def calculate_supplier_truck_arrival_types2():#! 仕入先ダイヤ

    def calculate_arrival_times(df, time_columns):
        # タイプが着時刻の行のみを抽出
        arrival_df = df[df['受入'].isin(['1Y', '1Z'])]

        # 1便以降の日付データを時間だけに変換
        for col in time_columns:
            arrival_df[col] = pd.to_datetime(arrival_df[col].astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0], format='%H:%M:%S', errors='coerce').dt.time

        # 早着、定刻、遅着の時間帯を計算
        arrival_df_with_times = arrival_df.copy()
        for col in time_columns:
            arrival_time = pd.to_datetime(arrival_df[col].astype(str), format='%H:%M:%S', errors='coerce')
            arrival_df_with_times[col + '_早着'] = (arrival_time - pd.Timedelta(hours=2)).dt.time
            #arrival_df_with_times[col + '_早着_終了'] = arrival_time.dt.time
            arrival_df_with_times[col + '_定刻'] = arrival_time.dt.time
            #arrival_df_with_times[col + '_定刻_終了'] = (arrival_time + pd.Timedelta(hours=1)).dt.time
            arrival_df_with_times[col + '_遅着'] = (arrival_time + pd.Timedelta(hours=2)).dt.time
            #arrival_df_with_times[col + '_遅着_終了'] = (arrival_time + pd.Timedelta(hours=2)).dt.time

        # 必要な列のみを抽出して返す
        result_columns = ['仕入先名', '発送場所名', '受入'] + [col for col in arrival_df_with_times.columns if '早着' in col or '定刻' in col or '遅着' in col]
        return arrival_df_with_times[result_columns]

    # ファイルパス
    file_path = '生データ/便ダイヤ/仕入先便ダイヤ20240922.xlsx'#こっちは無理
    #file_path = '生データ/便ダイヤ/仕入先便ダイヤ20240922.xlsx'

    # openpyxlエンジンを使用してExcelファイルを読み込む
    df = pd.read_excel(file_path, engine='openpyxl', skiprows=4)

    # 5行目を列名として設定し、5行目以降のデータを抽出
    #df.columns = df.iloc[4]
    #df = df[5:]

    # 列名をリセット
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

    # 抽出したい列名を指定
    #columns_to_extract = ['仕入先名', '発送場所名', '受入', '納入先', '1便', '2便', '3便', '4便', '5便', '6便', '7便', '8便', '9便', '10便', '11便', '12便']

    # 先に固定の列を定義
    columns_to_extract = ['仕入先名', '発送場所名', '受入', '納入先']

    # 1便から24便までの列名を生成して追加
    columns_to_extract += [f'{i}便' for i in range(1, 25)]

    # 指定した列のみを抽出
    extracted_df = df[columns_to_extract]

    #st.dataframe(extracted_df)

    # 関数を使用して早着、定刻、遅着の時間帯を計算
    # 4列名以降を計算に使用
    arrival_times_df = calculate_arrival_times(extracted_df, columns_to_extract[4:])

    # NaNの値を'< NULL >'に置換
    # 所在管理と結合するため
    arrival_times_df = arrival_times_df.fillna('< NULL >')

    #st.dataframe(arrival_times_df)

    return arrival_times_df

def calculate_weighted_average_of_kumitate(start_date, end_date):#! IT生産管理版

    def set_A_B_columns(row, df):
        #!昼勤
        if row['TYOKU_KBN'] == 1:
            jikankwari_map = {
                1: ('8:30', 0.5, None, '8:00'),
                2: ('9:30', 0.5, 1, '9:00'),
                3: ('10:30', 0.5, 2, '10:00'),
                4: ('11:30', 0.5, 3, '11:00'),
                5: ('12:30', 0.5, 4, '12:00'),
                6: ('13:25', 0.5, 5, '13:00'),
                7: ('14:20', 2/3, 6, '14:00', 1/3),
                8: ('15:20', 2/3, 7, '15:00', 1/3),
                9: ('16:20', 2/3, 8, '16:00', 1/3),
                10: ('17:20', 2/3, 9, '17:00', 1/3),
                11: ('18:30', 0.5, 10, '18:00', 0.5),
                12: ('19:30', 0.5, 11, '19:00', 0.5),
                13: ('20:30', 0.5, 12, '20:00', 0.5)
            }
            if row['JIKANWARI_KBN'] in jikankwari_map:
                mapping = jikankwari_map[row['JIKANWARI_KBN']]
                row['時間割区分_開始時刻'] = mapping[0]
                row['調整日時'] = mapping[3]
                row['LINE_DATE_修正済'] = row['LINE_DATE']
                weight = mapping[1]
                previous_jikankwari_kbn = mapping[2]
                if previous_jikankwari_kbn is not None:
                    previous_product_cnt = df[(df['LINE_DATE'] == row['LINE_DATE']) & (df['JIKANWARI_KBN'] == previous_jikankwari_kbn)]['PRODUCT_CNT']
                    previous_plan_product_cnt = df[(df['LINE_DATE'] == row['LINE_DATE']) & (df['JIKANWARI_KBN'] == previous_jikankwari_kbn)]['PLAN_PRODUCT_CNT']
                    if not previous_product_cnt.empty:
                        if len(mapping) == 4:
                            row['生産台数_加重平均済'] = (row['PRODUCT_CNT'] * weight + previous_product_cnt.iloc[0] * weight)
                            row['計画生産台数_加重平均済'] = (row['PLAN_PRODUCT_CNT'] * weight + previous_plan_product_cnt.iloc[0] * weight)
                        else:
                            previous_weight = mapping[4]
                            row['生産台数_加重平均済'] = (row['PRODUCT_CNT'] * weight + previous_product_cnt.iloc[0] * previous_weight)
                            row['計画生産台数_加重平均済'] = (row['PLAN_PRODUCT_CNT'] * weight + previous_plan_product_cnt.iloc[0] * previous_weight)
                    else:
                        row['生産台数_加重平均済'] = row['PRODUCT_CNT'] * weight
                        row['計画生産台数_加重平均済'] = row['PLAN_PRODUCT_CNT'] * weight
                else:
                    row['生産台数_加重平均済'] = row['PRODUCT_CNT'] * weight
                    row['計画生産台数_加重平均済'] = row['PLAN_PRODUCT_CNT'] * weight
        #!夜勤      
        elif row['TYOKU_KBN'] == 2:
            jikankwari_map = {
                1: ('21:00', '21:00', None, 0),
                2: ('22:00', '22:00', None, 0),
                3: ('23:00', '23:00', None, 0),
                4: ('0:00', '0:00', None, 1),
                5: ('1:00', '1:00', None, 1),
                6: ('2:00', '2:00', None, 1),
                7: ('3:00', '3:00', None, 1),
                8: ('4:00', '4:00', None, 1),
                9: ('5:00', '5:00', None, 1),
                10: ('6:00', '6:00', None, 1),
                11: ('7:00', '7:00', None, 1),
                #12: ('8:00', '8:00', None, 1)
            }
            if row['JIKANWARI_KBN'] in jikankwari_map:
                mapping = jikankwari_map[row['JIKANWARI_KBN']]
                row['時間割区分_開始時刻'] = mapping[0]
                row['計画生産台数_加重平均済'] = row['PLAN_PRODUCT_CNT']
                row['生産台数_加重平均済'] = row['PRODUCT_CNT']
                row['調整日時'] = mapping[1]
                row['LINE_DATE_修正済'] = row['LINE_DATE'] + pd.Timedelta(days=mapping[3])
                
        return row


    #! ここにIoTPFのデータを入れてもらう
    directory = "生データ\IT生産管理版"

    # 空のDataFrameを作成
    kumitate_data = pd.DataFrame()

    # ディレクトリ内のすべてのCSVファイルを統合
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            kumitate_data = pd.concat([kumitate_data, data])

    #st.header("合体データ")
    #st.dataframe(kumitate_data)

    # 'LINE_DATE'列をdatetime型に変換
    # 'LINE_DATE'列は常に0:00を表す
    kumitate_data['LINE_DATE'] = pd.to_datetime(kumitate_data['LINE_DATE'])

    #kumitate_data = kumitate_data[kumitate_data['LINE_CD'] == "AS2610"]

    # 'PLAN_PRODUCT_CNT' にNaNがある行を削除
    # ★関数通す前にこれしないと、NH11とNH12の結果が関数でヒットして、NH12の昼勤計算がうまくいかなくなる
    kumitate_data = kumitate_data.dropna(subset=['PRODUCT_CNT'])

    # すべての列をfloat型に変換
    #kumitate_data[['PLAN_PRODUCT_CNT','PRODUCT_CNT','TYOKU_KBN(1)']] = kumitate_data[['PLAN_PRODUCT_CNT','PRODUCT_CNT','TYOKU_KBN(1)']].astype(float)

    # 関数を適用
    #kumitate_data = kumitate_data.apply(lambda row: set_A_B_columns(row, kumitate_data), axis=1)

    # 'LINE_CD' が 'AS2610' のデータを抽出
    kumitate_data_2610 = kumitate_data[kumitate_data['LINE_CD'] == 'AS2610']

    # 'LINE_CD' が 'AS2650' のデータを抽出
    kumitate_data_2650 = kumitate_data[kumitate_data['LINE_CD'] == 'AS2650']

    # それぞれのデータに関数を適用
    kumitate_data_2610 = kumitate_data_2610.apply(lambda row: set_A_B_columns(row, kumitate_data_2610), axis=1)
    kumitate_data_2650 = kumitate_data_2650.apply(lambda row: set_A_B_columns(row, kumitate_data_2650), axis=1)

    # 再度データを結合
    #kumitate_data = pd.concat([kumitate_data_2610, kumitate_data_2650])

    def process_kumitate_data(kumitate_data):

        kumitate_data['計画達成率_加重平均済'] = kumitate_data['生産台数_加重平均済']/kumitate_data['計画生産台数_加重平均済']

        # 'PLAN_PRODUCT_CNT' にNaNがある行を削除
        kumitate_data = kumitate_data.dropna(subset=['PRODUCT_CNT'])

        # '計画達成率_加重平均済' 列の NaN を 0 に置き換える
        kumitate_data['計画達成率_加重平均済'] = kumitate_data['計画達成率_加重平均済'].fillna(0)
        kumitate_data['生産台数_加重平均済'] = kumitate_data['生産台数_加重平均済'].fillna(0)
        kumitate_data['計画生産台数_加重平均済'] = kumitate_data['計画生産台数_加重平均済'].fillna(0)

        # LINE_DATE_修正済と調整日時を結合して新しい列Xを作成
        kumitate_data['LINE_DATE_修正済'] = pd.to_datetime(kumitate_data['LINE_DATE_修正済'])

        # 調整日時 も datetime 型に変換　#怪しい
        kumitate_data['調整日時'] = pd.to_datetime(kumitate_data['調整日時'], format='%H:%M').dt.time 

        # NaTを処理するためにfillnaを使用して、調整日時の欠損値をデフォルトの時間に置き換え
        kumitate_data['調整日時'] = kumitate_data['調整日時'].fillna(pd.to_datetime('00:00').time())

        #kumitate_data['日時'] = kumitate_data.apply(lambda row: pd.to_datetime.combine(row['LINE_DATE_修正済'], row['調整日時']), axis=1)#古い
        # '調整日時' がすでに datetime.time オブジェクトかどうかをチェックし、必要に応じて変換します
        kumitate_data['調整日時'] = kumitate_data['調整日時'].apply(lambda x: pd.to_datetime(x).time() if not isinstance(x, time) else x)
        # 'LINE_DATE_修正済' を date オブジェクトに変換し、 '調整日時' を time オブジェクトとして使用
        kumitate_data['日時'] = kumitate_data.apply(lambda row: datetime.combine(pd.to_datetime(row['LINE_DATE_修正済']).date(), row['調整日時']), axis=1)     

        # 'PLAN_PRODUCT_CNT' にNaNがある行を削除
        kumitate_data = kumitate_data.dropna(subset=['時間割区分_開始時刻'])

        # 日時順に並び替え
        kumitate_data = kumitate_data.sort_values(by='日時')

        kumitate_data['日時'] = pd.to_datetime(kumitate_data['日時'], errors='coerce')

        # 同じKUMI_CDと同じ日時のPRODUCT_CNTとPLAN_PRODUCT_CNTを統合
        #kumitate_data2 = kumitate_data.groupby(['KUMI_CD', '日時']).agg({'生産台数_加重平均済': 'sum', '計画生産台数_加重平均済': 'sum'}).reset_index()
        #kumitate_data['計画生産台数_加重平均済']=kumitate_data2['計画生産台数_加重平均済']
        #kumitate_data['生産台数_加重平均済']=kumitate_data2['生産台数_加重平均済']

        return kumitate_data

    # 'LINE_CD' が 'AS2610' のデータを処理
    kumitate_data_2610 = process_kumitate_data(kumitate_data_2610)

    # 'LINE_CD' が 'AS2650' のデータを処理
    kumitate_data_2650 = process_kumitate_data(kumitate_data_2650)

    # 処理後のデータを再結合
    kumitate_data = pd.concat([kumitate_data_2610, kumitate_data_2650])

    #! データフレームの特定の列の内容を置換する
    kumitate_data['整備室コード'] = kumitate_data['LINE_CD'].replace({'AS2610': '1Y','AS2650': '1Z'})

    #! 特定日時のみを取り出す
    kumitate_data = kumitate_data[
    (kumitate_data['日時'].dt.date >= pd.to_datetime(start_date).date()) &
    (kumitate_data['日時'].dt.date <= pd.to_datetime(end_date).date())]

    #実行結果の確認
    #st.dataframe(kumitate_data)

    return kumitate_data