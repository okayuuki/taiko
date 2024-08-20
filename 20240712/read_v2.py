#ライブラリのimport
import os
import pandas as pd
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, time
import glob
import re


def read_data():

    #学習期間（解析期間）任意に設定できるように
    #start_date = '2023-10-01'
    #end_date = '2024-03-31'

    #確認項目
    #日付にダブりがないか

    #
    file_path = '中間成果物/定期便前処理.csv'
    teikibin_df = pd.read_csv(file_path, encoding='shift_jis')
    teikibin_df['日時'] = pd.to_datetime(teikibin_df['日時'])
    
    #!-----------------------------------------------------------------------
    #!LINKSと自動ラックQRのタイムスタンプをかんばん単位で結合したもの
    #!-----------------------------------------------------------------------
    file_path = '中間成果物/所在管理MBデータ_統合済&特定日時抽出済.csv'
    Timestamp_df = pd.read_csv(file_path, encoding='shift_jis')
    # 品番列の空白を削除
    Timestamp_df['品番'] = Timestamp_df['品番'].str.strip()
    # 印刷日時、入庫日時、出庫日時、検収日時をdatetime型に変換
    Timestamp_df['発注日時'] = pd.to_datetime(Timestamp_df['発注日時'], errors='coerce')
    Timestamp_df['印刷日時'] = pd.to_datetime(Timestamp_df['印刷日時'], errors='coerce')
    Timestamp_df['順立装置入庫日時'] = pd.to_datetime(Timestamp_df['順立装置入庫日時'], errors='coerce')
    Timestamp_df['順立装置出庫日時'] = pd.to_datetime(Timestamp_df['順立装置出庫日時'], errors='coerce')
    Timestamp_df['検収日時'] = pd.to_datetime(Timestamp_df['検収日時'], errors='coerce')
    # データフレームの列名を表示
    #columns = df.columns.tolist()
    #print(columns)

    #!-----------------------------------------------------------------------
    #!自動ラックの在庫データ
    #!-----------------------------------------------------------------------
    file_path = '中間成果物/在庫推移MBデータ_統合済&特定日時抽出済.csv'
    zaiko_df = pd.read_csv(file_path, encoding='shift_jis')
    # 品番列の空白を削除
    zaiko_df['品番'] = zaiko_df['品番'].str.strip()
    # '計測日時'をdatatime型に変換
    zaiko_df['計測日時'] = pd.to_datetime(zaiko_df['計測日時'], errors='coerce')
    zaiko_df = zaiko_df.rename(columns={'計測日時': '日時'})

    #! 自動ラックの間口別在庫数や全入庫数のデータ計算
    AutomatedRack_Details_df = calculate_AutomatedRack_Details(zaiko_df)

    #! 仕入先ダイヤ別の早着や遅れ時間を計算
    arrival_times_df = calculate_supplier_truck_arrival_types()
    
    #! 組立実績データの加重平均を計算
    kumitate_df = calculate_weighted_average_of_kumitate()

    #
    return AutomatedRack_Details_df, arrival_times_df, kumitate_df, teikibin_df, Timestamp_df, zaiko_df

def process_Activedata():

    # ディレクトリ内のすべてのCSVファイルを取得
    file_paths = glob.glob('生データ/手配必要数/*.csv')

    # 統合結果を保存するリスト
    all_data = []

    # 各CSVファイルに対して処理を実行
    for file_name in file_paths:
        # ファイル名から年と月を抽出
        year_month = re.findall(r'\d{6}', os.path.basename(file_name))[0]  # ファイル名の頭6文字から年と月を抽出
        year = int(year_month[:4])
        month = int(year_month[4:6])

        # CSVファイルを読み込む
        # todo 必要数のみ読み取り
        df_raw = pd.read_csv(file_name, encoding='shift_jis', skiprows=9, usecols=range(70))  # 10行目から読み込むために9行スキップ

        # 列名のクリーニング
        df_raw.columns = df_raw.columns.str.replace('="', '').str.replace('"', '')

        # 日量数列の選択
        daily_columns = df_raw.columns[df_raw.columns.str.contains(r'\d+\(.*\)')].tolist()
        print(daily_columns)
        df_relevant = df_raw[['品番', '収容数', 'サイクル間隔', 'サイクル回数', 'サイクル情報'] + daily_columns]

        # データフレームを縦に展開
        df_melted = df_relevant.melt(id_vars=['品番', '収容数', 'サイクル間隔', 'サイクル回数', 'サイクル情報'], var_name='日付', value_name='日量数')

        # 日付の列を整数型に変換
        df_melted['日付'] = df_melted['日付'].str.extract(r'(\d+)').astype(int)

        # 日付列に年と月を統合
        df_melted['日付'] = pd.to_datetime(df_melted.apply(lambda row: f"{year}-{month}-{row['日付']}", axis=1))

        # 値クリーニング
        df_melted['日量数'] = df_melted['日量数'].str.replace(',', '').fillna(0).astype(int)
        df_melted['品番'] = df_melted['品番'].str.replace('="', '').str.replace('"', '').str.replace('=', '').str.replace('-', '')
        df_melted['収容数'] = df_melted['収容数'].str.replace(',', '').fillna(0).astype(int)
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

    # すべてのデータフレームを統合
    df_final = pd.concat(all_data, ignore_index=True)

    # 週最大日量数の計算
    df_final['週最大日量数'] = df_final.groupby(['品番', '週番号'])['日量数'].transform('max')
    df_final['週最大日量数（箱数）'] = df_final['週最大日量数']//df_final['収容数']

    #設計値MIN
    df_final['設計値MIN'] = 0.1*(df_final['週最大日量数（箱数）']*df_final['サイクル間隔']*(1+df_final['サイクル情報'])/df_final['サイクル回数'])
    df_final['設計値MAX'] = df_final['設計値MIN'] + df_final['週最大日量数（箱数）']/df_final['サイクル回数']

    return df_final
    
def calculate_AutomatedRack_Details(zaiko_df):

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

def calculate_supplier_truck_arrival_types():

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
    #file_path = '生データ/便ダイヤ/仕入先便ダイヤ.xlsx'#こっちは無理
    file_path = "生データ\便ダイヤ\仕入先便ダイヤ.xlsx"

    # openpyxlエンジンを使用してExcelファイルを読み込む
    df = pd.read_excel(file_path, engine='openpyxl')

    # 5行目を列名として設定し、6行目以降のデータを抽出
    df.columns = df.iloc[5]
    df = df[6:]

    # 列名をリセット
    df.columns = df.columns.str.strip()
    df.reset_index(drop=True, inplace=True)

    # 抽出したい列名を指定
    columns_to_extract = ['仕入先名', '発送場所名', '受入', 'タイプ', '1便', '2便', '3便', '4便', '5便', '6便', '7便', '8便', '9便', '10便', '11便', '12便']
    #columns_to_extract = ['仕入先名', '発送場所名', '受入', 'タイプ','4便', '5便', '6便']

    # 指定した列のみを抽出
    extracted_df = df[columns_to_extract]

    # 関数を使用して早着、定刻、遅着の時間帯を計算
    arrival_times_df = calculate_arrival_times(extracted_df, columns_to_extract[4:])

    # NaNの値を'< NULL >'に置換
    # 所在管理と結合するため
    arrival_times_df = arrival_times_df.fillna('< NULL >')

    return arrival_times_df

def calculate_weighted_average_of_kumitate():

    def set_A_B_columns(row, df):

        if row['TYOKU_KBN(1)'] == 1:
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
                    #if row['KUMI_CD'] == 'NH11':
                            #print(row['時間割区分_開始時刻'],len(previous_product_cnt))
                    if not previous_product_cnt.empty:
                        if len(mapping) == 4:
                            row['生産台数_加重平均済'] = (row['PRODUCT_CNT'] * weight + previous_product_cnt.iloc[0] * weight)
                            row['計画生産台数_加重平均済'] = (row['PLAN_PRODUCT_CNT'] * weight + previous_plan_product_cnt.iloc[0] * weight)
                            #if row['KUMI_CD'] == 'NH12':
                                #print(row['時間割区分_開始時刻'],previous_product_cnt.iloc[0],row['PRODUCT_CNT'],len(previous_product_cnt))
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
                    
        elif row['TYOKU_KBN(1)'] == 2:
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
                10: ('6:00', '６:00', None, 1),
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

    #MBから吸い出したもの
    file_path_kumitatedaisu = '組立生産台数.csv'
    kumitate_data = pd.read_csv(file_path_kumitatedaisu, encoding='shift_jis')
    # 'LINE_DATE'列をdatetime型に変換
    # 'LINE_DATE'列は常に0:00を表す
    kumitate_data['LINE_DATE'] = pd.to_datetime(kumitate_data['LINE_DATE'])

    print(len(kumitate_data.columns))

    # 'PLAN_PRODUCT_CNT' にNaNがある行を削除
    # ★関数通す前にこれしないと、NH11とNH12の結果が関数でヒットして、NH12の昼勤計算がうまくいかなくなる
    kumitate_data = kumitate_data.dropna(subset=['PRODUCT_CNT'])

    # すべての列をfloat型に変換
    #kumitate_data[['PLAN_PRODUCT_CNT','PRODUCT_CNT','TYOKU_KBN(1)']] = kumitate_data[['PLAN_PRODUCT_CNT','PRODUCT_CNT','TYOKU_KBN(1)']].astype(float)

    # 関数を適用
    kumitate_data = kumitate_data.apply(lambda row: set_A_B_columns(row, kumitate_data), axis=1)

    kumitate_data['計画達成率_加重平均済'] = kumitate_data['生産台数_加重平均済']/kumitate_data['計画生産台数_加重平均済']

    # 'PLAN_PRODUCT_CNT' にNaNがある行を削除
    kumitate_data = kumitate_data.dropna(subset=['PRODUCT_CNT'])

    # '計画達成率_加重平均済' 列の NaN を 0 に置き換える
    #kumitate_data['計画達成率_加重平均済'] = kumitate_data['計画達成率_加重平均済'].fillna(0)
    #kumitate_data['生産台数_加重平均済'] = kumitate_data['生産台数_加重平均済'].fillna(0)
    #kumitate_data['計画生産台数_加重平均済'] = kumitate_data['計画生産台数_加重平均済'].fillna(0)

    # LINE_DATE_修正済と調整日時を結合して新しい列Xを作成
    kumitate_data['LINE_DATE_修正済'] = pd.to_datetime(kumitate_data['LINE_DATE_修正済'])
    # 調整日時 も datetime 型に変換
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

    return kumitate_data

#途中作成のまま
def calculate_teikibin_():

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
            #df[f'{i}時間前荷役時間'] = df['荷役時間'].shift(i)
            df[f'荷役時間(t-{i})'] = df['荷役時間'].shift(i)
        return df

    #ファイル読み込み
    file_path_teikibin = '定期便.csv'
    teikibin_data = pd.read_csv(file_path_teikibin, encoding='shift_jis')

    # 日時の列を datetime 型に変換
    teikibin_data['JISEKI_DT'] = pd.to_datetime(teikibin_data['JISEKI_DT'])
    teikibin_data['JISEKI_DT2'] = pd.to_datetime(teikibin_data['JISEKI_DT2'])
    #1時間単位に変換
    teikibin_data['定期便到着時刻（1H）'] = pd.to_datetime(teikibin_data['JISEKI_DT']).dt.floor('H')
    teikibin_data['定期便出発時刻（1H）'] = pd.to_datetime(teikibin_data['JISEKI_DT2']).dt.floor('H')
    # 日時の差を計算
    teikibin_data["荷役時間"] = teikibin_data['JISEKI_DT2'] - teikibin_data['JISEKI_DT']

    # 各WORK_IDと定期便到着時刻（1H）の組み合わせに対して荷役時間の合計を計算
    grouped = teikibin_data.groupby(['WORK_ID', '定期便到着時刻（1H）'])['荷役時間'].sum().reset_index()

    # 1時間毎のデータフレームに各WORK_IDごとの「定期便到着時刻（1H）」列を追加する
    date_range = pd.date_range(start = start_date, end = end_date, freq='H')

    # YYYYMMDDHに全ての時間帯をマッピング
    all_hours_df = pd.DataFrame(date_range, columns=['日時']).set_index('日時')

    # 結果を保存するための空のDataFrameを準備
    result_df = all_hours_df.copy()

    # 元のデータセットからユニークなWORK_IDを抽出する
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
    X = 7  # 1時間前から5時間前までのデータ列を追加
    teikibin_df = add_previous_hours_data(result_df_reset, X)

    # 統合したデータを新しいCSVファイルに保存
    with open(file_path_teikibin2, mode='w',newline='', encoding='shift_jis',errors='ignore') as f:
        teikibin_df.to_csv(f)

    # 更新されたデータフレームの最初の数行を表示して内容を確認
    teikibin_df.head()

    # 結果の一部を表示して確認
    # 日付範囲に基づいてデータをフィルタリング
    start_date_temp = pd.Timestamp('2023-10-05')
    end_date_temp = pd.Timestamp('2023-10-06')
    f = teikibin_df[(teikibin_df['日時'] >= start_date_temp) & (teikibin_df['日時'] <= end_date_temp)]
    f.head(20)