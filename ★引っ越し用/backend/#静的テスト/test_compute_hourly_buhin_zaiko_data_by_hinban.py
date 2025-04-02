import pandas as pd
import os
#os.add_dll_directory('C:/Program Files/IBM/IBM DATA SERVER DRIVER/bin')
#import ibm_db
#import pyodbc

# 品番＆受入整備室毎の部品在庫を抽出する関数
def compute_hourly_buhin_zaiko_data_by_hinban(hinban_info, start_datetime, end_datetime):

    
    """
    """

    # T403物流情報_在庫推移テーブル（T403とT157）の在庫データを読み込み
    def load_zaiko_data_from_Drsum( hinban, start_datetime, end_datetime):

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
            WHERE 品番 = '{}', 更新日時 >= '{}' AND 更新日時 <= '{}'
        """.format(hinban, start_datetime, end_datetime)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        df = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()

        return df


    # 拠点諸番地を整備室名に置き換えるためのマスターを作成する関数
    def map_kyotenshobanchi_to_seibishitsu():

        """
        """

        # 品番&整備室&拠点所番地のユニークな組み合わせを計算する
        #todo 品番＆拠点所番地のユニークな組み合わせを抽出する（DISTINCT）関数の実行

        return master_kyotenshobanchi_to_seibishitsu

    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]
    # 将来）T403とT447を識別できるようにする

    # 対象品番＆対象期間のデータを抽出する
    buhin_zaiko_data_by_hinban_df = load_zaiko_data_from_Drsum(hinban, start_datetime, end_datetime)

    # 対象整備室のデータを抽出する
    # まず品番、整備室、拠点所番地のマスターを作成する
    master_kyotenshobanchi_to_seibishitsu = map_kyotenshobanchi_to_seibishitsu()
    # 統合
    # 対象整備室だけ抽出
    buhin_zaiko_data_by_hinban_df = 0

    return buhin_zaiko_data_by_hinban_df

