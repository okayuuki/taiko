import pandas as pd

# 品番＆受入整備室毎の部品在庫を抽出する関数
def compute_hourly_buhin_zaiko_data_by_hinban(hinban_info, start_datetime, end_datetime):

    
    """

    以下を実現する関数:
    1.　対象品番＆対象期間の自動ラックの在庫データの読み込み
    2.　拠点所番地と整備室を紐づけるマスター作成
    3.　在庫データに整備室情報の追加
    4.　必要な列情報を選択
    
    Parameters:
    hinban_info（list）：ある品番
    start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
    end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）
    
    Returns:
    buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

    """

    # T403物流情報_在庫推移テーブル（T403とT157）の在庫データを読み込み
    def load_zaiko_data_from_Drsum( hinban, start_datetime, end_datetime):

        """
        対象品番＆対象期間の自動ラックの在庫データの読み込む

        Parameters:
        hinban_info（str）：ある品番
        start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
        end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）

        Returns:
        buhin_zaiko_data_by_hinban_df (pd.DataFrame):品番毎1時間毎の在庫データ

        """

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
        buhin_zaiko_data_by_hinban_df = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()

        return buhin_zaiko_data_by_hinban_df

    # 拠点諸番地を整備室名に置き換えるためのマスターを作成する関数
    def map_kyotenshobanchi_to_seibishitsu( start_datetime, end_datetime):

        """
        """

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

        # SQL文の作成（ユニークな品番と整備室を抽出）
        sql = """
            SELECT DISTINCT 品番, 整備室コード, 拠点所番地
            FROM T403物流情報_所在管理_リードタイム
            WHERE 更新日時 >= '{}' AND 更新日時 <= '{}'
        """.format(start_datetime, end_datetime)

        # SQL文の実行
        cur.execute(sql)

        # 結果をデータフレームに読み込み
        master_kyotenshobanchi_to_seibishitsu = pd.read_sql(sql, con=connection)

        # 接続を閉じる
        cur.close()
        connection.close()

        return master_kyotenshobanchi_to_seibishitsu

    # 品番情報設定
    hinban = hinban_info[0]
    seibishitsu = hinban_info[1]
    # 将来）T403とT447を識別できるようにする

    # 対象品番＆対象期間のデータを抽出する
    buhin_zaiko_data_by_hinban_df = load_zaiko_data_from_Drsum(hinban, start_datetime, end_datetime)

    # 対象整備室のデータを抽出する
    # 品番、整備室、拠点所番地のマスターを作成する
    master_kyotenshobanchi_to_seibishitsu = map_kyotenshobanchi_to_seibishitsu( start_datetime, end_datetime)
    # 統合
    buhin_zaiko_data_by_hinban_df = pd.merge(buhin_zaiko_data_by_hinban_df, master_kyotenshobanchi_to_seibishitsu, on=['品番', '拠点所番地'], how='inner')

    # 対象整備室だけ抽出
    buhin_zaiko_data_by_hinban_df = 0

    return buhin_zaiko_data_by_hinban_df

# 品番＆受入整備室毎のかんばんデータを関所毎に抽出する関数
def compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime):#現状、納入予定時間

    return 








def main(start_datetime, end_datetime):

    """

    以下を実現する関数:
    ・かんばんタイムスタンプデータの読み込み
    ・自動ラックの在庫データの読み込み
    ・手配必要数＆手配運用情報データの読み込み
    
    Parameters:
    start_datetime (str): 開始日時（YYYY-MM-DD-HH形式）
    end_datetime (str): 終了日時（YYYY-MM-DD-HH形式）
    
    Returns:
    df_kanban_timestamp_data (pd.DataFrame):かんばんタイムスタンプデータ
    df_lack_zaiko_data (pd.DataFrame):自動ラックの在庫データ
    df_tehai_data (pd.DataFrame):手配必要数＆手配運用情報データ

    """

    #! かんばんタイムスタンプデータの読み込み

    #! 自動ラックの在庫データの読み込み

    #! 手配必要数＆手配運用情報データの読み込み

    
    return 

def load_kanban_timestamp_data():

    #! 所在管理MBの読み込み

    #! 仕入先ダイヤの読み込み

    #! 統合

    return

def load_lack_zaiko_data():

    return

def load_tehai_data():

    return