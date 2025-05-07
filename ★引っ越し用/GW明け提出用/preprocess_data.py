import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import Parallel, delayed
import time
import numpy as np

from get_data import compute_hourly_buhin_zaiko_data_by_hinban,\
    compute_hourly_specific_checkpoint_kanbansu_data_by_hinban,\
        compute_hourly_tehai_data_by_hinban,\
            compute_hourly_chakou_data_by_hinban,\
                get_kado_schedule_from_172_20_113_185

# MARK: データ結合
#todo 在庫推移と時間ズレがある
def merge_data(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

    #　時間粒度設定
    time_granularity = 'h'

    #　並列処理（マルチプロセス）
    def run_parallel_processing(process_number, hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

        # プロセス1の処理
        if process_number == 0:

            # 自動ラックの在庫データの読み込み
            st.header("在庫データの読み込み")
            zaiko_df = compute_hourly_buhin_zaiko_data_by_hinban(hinban_info, start_datetime, end_datetime, flag_useDataBase, kojo)
            st.dataframe(zaiko_df)

            # 関所後のかんばんデータ読み込み
            st.header("関所毎のかんばんデータの読み込み")
            target_column = '発注日時'
            haccyu_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(haccyu_kanban_df)

            target_column = '納入予定日時'
            nonyu_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(nonyu_kanban_df)

            target_column = '順立装置入庫日時'
            nyuuko_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(nyuuko_kanban_df)

            target_column = '順立装置出庫日時'
            syukko_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(syukko_kanban_df)

            target_column = '回収日時'
            kaisyu_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(kaisyu_kanban_df)
            
            target_column = '西尾東~部品置き場の間の滞留'
            tairyu_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(tairyu_kanban_df)

            target_column = '期待かんばん在庫'
            kako_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(kako_kanban_df)

            target_column = '順立装置内の滞留と前倒し出庫の差分'
            abnornal_syukko_kanban_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(abnornal_syukko_kanban_df)

            return zaiko_df, haccyu_kanban_df, nonyu_kanban_df, nyuuko_kanban_df, syukko_kanban_df, kaisyu_kanban_df, tairyu_kanban_df, kako_kanban_df, abnornal_syukko_kanban_df

        # プロセス2の処理
        elif process_number == 1:

            # 手配データの読み込み
            st.header("手配データの読み込み")
            tehai_df = compute_hourly_tehai_data_by_hinban(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            st.dataframe(tehai_df)

            return tehai_df

        # プロセス3の処理
        elif process_number == 2:

            # 着工データの読み込み
            st.header("着工データの読み込み（刈谷地区だと2分程度時間かかっています）")
            seisan_buturyu_df = compute_hourly_chakou_data_by_hinban(hinban_info, start_datetime, end_datetime)
            st.dataframe(seisan_buturyu_df)

            # target_column = '順立装置出庫日時'
            # seisan_buturyu_df, _ = compute_hourly_specific_checkpoint_kanbansu_data_by_hinban(hinban_info, target_column, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
            # st.dataframe(seisan_buturyu_df)

            return seisan_buturyu_df

        # プロセス4の処理
        elif process_number == 3:

            # 稼働時間データの読み込み
            st.header("稼働データの読み込み")
            day_col='確定(昼)'
            night_col='確定(夜)'
            kado_df = get_kado_schedule_from_172_20_113_185(start_datetime, end_datetime, day_col, night_col, time_granularity)
            st.dataframe(kado_df)

            return kado_df

    n_jobs = 4

    # # 並列処理
    # start_time_parallel = time.time()
    # results_parallel = Parallel(n_jobs=n_jobs)(
    #     delayed(run_parallel_processing)(process_number, hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo) for process_number in range(4)
    # )
    # parallel_time = time.time() - start_time_parallel

    # 逐次処理
    start_time_sequential = time.time()
    results_sequential = [
        run_parallel_processing(process_number, hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo) for process_number in range(4)
    ]
    sequential_time = time.time() - start_time_sequential
    results_parallel = results_sequential

    #実行結果の確認
    #print(f"並列処理時間: {parallel_time:.2f}秒")
    #print(f"逐次処理時間: {sequential_time:.2f}秒")
    #print(f"速度向上率: {sequential_time/parallel_time:.2f}倍")

    # 各プロセスの結果を変数に引き渡す
    zaiko_df, haccyu_kanban_df, nonyu_kanban_df, nyuuko_kanban_df, syukko_kanban_df, kaisyu_kanban_df, tairyu_kanban_df, kako_kanban_df, abnornal_syukko_kanban_df = results_parallel[0]
    tehai_df = results_parallel[1]
    seisan_buturyu_df = results_parallel[2]
    kado_df = results_parallel[3]

    # 統合
    st.header("データの統合")

    #　参考
    #  日時　入庫数　出庫数　在庫数　滞留数
    #  17時　
    #  18時　10個　　1個　　10個    2個
    #  19時　 0個　　0個　　17個
    # 
    #　※　在庫は○○時点の値

    # 自動ラックの在庫データ + 関所毎のかんばんデータ（納入予定日時）
    merge_data_df = pd.merge(zaiko_df, nonyu_kanban_df, on=['日時'], how='left')

    # 統合データ　+ 関所毎のかんばんデータ（発注日時）
    merge_data_df = pd.merge(merge_data_df, haccyu_kanban_df, on=['日時'], how='left')
    merge_data_df['発注日時のかんばん数'] = merge_data_df['発注日時のかんばん数'].shift(1)

    # 統合データ　+ かんばんデータ（入庫日時）
    merge_data_df = pd.merge(merge_data_df, nyuuko_kanban_df, on=['日時'], how='left')
    #! 在庫が増えるときに、入庫数をつけるために、１つ下の行にシフトさせる
    merge_data_df['順立装置入庫日時のかんばん数'] = merge_data_df['順立装置入庫日時のかんばん数'].shift(1)

    # 統合データ　+ かんばんデータ（出庫日時）
    merge_data_df = pd.merge(merge_data_df, syukko_kanban_df, on=['日時'], how='left')
    #! 在庫が減るるときに、出庫数をつけるために、１つ下の行にシフトさせる
    merge_data_df['順立装置出庫日時のかんばん数'] = merge_data_df['順立装置出庫日時のかんばん数'].shift(1)

    # 統合データ　+ 関所毎のかんばんデータ（回収日時）
    merge_data_df = pd.merge(merge_data_df, kaisyu_kanban_df, on=['日時'], how='left')
    merge_data_df['回収日時のかんばん数'] = merge_data_df['回収日時のかんばん数']

    # 統合データ　+ かんばんデータ（西尾東～部品置き場の間の滞留かんばん数）
    merge_data_df = pd.merge(merge_data_df, tairyu_kanban_df, on=['日時'], how='left')
    #! 在庫が増える1つ前まで滞留しているようにしておくためにshift無し
    merge_data_df['西尾東~部品置き場の間の滞留かんばん数'] = merge_data_df['西尾東~部品置き場の間の滞留かんばん数_枚数単位']

    # 統合データ　+ かんばんデータ（期待かんばん在庫数）
    merge_data_df = pd.merge(merge_data_df, kako_kanban_df, on=['日時'], how='left')
    merge_data_df['期待かんばん在庫数'] = merge_data_df['期待かんばん在庫数']

    # 統合データ　+ かんばんデータ（順立装置内の滞留と前倒し出庫の差分数）
    merge_data_df = pd.merge(merge_data_df, abnornal_syukko_kanban_df, on=['日時'], how='left')
    merge_data_df['順立装置内の滞留と前倒し出庫の差分_h単位'] = merge_data_df['順立装置内の滞留と前倒し出庫の差分_時間単位'].shift(1)

    #! 在庫が増えときに入庫があるようにする
    merge_data_df["入庫（箱）"] = merge_data_df["入庫（箱）"].shift(1)

    # 統合データ　+ 手配データ
    merge_data_df = pd.merge(merge_data_df, tehai_df, on=['日時'], how='left')

    # 統合データ + 着工データ
    merge_data_df = pd.merge(merge_data_df, seisan_buturyu_df, on=['日時'], how='left')

    # 統合データ + 稼働時間データ
    merge_data_df = pd.merge(merge_data_df, kado_df, on=['日時'], how='left')

    # 数値列のみを選択してNaN/Noneを0に置換
    #! これしないと平均計算などが期待通りにならない。pythonのmeanはNoneを無視する
    numeric_columns = merge_data_df.select_dtypes(include=['int64', 'float64']).columns
    merge_data_df[numeric_columns] = merge_data_df[numeric_columns].fillna(0)

    # 実行結果の保存
    merge_data_df.to_csv('merge_data関数_統合データ.csv', index=False, encoding='shift_jis')

    return merge_data_df

# MARK:目的変数と説明変数の決定
#todo 長期休暇除去
def compute_features_and_target(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo):

    # targetの計算
    def compute_target_variable(df, datetime_column, input_column, output_column, target_column, method, trim_ratio):
        
        """
        指定された集計方法（平均・中央値・トリム平均）に基づいて、
        各時刻（hour）ごとの在庫数の「いつもの傾向（目的変数）」を計算し、
        指定の列に出力する関数。

        処理の流れ：
        1. datetime列をdatetime型に変換し、そこからhour（時刻）を抽出する。
        2. hour（時刻）単位で在庫数をグルーピングする。
        3. 指定された手法（method）で各hourの代表値を算出する：
        - 'mean'：平均
        - 'median'：中央値
        - 'trim_mean'：上下一定割合を除外して平均
        4. 各行の時刻（hour）に応じて、算出した「いつもの在庫数」をoutput_columnに記録する。
        5. input_column - output_column の差を取り、目的変数（target_column）として記録。

        Parameters:
            df (pd.DataFrame): 対象データフレーム（日時と在庫数を含む）
            datetime_column (str): 日時列の列名（datetime型である必要があるが、内部で変換も実施）
            input_column (str): 現在の在庫数が記録されている列
            output_column (str): 「いつもの在庫数」を格納する列
            target_column (str): 差分としての目的変数を格納する列
            method (str): 集計方法。'mean', 'median', 'trim_mean' のいずれかを指定
            trim_ratio (float): トリム平均を使う場合に上下で除外する割合（例：0.1 → 上下10%ずつ除く）

        Returns:
            pd.DataFrame: 新しい列（output_column）に目的変数を追加したデータフレーム

        Example:
            入力データ：

                日時               | 在庫数
                -------------------|--------
                2025-03-29 09:00   |   50
                2025-03-28 09:00   |   55
                2025-03-27 09:00   |   53
                2025-03-29 15:00   |   42
                2025-03-28 15:00   |   45

            method='median' の場合：
                - 9時の中央値 = 53 → output_column = 53
                - 15時の中央値 = 43.5 → output_column = 43.5
                - 目的変数（target_column）＝ 在庫数 − output_column

            結果：
                在庫数 50 → 差分 = -3（= 50 - 53）
                在庫数 42 → 差分 = -1.5（= 42 - 43.5）
        """

        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
        df['hour'] = df[datetime_column].dt.hour

        # 時刻ごとに処理を分岐
        grouped = df.groupby('hour')[input_column]

        if method == 'mean':
            hour_values = grouped.mean().to_dict()

        elif method == 'median':
            hour_values = grouped.median().to_dict()

        elif method == 'trim_mean':
            # 各 group を個別に処理して、トリム平均を計算
            hour_values = {
                hour: trim_mean(values, proportiontocut=trim_ratio)
                for hour, values in grouped
            }
        else:
            raise ValueError(f"不正なmethod指定: {method}. 'mean', 'median', 'trim_mean' のいずれかにしてください。")

        # 各行の hour に応じて「いつもの在庫数」を設定
        df[output_column] = df['hour'].map(hour_values)

        # 補助列を削除
        df.drop(columns=['hour'], inplace=True)

        # 目的変数を計算（現在の在庫数との差分）
        df[target_column] = df[input_column] - df[output_column]

        return df

    # feature_No1の計算
    # 実績（入庫実績）と設計（入庫予定）のデータを±X時間のタイムトレランス内でスナップ処理しマッチングさせる
    # 納入予定のある行が、近くの実績がある行にスナップされるようにする
    def compute_nyuuko_yotei_kanbansu_by_snapping_with_tolerance( df, kado_column, input_column,
                                                                  shiftted_column, lt_column, base_column, snap_column, snap_column_abnormal, time_tolerance):

        # 入庫予定かんばん数列の追加
        def compute_nyuuko_yotei_kanbansu( df, kado_column, input_column, shiftted_column, lt_column):

            """
            入庫予定かんばん数（shiftted_column）を計算してDataFrameに追加する関数
            shiftted_columnの値は、input_columnの値をリードタイムの値分先に進んだ行にシフトした値

            処理の流れ：
            1. input_column（例：発注かんばん数）が0でない行のみを対象とする。
            2. 各対象行について、リードタイム（lt_column）を取得し、
            3. 稼働フラグ（kado_column）が0でない行をリードタイム件数分カウントしながら進める。
            4. リードタイム後の行が存在すれば、その行のshiftted_columnにinput_columnの値を加算する。

            Parameters:
                df (pd.DataFrame): 入出庫データを含むデータフレーム
                kado_column (str): 稼働フラグ列名（1=稼働、0=非稼働）
                input_column (str): 入庫元データ列名（例：発注かんばん数）
                shiftted_column (str): 出力列名（計算結果として追加する列）
                lt_column (str): リードタイム（日数）を示す列名

            Returns:
                pd.DataFrame: 入庫予定かんばん数列を追加したデータフレーム

            Example:
                以下のようなDataFrameがあったとする（リードタイム = 3）:

                    index | 稼働フラグ  | 発注かんばん数  | 納入予定かんばん数
                    ------|------------|----------------|-------------------
                    0     |     1      |      10        |        0
                    1     |     0      |       0        |        0
                    2     |     1      |       0        |        0
                    3     |     1      |       0        |        0
                    4     |     1      |       0        |       10  ← ここに納入予定が記録される！

                → index=0 の発注（10）は、リードタイム3行分の稼働である index=4 にスライドされ、
                その行の「納入予定かんばん数」に加算される。

                shiftted_column（納入予定）は、input_column（発注）より
                後ろの行に記録される点に注意
            """

            # shiftted_column列を初期化
            df[shiftted_column] = 0

            # 全行を走査して、input_column列の値 ≠ 0 の行のみ処理する
            for idx in df.index:

                # input_column列の値 = 0 ならその行はスキップする（入庫予定かんばん数の計算に関係がないため）
                # mergedの方でNoneやNanを0にできているはずだけど、念のため、NoneやNanも条件に含める
                # ここで0やNone、NaNをスキップできないと、後の処理が期待通りにならないので、注意すること
                if df.at[idx, input_column] == 0 or pd.isna(df.at[idx, input_column]):
                    continue

                # input_column列の値 ≠ 0 なら、リードタイムの値を確認する
                # lead_time = リードタイムの値
                lead_time = df.at[idx, lt_column]

                count = 0
                cursor = idx #リードタイム後のインデックス
                # リードタイム日数分だけ「稼働フラグが1の行」をカウントして進める
                # count = lead_time が基本終了条件
                # cursor + 1 < len(df) で データの末尾に到達しないように制限
                while count < lead_time and cursor + 1 < len(df):
                    cursor += 1
                    # 稼働フラグ列の値が0でない数を数える
                    if df.at[cursor, kado_column] != 0:
                        count += 1
                #st.write(df.at[idx, "日時"],df.at[idx, input_column],count)

                # リードタイム先が範囲内であればinput_column列の値をlead_time分シフトして、shiftted_column列の値に加算する
                if count == lead_time:
                    df.at[cursor, shiftted_column] += df.at[idx, input_column]

            return df
        
        # ±X時間のタイムトレランス内でスナップ処理
        def snap_with_tolerance(df, kado_column, shiftted_column, base_column, snap_column, time_tolerance):

            """
            ±X時間（稼働時間）以内にbase_columnの値がある場合に、shiftted_columnの値をスナップ（吸着）させる関数。

            処理の流れ：
            1. shiftted_column（納入予定かんばん数）に値がある行のみ対象とする。
            2. その行を中心として、稼働フラグが0でない前後time_tolerance件分の稼働行を取得する。
            3. その範囲内に base_column（納入実績）が0でない行があれば、その最初の行に planned 値をスナップする。
            4. 実績が見つからなければ、自分自身の行に planned を保持する。

            Parameters:
                df (pd.DataFrame): 処理対象のデータフレーム
                kado_column (str): 稼働フラグ列（1=稼働、0=非稼働）
                shiftted_column (str): 納入予定かんばん数の列（スナップ対象）
                base_column (str): 納入実績かんばん数の列（スナップ先の基準）
                snap_column (str): 結果を格納する列（スナップ後の納入予定）
                time_tolerance (int): 許容する稼働日数（前後X件の稼働日）

            Returns:
                pd.DataFrame: snap_columnにスナップ結果を格納したデータフレーム

            Example:
                以下のようなデータを想定（time_tolerance=2）:

                    index | 稼働フラグ  | 納入予定　| 納入実績 | スナップ結果
                    ------|------------|----------|----------|--------------
                    0     |     1      |    5     |    0     |      0
                    1     |     1      |    0     |    4     |      5  ← index=0の予定がここにスナップ！
                    2     |     1      |    0     |    0     |      0
                    3     |     1      |    3     |    0     |      3  ← 実績がなければ自分自身に残る

                → スナップの対象は、納入予定がある行のみ（shiftted_column ≠ 0）
                → 前後の稼働行に納入実績があるかを調べ、最初に見つかった行へ移す
                → 見つからなければ、自身の行にそのまま残す
            """
            
            # 新しい列を0で初期化
            df[snap_column] = 0

            # 稼働日のインデックス一覧
            kado_indices = df[df[kado_column] != 0].index.tolist()

            for idx in df.index:
                planned = df.at[idx, shiftted_column]

                # 納入予定がない場合はスキップ
                if planned == 0:
                    continue

                # 現在の行が稼働日リストの何番目かを取得（見つからない場合はスキップ）
                try:
                    kado_pos = kado_indices.index(idx)
                except ValueError:
                    continue  # 稼働フラグ=0の行は対象外

                # 前後X件の稼働インデックス（自分を除く）
                nearby_kado_indices = kado_indices[max(0, kado_pos - time_tolerance):kado_pos] + \
                                    kado_indices[kado_pos + 1:kado_pos + 1 + time_tolerance]

                # 前後の稼働行の中で、納入かんばん数 ≠ 0 の行を探す（上から順に）
                matched_idx = None
                for near_idx in nearby_kado_indices:
                    if df.at[near_idx, base_column] != 0:
                        matched_idx = near_idx
                        break  # 一番最初に見つけた行でOK

                # 見つかればその行にコピー、見つからなければ自分にコピー
                if matched_idx is not None:
                    df.at[matched_idx, snap_column] += planned
                # 見つからない場合
                else:
                    #df.at[idx, snap_column] += planned#元の行に入れる
                    #df.at[idx, snap_column_abnormal] += planned #もとの行の違う列に入れる
                    df.at[idx, snap_column] = 0

            return df

        df[snap_column_abnormal] = 0
        
        # 入庫予定かんばん数列の追加
        df = compute_nyuuko_yotei_kanbansu(df, kado_column, input_column, shiftted_column, lt_column)

        # 入庫かんばん数_スナップ済の追加
        df = snap_with_tolerance(df, kado_column, shiftted_column, base_column, snap_column, time_tolerance)

        # 既存dfに新しい列を追加して返す
        return df

    # feature_No2の計算
    # 直近のX時間の着工数を活用して、直近の生産状況を定量化する
    def quantify_recent_chakkousuu_status_and_trend(df, kado_column,
                                                              chakkou_column, window,
                                                              recent_chakkousuu_status_and_trend_column):
        
        """
        稼働していた直近X時間分のデータから、1分あたりの着工数を
        「着工数 ÷（稼働フラグ × 60）」の平均として定量化する関数。

        処理の流れ：
        1. 各行について、その行までのデータを時系列で遡って取得する。
        2. その中から「稼働フラグ > 0」の行のみを抽出する。
        3. 抽出された稼働行の中から直近 window 件分を取得する。
        4. 各行で「着工数 ÷（稼働フラグ × 60）」を計算し、その平均値を
        現在の行の傾向値として記録する。

        Parameters:
            df (pd.DataFrame): 入力データフレーム。kado_column（稼働フラグ）と
                            chakkou_column（着工数）を含む必要がある。
            kado_column (str): 稼働フラグ列の名前（例：1 = 完全稼働、0.5 = 半稼働、0 = 非稼働）。
            chakkou_column (str): 着工数を示す列名。
            window (int): 稼働行ベースで何件分の履歴を使って傾向を算出するか。
            recent_chakkousuu_status_and_trend_column (str): 結果を格納する新しい列名。

        Returns:
            pd.DataFrame: 各行に「1分あたりの着工傾向」を示す新しい列を追加したDataFrame。

        Example:
            入力データ（window=3）の場合：

                index | 時刻   | 稼働フラグ | 着工数 | 傾向値（1分あたり）
                ------|--------|-----------|--------|---------------------
                0     | 8:00   |   1.0     |   60   |   1.00
                1     | 9:00   |   0.0     |    0   |   1.00
                2     | 10:00  |   0.5     |   30   |   1.00
                3     | 11:00  |   1.0     |   45   |   0.92
                4     | 12:00  |   1.0     |   48   |   0.90

            → 傾向値列は、各行で「着工数 ÷（稼働フラグ × 60）」を計算し、その平均で算出される。
            例：index=3 のとき、直近3件は：
                - 60 ÷ (1.0 × 60) = 1.00
                - 30 ÷ (0.5 × 60) = 1.00
                - 45 ÷ (1.0 × 60) = 0.75
                → 平均 = (1.00 + 1.00 + 0.75) / 3 = 0.92

            ※ 稼働フラグが0の行は除外される（ゼロ除算防止のため）。
        """

        df = df.copy()
        df[recent_chakkousuu_status_and_trend_column] = None

        for idx in df.index:
            # 現在の行までのデータを取得
            df_up_to_now = df.loc[:idx]

            # 稼働していた行のみを抽出（稼働フラグが0でない）
            valid_rows = df_up_to_now[df_up_to_now[kado_column] != 0]

            # 直近window件を取得
            recent_valid = valid_rows.tail(window)

            if not recent_valid.empty:
                # 各行で「着工数 ÷（稼働フラグ × 60）」を計算
                per_minute_values = recent_valid[chakkou_column] / (recent_valid[kado_column] * 60)

                # 平均値を傾向として記録
                mean_value = per_minute_values.mean()
            else:
                mean_value = 0

            # 傾向値を新しい列に記録
            df.at[idx, recent_chakkousuu_status_and_trend_column] = mean_value

        return df

    # 日単位でかんばんを集計
    def compute_daily_total( df, target_col, datetime_col='日時'):

        """
        指定された列の日次合計を計算する（8時から翌7時までを1日として集計）
        
        Parameters:
        -----------
        df : pandas.DataFrame
            入力データフレーム
        target_col : str
            集計対象の列名
        datetime_col : str, default='日時'
            日時が格納された列名
        
        Returns:
        --------
        pandas.DataFrame
            日次合計列が追加されたデータフレーム
        """

        df_copy = df.copy()
        
        # 日時を8時間ずらして新しい日付列を作成
        df_copy['集計日'] = df_copy[datetime_col].dt.date - pd.Timedelta(hours=8)
        
        # 集計日ごとに合計を計算
        daily_sum = df_copy.groupby('集計日')[target_col].sum().reset_index()
        st.write(daily_sum)
        
        # 元のデータフレームと結合
        df_copy = df_copy.merge(
            daily_sum,
            left_on='集計日',
            right_on='集計日',
            suffixes=('', '_日単位')
        )
        
        # 不要な集計日列を削除
        df_copy = df_copy.drop('集計日', axis=1)
        
        return df_copy

    # feature_No3の計算
    # 在庫水準
    # 現在庫は、過去にどれだけ発注され、どれだけ消費（回収）されたかの差分の累積によって形成される
    def compute_zaiko_level(df, kado_column, lt_column, input_column, window,
                            output_column):
        
        """
        稼働日ベースでLT時間分さかのぼり、そこからwindow件の移動平均を計算する。

        Parameters:
            df (pd.DataFrame): 対象のデータフレーム
            lt_column (str): LT（さかのぼる稼働時間）が入っている列名
            kado_column (str): 稼働フラグの列名（1=稼働、0=非稼働）
            input_column (str): 生産台数の列名
            window (int): 平均をとる稼働時間の行数
            output_column (str): 結果を保存する新しい列名
        """

        #todo 手配の方で計算してもいいかも
        df["かんばん回転日数"] = (df["サイクル間隔"] * (df["サイクル情報"] + 1)) / df["サイクル回数"]

        # 結果を保存する列を初期化
        df[output_column] = None

        # 稼働フラグが0でない行のインデックスをリスト化
        kado_indices = df[df[kado_column] != 0].index.tolist()

        # 全行をループ
        for current_index in df.index:

            # LTの値を取得（何件分さかのぼるか）
            lt_value = df.at[current_index, lt_column]

            # LTが欠損または数値でない場合はスキップ
            if pd.isna(lt_value) or not isinstance(lt_value, (int, float)):
                continue

            # 現在の行が稼働日インデックスの何番目かを取得
            try:
                current_kado_pos = kado_indices.index(current_index)
            except ValueError:
                continue  # 万一見つからなければスキップ

            # LT件分だけさかのぼる（位置ベース）
            start_kado_pos = current_kado_pos - int(lt_value)
            if start_kado_pos < 0:
                continue  # 範囲外ならスキップ

            # LT位置から window 件分の稼働日を取得
            target_kado_indices = kado_indices[start_kado_pos : start_kado_pos + window]

            # 平均を計算
            if target_kado_indices:
                production_values = df.loc[target_kado_indices, input_column]
                average_value = production_values.median()
                df.at[current_index, output_column] = average_value

        return df
    
    # 結果を表示する汎用関数
    def plot_result(df, datetime_column,
        value_columns,  # list型: 表示したい列（複数OK）
        flag_show, graph_title, yaxis_title,
        kado_column = None):
        
        """
        指定した日時列と複数の値列をPlotly+Streamlitで表示する汎用関数。

        Parameters:
            df (pd.DataFrame): 対象データ
            datetime_column (str): 時系列軸となる列（datetime型推奨）
            value_columns (list[str]): 可視化したい複数の列名（在庫数・いつも・差分など）
            flag_show (bool): 表示するかどうかのフラグ
            graph_title (str): グラフタイトル（任意指定可）
            yaxis_title (str): Y軸タイトル（任意指定可）
            kado_column (str or None): 稼働フラグの列名（任意）。指定されれば表に表示されます。
        """

        if not flag_show:
            return  # フラグがFalseなら何も表示しない

        # データの抽出と表示
        display_columns = [datetime_column] + value_columns
        if kado_column and kado_column in df.columns:
            display_columns.append(kado_column)

        temp_df = df[display_columns]
        st.dataframe(temp_df)

        # グラフ作成
        fig = go.Figure()

        # 各列をトレースとして追加
        dash_styles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']  # 複数用
        for i, col in enumerate(value_columns):
            fig.add_trace(go.Scatter(
                x=temp_df[datetime_column],
                y=temp_df[col],
                mode='lines+markers',
                name=col,
                line=dict(dash=dash_styles[i % len(dash_styles)])  # 見やすく複数対応
            ))

        # レイアウト調整（ホバーモード・タイトルなど）
        fig.update_layout(
            title=graph_title,
            xaxis_title=datetime_column,
            yaxis_title=yaxis_title,
            hovermode='x unified',  # ← ホバーで同一日時の値をまとめて表示
            legend_title='凡例',
            template='plotly_white'
        )

        # グラフ表示
        st.plotly_chart(fig, use_container_width=True)
    
    # 相関カラーマップ
    def create_correlation_plots(df, column_contains=[], cols_per_row=2):

        # 特定の文字列を含む列を選択
        selected_columns = df.columns[
            df.columns.str.contains('|'.join(column_contains))
        ]
        
        # 相関係数を計算
        correlation = df[selected_columns].corr()
        
        # targetカラムの確認
        target_col = [col for col in selected_columns if 'target' in col.lower()]
        if not target_col:
            return {"error": "No target column found"}
        target_col = target_col[0]
        
        # x軸の列（target以外）
        x_columns = [col for col in selected_columns if col != target_col]
        
        figures = {}
        
        # 1. ヒートマップの作成
        heatmap = go.Figure(data=[go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        )])
        
        heatmap.update_layout(
            title='Correlation Heatmap',
            width=800,
            height=800,
            xaxis_tickangle=45,
            template='plotly_white'
        )
        
        figures['heatmap'] = heatmap
        
        # 2. 散布図の作成
        for col in x_columns:
            scatter = go.Figure(data=[go.Scatter(
                x=df[col],
                y=df[target_col],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.6,
                    color='rgba(0, 116, 217, 0.7)'
                )
            )])
            
            scatter.update_layout(
                title=f'{col} vs {target_col}',
                xaxis_title=col,
                yaxis_title=target_col,
                width=600,
                height=600,
                template='plotly_white',
                showlegend=False
            )
            
            figures[f'scatter_{col}'] = scatter
        
        return figures

    # 統合データの読み込み
    merged_df = merge_data(hinban_info, start_datetime, end_datetime, time_granularity, flag_useDataBase, kojo)
    st.dataframe(merged_df)

    st.header("データの前処理")

    #todo 長期休暇削除
    # 日付範囲を指定して削除
    start_date = '2024-12-26'
    end_date = '2025-01-07'
    merged_df = merged_df[~(merged_df['日時'].between(start_date, end_date))]
    merged_df = merged_df.reset_index(drop=True)

    # ★使用する列名が変わる可能性があるため変数で定義する

    # 結果を描画するか判定するためのフラグ変数
    # 描画する：True、しない：False
    flag_show = True

    #!-----------------------------------------------------------------------
    #! 目的変数の設定
    #!-----------------------------------------------------------------------
    target_datetime_column = '日時'
    target_input_column = '在庫数（箱）'
    target_output_column = 'いつもの在庫数（箱）'
    target_column = 'target_在庫数（箱）-いつもの在庫数（箱）'
    target_method = 'median'
    target_trim_ratio = 0.1 #トリム平均を使う場合の切り捨て比率（例：0.1で上下10%カット）
    features_df = compute_target_variable(merged_df, target_datetime_column, target_input_column, target_output_column, target_column,
                            target_method, target_trim_ratio)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [target_input_column,target_output_column,target_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title='在庫数の推移（実測・いつも・差分）', yaxis_title = '在庫数（箱）')

    # 説明変数の計算

    #!-----------------------------------------------------------------------
    #! 入庫予定かんばん数_スナップ済を計算する
    #!-----------------------------------------------------------------------
    feature_No1_kado_column = '稼働フラグ'
    feature_No1_input_column = '納入予定日時のかんばん数'
    feature_No1_shiftted_column = '入庫予定かんばん数'
    feature_No1_lt_column = '納入LT(H)'
    feature_No1_base_column = "入庫（箱）" #todo 所在管理で再計算した方がいい？
    delay_start = 0
    delay_end = 0
    feature_No1_snap_column = f'feature_No1_入庫予定かんばん数_スナップ済（t-{delay_start}~t-{delay_end}）'
    feature_No5_snap_column_abnormal = f'feature_No5_設計外の入庫かんばん数（t-{delay_start}~t-{delay_end}）'
    feature_No1_time_tolerance = 1
    features_df = compute_nyuuko_yotei_kanbansu_by_snapping_with_tolerance( features_df, feature_No1_kado_column,
                                                                            feature_No1_input_column, feature_No1_shiftted_column,feature_No1_lt_column,
                                                                              feature_No1_base_column, feature_No1_snap_column, feature_No5_snap_column_abnormal,  feature_No1_time_tolerance)
    #!　設計外の入庫かんばん数を計算する
    # 新しい列 'feature_No5_snap_column_abnormal'に結果を格納
    features_df[feature_No5_snap_column_abnormal] = np.maximum(features_df[feature_No1_base_column] - features_df[feature_No1_snap_column], 0)
    # youin列作成
    youin_No1_column = f'youin_No1_入庫予定かんばん数_スナップ済（t-{delay_start}~t-{delay_end}）'
    features_df[youin_No1_column] = features_df[feature_No1_snap_column]
    youin_No5_column = f'youin_No5_設計外の入庫かんばん数（t-{delay_start}~t-{delay_end}）'
    features_df[youin_No5_column] = features_df[feature_No5_snap_column_abnormal]
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No1_input_column, feature_No1_shiftted_column, feature_No1_base_column, feature_No1_snap_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = 'IN関係の推移', yaxis_title = 'IN関係', kado_column = feature_No1_kado_column)

    #!-----------------------------------------------------------------------
    #! 直近のX時間の着工数を活用して、直近の生産状況を定量化する（旧No2）
    #!-----------------------------------------------------------------------
    feature_No2_kado_column = '稼働フラグ'
    feature_No2_chakkou_column = '生産台数'#'順立装置出庫日時のかんばん数'
    feature_No2_window = 40
    delay_start = feature_No2_window
    delay_end = 0
    # youin列作成
    youin_No2_column = f"youin_No2_順立装置内の滞留と前倒し出庫の差分数_間接生産要因（t-{delay_start}~t-{delay_end}）"
    #feature_No2_recent_chakkousuu_status_and_trend_column = f'feature_No2_最近の着工数の状況（t-{delay_start}~t-{delay_end}）'
    feature_No2_recent_chakkousuu_status_and_trend_column = '流動機種生産密度'
    features_df = quantify_recent_chakkousuu_status_and_trend(features_df, feature_No2_kado_column,
                                                              feature_No2_chakkou_column, feature_No2_window,
                                                              feature_No2_recent_chakkousuu_status_and_trend_column)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No2_chakkou_column,feature_No2_recent_chakkousuu_status_and_trend_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = '最近の着工数の推移', yaxis_title = '着工数', kado_column = feature_No2_kado_column)
    #!-----------------------------------------------------------------------
    #! 順立装置内の滞留と前倒し出庫の差分数（新No2）
    #!-----------------------------------------------------------------------
    delay_start = 0
    delay_end = 0
    feature_No2_output_column = f"feature_No2_順立装置内の滞留と前倒し出庫の差分数_間接生産要因（t-{delay_start}~t-{delay_end}）"
    features_df[feature_No2_output_column] = features_df["順立装置内の滞留と前倒し出庫の差分_枚数単位"]
    features_df[youin_No2_column] = features_df[feature_No2_recent_chakkousuu_status_and_trend_column]

    #!-----------------------------------------------------------------------
    #! 過去かんばん
    #todo 発注数と回収数がいる
    #todo 仕入先の稼働も入るから、アイシンの稼働フラグで計算するの難しいな
    #!-----------------------------------------------------------------------
    # 在庫水準の計算準備
    target_col_IN = '納入予定日時のかんばん数'
    target_col_IN_daily = f'{target_col_IN}_日単位'
    st.dataframe(features_df)
    features_df = compute_daily_total( features_df, target_col_IN, datetime_col='日時')
    target_col_OUT = '回収日時のかんばん数'
    target_col_OUT_daily = f'{target_col_OUT}_日単位'
    features_df = compute_daily_total( features_df, target_col_OUT, datetime_col='日時')
    # 在庫水準の計算
    feature_No3_kado_column = '稼働フラグ'
    feature_No3_lt_column = 'かんばん回転日数'
    feature_No3_input_column = 'かんばんの入出差分'
    features_df[feature_No3_input_column] = features_df[target_col_IN_daily] - features_df[target_col_OUT_daily] 
    feature_No3_window = 24*5
    temp = features_df["かんばん回転日数"] = (features_df["サイクル間隔"] * (features_df["サイクル情報"] + 1)) / features_df["サイクル回数"]
    #st.write(temp)
    delay_start = int(temp.max())*24
    delay_end = int(delay_start + feature_No3_window)
    feature_No3_output_column = f'feature_No3_過去のかんばん（t-{delay_start}~t-{delay_end}）'
    features_df = compute_zaiko_level(features_df, feature_No3_kado_column, feature_No3_lt_column,
                                       feature_No3_input_column, feature_No3_window,
                                       feature_No3_output_column)
    #todo
    #todo 色々計算しているけど、期待かんばん在庫数を入れる
    features_df[feature_No3_output_column] = features_df['期待かんばん在庫数']
    # youin列作成
    youin_No3_column = f'youin_No3_過去のかんばん（t-{delay_start}~t-{delay_end}）'
    features_df[youin_No3_column] = features_df[feature_No3_output_column]
    # 結果の確認
    features_df_temp = features_df[['日時', feature_No3_kado_column, feature_No3_input_column,
                                     feature_No3_output_column]]
    st.dataframe(features_df_temp)
    # 結果の確認
    # 結果確認する列をリストとして定義
    value_columns = [feature_No3_input_column, feature_No3_output_column]
    # 結果を出力
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = '過去のかんばん推移', yaxis_title = 'かんばん数', kado_column = feature_No3_kado_column)

    #!-----------------------------------------------------------------------
    #! 物流センターから部品置き場の間の滞留かんばん数を計算する
    #!-----------------------------------------------------------------------
    delay_start = 0
    delay_end = 0
    feature_No4_output_column = f"feature_No4_西尾東~部品置き場の間の滞留かんばん数（t-{delay_start}~t-{delay_end}）"
    features_df[feature_No4_output_column] = features_df["西尾東~部品置き場の間の滞留かんばん数_枚数単位"]
    # youin列作成
    youin_No4_column = f"youin_No4_西尾東~部品置き場の間の滞留かんばん数（t-{delay_start}~t-{delay_end}）"
    features_df[youin_No4_column] = features_df["西尾東~部品置き場の間の滞留かんばん数_枚数単位"]
    # 結果の確認
    features_df_temp = features_df[['日時', feature_No3_kado_column, target_column,
                                     feature_No4_output_column]]
    st.dataframe(features_df_temp)
    # 結果を出力
    value_columns = [feature_No4_output_column]
    plot_result( features_df, target_datetime_column, value_columns, flag_show,
                 graph_title = '滞留かんばん推移', yaxis_title = 'かんばん数', kado_column = feature_No3_output_column)
    #　数値型に変換
    features_df[feature_No1_snap_column] = features_df[feature_No1_snap_column].astype(float)
    features_df[feature_No2_recent_chakkousuu_status_and_trend_column] = features_df[feature_No2_recent_chakkousuu_status_and_trend_column].astype(float)
    features_df[feature_No3_output_column] = features_df[feature_No3_output_column].astype(float)
    features_df[feature_No4_output_column] = features_df[feature_No4_output_column].astype(float)
    features_df[feature_No5_snap_column_abnormal] = features_df[feature_No5_snap_column_abnormal].astype(float)
    features_df[target_column] = features_df[target_column].astype(float)

    # データ調整

    #! 小数点第三で四捨五入（小数多いとナイーブに反応する)
    features_df[feature_No2_recent_chakkousuu_status_and_trend_column] = features_df[feature_No2_recent_chakkousuu_status_and_trend_column].round(2)

    # 特定の列の型を確認
    column_name = feature_No2_recent_chakkousuu_status_and_trend_column  # 確認したい列名
    st.write(f"\n{column_name}の型:")
    st.write(f"データ型: {features_df[column_name].dtype}")

    # 数値列のみを選択してNaN/Noneを0に置換
    #! これしないと平均計算などが期待通りにならない。pythonのmeanはNoneを無視する
    numeric_columns = features_df.select_dtypes(include=['int64', 'float64']).columns
    features_df[numeric_columns] = features_df[numeric_columns].fillna(0)

    figs = create_correlation_plots(features_df, ['feature', 'target'])

    # ヒートマップの表示
    st.subheader("Correlation Heatmap")
    st.plotly_chart(figs['heatmap'], use_container_width=True)

    # 散布図の表示
    st.subheader("Scatter Plots")
    for key, fig in figs.items():
        if key.startswith('scatter'):
            st.plotly_chart(fig, use_container_width=True)
    
    return features_df

# MARK: 単独テスト用
if __name__ == "__main__":

    kojo = 'anjo1'
    hinban_info = ['9014860027', '1Y']
    start_datetime = '2024-10-01 00:00:00'
    end_datetime = '2025-03-12 09:00:00'
    flag_useDataBase = 1
    target_column = '納入予定日時'

    merge_data(hinban_info, start_datetime, end_datetime, flag_useDataBase, kojo)



    


   